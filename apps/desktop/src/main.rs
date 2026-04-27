#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use dictator::audio_capture;
#[cfg(feature = "gpu")]
use dictator::gpu_detect;
use dictator::gpu_guard;
use dictator::injection::{self, InjectionMode};
use dictator::live::{self, LiveEvent};
use dictator::llm;
use dictator::model_manager;
use dictator::power_events;
use dictator::server;
use dictator::tailscale;
use dictator::templates::{ChatFormat, FormatType};
use dictator::history;
use dictator::transcription;
use dictator::tts;
use dictator::user_prefs;
use dictator::wake_word;
use dictator::word_bank;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconEvent};
use tauri::{Emitter, Manager, State};

/// Recording session: a background thread captures audio into a shared buffer.
struct RecordingSession {
    stop_flag: Arc<AtomicBool>,
    samples: Arc<Mutex<Vec<f32>>>,
    /// Maximum audio peak since last read — enables real-time silence detection.
    /// Stored as f32 bits via `to_bits()`; read-and-reset via `swap(0, Relaxed)`.
    peak_level: Arc<AtomicU32>,
    thread: Option<std::thread::JoinHandle<()>>,
    /// If live preview is on, holds the frame sender. Dropping it causes the
    /// preview pipeline to wind down. The event_rx is owned by a detached
    /// drain task that emits Tauri events.
    preview_tx: Option<tokio::sync::mpsc::UnboundedSender<Vec<f32>>>,
}

/// Desktop-mic live-mode session. Holds the handles needed to stop the
/// session cleanly; the event_rx is consumed by a detached tokio task that
/// drains live events and emits them as Tauri events to the frontend.
struct DesktopLiveSession {
    /// Our clone of the live session's audio input channel. Dropping it
    /// (after the mic capture handle is also stopped) causes the session
    /// to see no remaining senders and flush its final utterance.
    frame_tx: tokio::sync::mpsc::UnboundedSender<Vec<f32>>,
    /// Stopping/dropping this tears down the cpal stream.
    mic: audio_capture::LiveCaptureHandle,
}

/// Shared app state accessible from Tauri commands.
struct AppState {
    transcriber: Arc<Mutex<Option<transcription::Transcriber>>>,
    recording: Mutex<Option<RecordingSession>>,
    injection_mode: Arc<Mutex<String>>,
    auto_space: Arc<Mutex<bool>>,
    auto_format: Arc<Mutex<bool>>,
    auto_format_type: Arc<Mutex<String>>,
    model_error: Mutex<Option<String>>,
    server_url: Mutex<Option<String>>,
    connected_phones: Arc<Mutex<u32>>,
    /// "tailscale" or "self-signed"
    cert_type: Mutex<String>,
    /// Tailscale URL if Tailscale is available (works from anywhere)
    tailscale_url: Mutex<Option<String>>,
    /// None while loading, Some(true) for GPU, Some(false) for CPU (Whisper)
    using_gpu: Arc<Mutex<Option<bool>>>,
    /// None while not loaded, Some(true) for GPU, Some(false) for CPU (LLM)
    llm_using_gpu: Arc<Mutex<Option<bool>>>,
    /// User preference: force CPU mode (skip GPU for both Whisper and LLM)
    force_cpu: Arc<Mutex<bool>>,
    /// Selected audio input device name (None = system default)
    selected_device: Arc<Mutex<Option<String>>>,
    /// Current hotkey preset name (e.g. "Ctrl+Shift+Space")
    hotkey: Mutex<String>,
    /// Inject-at-cursor hotkey preset
    inject_hotkey: Mutex<String>,
    /// Read-aloud hotkey preset
    tts_hotkey: Mutex<String>,
    /// Reformat-selection hotkey preset
    reformat_hotkey: Mutex<String>,
    /// Copy-last-transcription-to-clipboard hotkey preset
    copy_hotkey: Mutex<String>,
    /// SAPI TTS speaker (lazy-loaded on first speak)
    speaker: Arc<Mutex<Option<tts::SapiSpeaker>>>,
    /// Piper TTS speaker (neural voices)
    piper: Arc<tts::PiperSpeaker>,
    /// TTS engine: "sapi", "piper", or "minimax"
    tts_engine: Mutex<String>,
    /// Minimax API key for cloud TTS
    minimax_api_key: Mutex<String>,
    /// Minimax voice ID
    minimax_voice: Mutex<String>,
    /// Minimax playback state
    minimax_playing: Arc<AtomicBool>,
    /// Minimax sink for stop control
    minimax_sink: Arc<Mutex<Option<rodio::Sink>>>,
    /// LLM formatter (lazy-loaded on first format request)
    formatter: Arc<Mutex<Option<llm::Formatter>>>,
    /// LLM status: None = not loaded, Some(true) = loaded, Some(false) = failed
    llm_status: Arc<Mutex<Option<bool>>>,
    /// LLM error message if loading failed
    llm_error: Mutex<Option<String>>,
    /// True when no whisper model was found at startup
    no_model: Mutex<bool>,
    /// Whether the Tailscale HTTPS server is running
    tailscale_server_running: Arc<AtomicBool>,
    /// Handle for gracefully shutting down the LAN server. Present while the
    /// server is running; cleared when stopped.
    lan_server_handle: Mutex<Option<axum_server::Handle>>,
    /// Shared server state pieces needed to start Tailscale server dynamically
    server_shared: Mutex<Option<ServerShared>>,
    /// When the Tailscale cert was last generated (epoch secs), for expiry tracking
    tailscale_cert_generated: Mutex<Option<u64>>,
    /// Custom template instructions (overrides defaults in templates.rs)
    custom_templates: Arc<Mutex<HashMap<String, String>>>,
    /// Extra user guidance appended to the Strict cleanup prompt. Persisted
    /// to disk so notes like "focus on action items" or "preserve technical
    /// terms exactly" carry across sessions. Empty string = no guidance
    /// (default behaviour, same as before the feature existed).
    strict_cleanup_notes: Arc<Mutex<String>>,
    /// Cross-segment Whisper context for live mode (feeds previous
    /// segment's text as initial_prompt to the next segment).
    live_context: Arc<Mutex<bool>>,
    /// Live preview: show VAD-segmented preview text during regular
    /// recordings. Purely visual — the final text always comes from
    /// the full-audio Whisper pass.
    live_preview: Arc<Mutex<bool>>,
    /// Active desktop-mic live session, if any.
    live_session: Mutex<Option<DesktopLiveSession>>,
    /// Silence-auto-stop: end the recording automatically after the user
    /// has stopped talking for the configured number of seconds. Off by
    /// default; UI restores the saved preference on startup.
    auto_stop_enabled: Arc<Mutex<bool>>,
    /// Silence span (in seconds) after which auto-stop fires. Tunable per
    /// user — too short clips mid-thought, too long delays transcription.
    auto_stop_secs: Arc<Mutex<f32>>,
    /// User-configurable wake phrase (case-insensitive substring match
    /// against live transcripts). Defaults to "hey dictator". The "is
    /// the wake word listener currently active?" question is answered
    /// by `wake_word_handle.is_some()` rather than a separate flag —
    /// the UI restores its enabled-or-not preference from localStorage.
    wake_word_phrase: Arc<Mutex<String>>,
    /// rustpotter detection threshold in 0.0..1.0. 0.45 default — lower
    /// = looser (catches mumbled / quiet pronunciations, more false
    /// positives), higher = stricter. Read at listener-start time, so
    /// the user has to toggle off/on for a slider change to apply.
    wake_word_threshold: Arc<Mutex<f32>>,
    /// Active wake-word listener, if any. Holds both the live session and
    /// the cpal capture so dropping the field cleanly releases the mic.
    wake_word_handle: Mutex<Option<WakeWordHandle>>,
}

/// Resources that live for the duration of a wake-word listener. Dropping
/// the struct ends the cpal capture (frees the mic), which closes the
/// frame channel and lets the matcher task exit naturally.
struct WakeWordHandle {
    _capture: audio_capture::LiveCaptureHandle,
}

/// Shared references needed to construct a ServerState for the dynamic
/// LAN + Tailscale servers. Cloned out of AppState's mutex each time a
/// server is (re)started, so the fields are all cheap Arc clones.
#[derive(Clone)]
struct ServerShared {
    transcriber: Arc<Mutex<Option<transcription::Transcriber>>>,
    formatter: Arc<Mutex<Option<llm::Formatter>>>,
    injection_mode: Arc<Mutex<String>>,
    connected_phones: Arc<Mutex<u32>>,
    auto_space: Arc<Mutex<bool>>,
    auto_format: Arc<Mutex<bool>>,
    auto_format_type: Arc<Mutex<String>>,
    using_gpu: Arc<Mutex<Option<bool>>>,
    force_cpu: Arc<Mutex<bool>>,
    live_context: Arc<Mutex<bool>>,
    tts_trigger: std::sync::mpsc::Sender<()>,
}

const LAN_PORT: u16 = 3456;
const TAILSCALE_PORT: u16 = 3457;

// ── Tauri commands ──────────────────────────────────────────────────────

#[tauri::command]
fn get_status(state: State<AppState>) -> String {
    if let Some(err) = state.model_error.lock().unwrap().as_ref() {
        return format!("error:{err}");
    }

    if *state.no_model.lock().unwrap() {
        return "no-model".into();
    }

    let has_model = state.transcriber.lock().unwrap().is_some();
    let is_recording = state.recording.lock().unwrap().is_some();

    if is_recording {
        "recording".into()
    } else if has_model {
        "ready".into()
    } else {
        "loading".into()
    }
}

#[tauri::command]
fn get_server_url(state: State<AppState>) -> Option<String> {
    state.server_url.lock().unwrap().clone()
}

#[tauri::command]
fn get_connected_phones(state: State<AppState>) -> u32 {
    *state.connected_phones.lock().unwrap()
}

#[tauri::command]
fn get_cert_type(state: State<AppState>) -> String {
    state.cert_type.lock().unwrap().clone()
}

#[tauri::command]
fn get_tailscale_url(state: State<AppState>) -> Option<String> {
    state.tailscale_url.lock().unwrap().clone()
}

/// Reusable Tailscale refresh logic — called by Tauri command, startup, and cert renewal.
fn invoke_refresh_tailscale(app_handle: &tauri::AppHandle) -> Result<String, String> {
    let state: State<AppState> = app_handle.state();
    let was_running = state.tailscale_server_running.load(Ordering::Relaxed);

    let ts = match tailscale::detect() {
        tailscale::DetectResult::Ok(ts) => ts,
        tailscale::DetectResult::NotInstalled => {
            *state.tailscale_url.lock().unwrap() = None;
            *state.tailscale_cert_generated.lock().unwrap() = None;
            let _ = app_handle.emit("tailscale-changed", "");
            return Err("Tailscale is not installed. Install it from tailscale.com on both your PC and phone.".to_string());
        }
        tailscale::DetectResult::NotRunning(s) => {
            *state.tailscale_url.lock().unwrap() = None;
            *state.tailscale_cert_generated.lock().unwrap() = None;
            let _ = app_handle.emit("tailscale-changed", "");
            return Err(format!("Tailscale is installed but not connected (state: {s}). Open Tailscale and sign in."));
        }
        tailscale::DetectResult::NoHttps => {
            *state.tailscale_url.lock().unwrap() = None;
            *state.tailscale_cert_generated.lock().unwrap() = None;
            let _ = app_handle.emit("tailscale-changed", "");
            return Err("Tailscale is running but HTTPS is not enabled. Go to login.tailscale.com/admin/dns, scroll to HTTPS Certificates, and click Enable.".to_string());
        }
    };

    let url = format!("https://{}:{}", ts.cert_domain, TAILSCALE_PORT);
    let cert_dir = std::env::temp_dir().join("dictator-certs");
    let _ = std::fs::create_dir_all(&cert_dir);
    let cert_path = cert_dir.join("cert.pem");
    let key_path = cert_dir.join("key.pem");

    let (cert_pem, key_pem) = match tailscale::generate_cert(&ts.cert_domain, &cert_path, &key_path) {
        Ok(pair) => pair,
        Err(e) => {
            *state.tailscale_url.lock().unwrap() = None;
            let _ = app_handle.emit("tailscale-changed", "");
            let hint = if e.contains("Access") || e.contains("denied") || e.contains("permission") {
                "Try running Dictator as Administrator."
            } else {
                "Make sure HTTPS certificates are enabled at login.tailscale.com/admin/dns"
            };
            return Err(format!("Certificate failed: {hint}"));
        }
    };

    // Track when the cert was generated
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    *state.tailscale_cert_generated.lock().unwrap() = Some(now);
    *state.tailscale_url.lock().unwrap() = Some(url.clone());

    eprintln!("Tailscale cert OK for {}", ts.cert_domain);

    if !was_running {
        // Start the Tailscale server
        let shared = state.server_shared.lock().unwrap();
        if let Some(ref ss) = *shared {
            let ts_state = Arc::new(server::ServerState {
                transcriber: ss.transcriber.clone(),
                formatter: ss.formatter.clone(),
                injection_mode: ss.injection_mode.clone(),
                connected_phones: ss.connected_phones.clone(),
                auto_space: ss.auto_space.clone(),
                auto_format: ss.auto_format.clone(),
                auto_format_type: ss.auto_format_type.clone(),
                cert_type: "tailscale".to_string(),
                using_gpu: ss.using_gpu.clone(),
                force_cpu: ss.force_cpu.clone(),
                live_context: ss.live_context.clone(),
                tts_trigger: Some(ss.tts_trigger.clone()),
            });

            let running_flag = state.tailscale_server_running.clone();
            std::thread::spawn(move || {
                let rt = tokio::runtime::Runtime::new()
                    .expect("Failed to create tokio runtime for Tailscale");
                running_flag.store(true, Ordering::Relaxed);
                rt.block_on(async move {
                    server::run_server(
                        ts_state,
                        TAILSCALE_PORT,
                        server::TlsSource::Tailscale { cert_pem, key_pem },
                        None,
                    )
                    .await;
                });
                running_flag.store(false, Ordering::Relaxed);
            });

            eprintln!("Tailscale server started on port {TAILSCALE_PORT}");
        }
    }
    // Note: if server was already running, the cert renewal doesn't hot-swap.
    // The server continues with the old cert until next restart. This is fine
    // because tailscale cert renews well before expiry (certs are 90 days,
    // renewal happens ~30 days before).

    let _ = app_handle.emit("tailscale-changed", url.as_str());
    Ok(format!("Tailscale connected: {url}"))
}

#[tauri::command]
fn refresh_tailscale(app_handle: tauri::AppHandle) -> Result<String, String> {
    invoke_refresh_tailscale(&app_handle)
}

/// Start the LAN phone-companion server. Idempotent — calling while already
/// running is a no-op that returns the current URL. Rebuilds the
/// `ServerState` from `server_shared` stored at startup, generates a fresh
/// self-signed cert covering current local IPs, and spawns a dedicated
/// tokio runtime to host axum.
#[tauri::command]
fn start_lan_server(app_handle: tauri::AppHandle) -> Result<String, String> {
    let state: State<AppState> = app_handle.state();

    // Already running? Return the cached URL.
    if state.lan_server_handle.lock().unwrap().is_some() {
        if let Some(url) = state.server_url.lock().unwrap().clone() {
            return Ok(url);
        }
    }

    // Pull the shared server state that was stashed at startup.
    let shared = match state.server_shared.lock().unwrap().clone() {
        Some(s) => s,
        None => return Err("Server state not initialized".to_string()),
    };

    let ips = server::get_local_ips();
    let url = format!("https://{}:{}", ips[0], LAN_PORT);

    let handle = axum_server::Handle::new();
    *state.lan_server_handle.lock().unwrap() = Some(handle.clone());
    *state.server_url.lock().unwrap() = Some(url.clone());

    let app_handle_thread = app_handle.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime for LAN server");
        rt.block_on(async move {
            let lan_state = std::sync::Arc::new(server::ServerState {
                transcriber: shared.transcriber,
                formatter: shared.formatter,
                injection_mode: shared.injection_mode,
                connected_phones: shared.connected_phones,
                auto_space: shared.auto_space,
                auto_format: shared.auto_format,
                auto_format_type: shared.auto_format_type,
                cert_type: "self-signed".to_string(),
                using_gpu: shared.using_gpu,
                force_cpu: shared.force_cpu,
                live_context: shared.live_context,
                tts_trigger: Some(shared.tts_trigger),
            });
            server::run_server(
                lan_state,
                LAN_PORT,
                server::TlsSource::SelfSigned { local_ips: ips },
                Some(handle),
            )
            .await;
        });
        // Server returned — clear state so UI reflects stopped state.
        let s: State<AppState> = app_handle_thread.state();
        *s.lan_server_handle.lock().unwrap() = None;
        *s.server_url.lock().unwrap() = None;
        *s.connected_phones.lock().unwrap() = 0;
        let _ = app_handle_thread.emit("lan-server-stopped", ());
    });

    eprintln!("LAN server started on port {LAN_PORT} ({url})");
    let _ = app_handle.emit("lan-server-started", url.clone());
    Ok(url)
}

/// Stop the LAN server gracefully. No-op if not running.
#[tauri::command]
fn stop_lan_server(state: State<AppState>) -> Result<(), String> {
    let handle_opt = state.lan_server_handle.lock().unwrap().take();
    if let Some(handle) = handle_opt {
        handle.graceful_shutdown(Some(std::time::Duration::from_secs(2)));
        eprintln!("LAN server shutdown signalled");
    }
    *state.server_url.lock().unwrap() = None;
    Ok(())
}

/// Whether the LAN companion server is currently accepting connections.
#[tauri::command]
fn get_lan_server_running(state: State<AppState>) -> bool {
    state.lan_server_handle.lock().unwrap().is_some()
}

/// Returns Tailscale status info: url, running, days until cert expiry.
#[tauri::command]
fn get_tailscale_status(state: State<AppState>) -> serde_json::Value {
    let url = state.tailscale_url.lock().unwrap().clone();
    let running = state.tailscale_server_running.load(Ordering::Relaxed);
    let cert_generated = *state.tailscale_cert_generated.lock().unwrap();

    let days_remaining = cert_generated.map(|generated_at| {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let elapsed_days = (now.saturating_sub(generated_at)) / 86400;
        90_u64.saturating_sub(elapsed_days) // Let's Encrypt certs are 90 days
    });

    serde_json::json!({
        "url": url,
        "running": running,
        "days_remaining": days_remaining,
    })
}

#[tauri::command]
fn get_auto_space(state: State<AppState>) -> bool {
    *state.auto_space.lock().unwrap()
}

#[tauri::command]
fn set_auto_space(state: State<AppState>, enabled: bool) {
    *state.auto_space.lock().unwrap() = enabled;
}

#[tauri::command]
fn get_auto_format(state: State<AppState>) -> bool {
    *state.auto_format.lock().unwrap()
}

#[tauri::command]
fn set_auto_format(state: State<AppState>, enabled: bool) {
    *state.auto_format.lock().unwrap() = enabled;
}

#[tauri::command]
fn get_auto_format_type(state: State<AppState>) -> String {
    state.auto_format_type.lock().unwrap().clone()
}

#[tauri::command]
fn set_auto_format_type(state: State<AppState>, format_type: String) {
    *state.auto_format_type.lock().unwrap() = format_type;
}

#[tauri::command]
fn get_live_context(state: State<AppState>) -> bool {
    *state.live_context.lock().unwrap()
}

#[tauri::command]
fn set_live_context(state: State<AppState>, enabled: bool) {
    *state.live_context.lock().unwrap() = enabled;
}

#[tauri::command]
fn get_live_preview(state: State<AppState>) -> bool {
    *state.live_preview.lock().unwrap()
}

#[tauri::command]
fn set_live_preview(state: State<AppState>, enabled: bool) {
    *state.live_preview.lock().unwrap() = enabled;
}

#[tauri::command]
fn get_gpu_status(state: State<AppState>) -> serde_json::Value {
    let whisper_gpu = *state.using_gpu.lock().unwrap();
    let llm_gpu = *state.llm_using_gpu.lock().unwrap();
    let force_cpu = *state.force_cpu.lock().unwrap();

    // Pre-flight detection result. Two tiers:
    // - `hardware_gpu_available`: good enough for Whisper (legit GPU at all)
    // - `llm_gpu_capable`: good enough for LLM (discrete GPU or ≥4 GB iGPU)
    // The UI uses both to distinguish "no GPU" from "GPU disabled after
    // crash" and "Whisper GPU but LLM will stay on CPU".
    #[cfg(feature = "gpu")]
    let (hardware_gpu_available, llm_gpu_capable, gpu_summary) = {
        let d = gpu_detect::detect();
        (d.is_usable(), d.is_suitable_for_llm(), d.summary())
    };
    #[cfg(not(feature = "gpu"))]
    let (hardware_gpu_available, llm_gpu_capable, gpu_summary): (bool, bool, String) =
        (false, false, "GPU not compiled in".to_string());

    serde_json::json!({
        "whisper_gpu": whisper_gpu,
        "llm_gpu": llm_gpu,
        "force_cpu": force_cpu,
        // True if this session started with a crash marker from a previous run.
        // The UI uses this to show a "Try GPU Again" affordance.
        "crash_recovered": gpu_guard::crash_recovered(),
        // True if GPU is currently being skipped because of that crash marker.
        // Cleared once the user clicks "Try GPU Again".
        "gpu_crash_skipped": gpu_guard::session_disabled(),
        // True if the hardware itself has a usable Vulkan GPU (passed our
        // pre-flight filters). False means no GPU, only a software renderer,
        // or not enough VRAM — "Try GPU Again" won't help those cases.
        "hardware_gpu_available": hardware_gpu_available,
        // True if the detected GPU is also strong enough for LLM offload
        // (discrete GPU or integrated with ≥4 GB VRAM). False on e.g.
        // Intel UHD: Whisper GPU is fine but LLM stays on CPU.
        "llm_gpu_capable": llm_gpu_capable,
        // Human-readable summary of the detected GPU (or reason it was
        // rejected). Safe to surface in the UI.
        "gpu_summary": gpu_summary,
        // Keep "active" for backward compat with companion page
        "active": whisper_gpu,
        // True if this build was compiled with the `gpu` feature. A false here
        // means the "No compatible GPU detected" UI branch should be skipped
        // (there's no detection to report) and GPU toggles should be hidden.
        "compiled": cfg!(feature = "gpu"),
    })
}

#[tauri::command]
fn get_force_cpu(state: State<AppState>) -> bool {
    *state.force_cpu.lock().unwrap()
}

// ── Silence-auto-stop & wake-word settings ────────────────────────────
// Both features are read every time `start_recording` (and the wake-word
// listener) fires, so we just expose plain getters/setters here. The UI
// is the source of truth — it persists to localStorage and pushes the
// values down on settings change.

#[tauri::command]
fn get_auto_stop(state: State<AppState>) -> bool {
    *state.auto_stop_enabled.lock().unwrap()
}

#[tauri::command]
fn set_auto_stop(state: State<AppState>, enabled: bool) {
    *state.auto_stop_enabled.lock().unwrap() = enabled;
}

#[tauri::command]
fn get_auto_stop_secs(state: State<AppState>) -> f32 {
    *state.auto_stop_secs.lock().unwrap()
}

#[tauri::command]
fn set_auto_stop_secs(state: State<AppState>, secs: f32) {
    // Clamp to a sensible range so a malformed UI value can't trap users
    // in either "auto-stop never fires" or "auto-stop fires mid-word".
    let clamped = secs.clamp(0.5, 10.0);
    *state.auto_stop_secs.lock().unwrap() = clamped;
}

#[tauri::command]
fn get_wake_word_threshold(state: State<AppState>) -> f32 {
    *state.wake_word_threshold.lock().unwrap()
}

#[tauri::command]
fn set_wake_word_threshold(state: State<AppState>, threshold: f32) {
    // Clamp away from the extremes — values near 0 instantly fire on
    // noise; values near 1 mean the listener will never detect anything.
    *state.wake_word_threshold.lock().unwrap() = threshold.clamp(0.1, 0.95);
}

#[tauri::command]
fn list_wake_word_phrases() -> Vec<serde_json::Value> {
    wake_word::WakePhrase::all()
        .iter()
        .map(|p| serde_json::json!({ "id": p.id(), "label": p.label() }))
        .collect()
}

#[tauri::command]
fn get_wake_word_phrase_id(state: State<AppState>) -> String {
    state.wake_word_phrase.lock().unwrap().clone()
}

#[tauri::command]
fn set_wake_word_phrase_id(state: State<AppState>, phrase_id: String) {
    if wake_word::WakePhrase::from_id(&phrase_id).is_some() {
        *state.wake_word_phrase.lock().unwrap() = phrase_id;
    }
}

/// Start an always-on wake-word listener using OpenWakeWord's pre-trained
/// ONNX pipeline (mel spectrogram → embedding → keyword classifier). The
/// keyword model is selected by the user's stored phrase id.
///
/// On match: emits `wake-word-triggered` and tears down its own handle so
/// the mic is free by the time the UI fires `start_recording`.
#[tauri::command]
async fn start_wake_word_listener(
    state: State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    // Idempotent: multiple invocations (e.g. UI restoring on startup +
    // toggle change) should be a no-op rather than spawn duplicates.
    if state.wake_word_handle.lock().unwrap().is_some() {
        return Ok(());
    }

    // Refusing to start while a recording is in flight avoids fighting over
    // the mic. The UI restarts the listener after stop_and_transcribe.
    if state.recording.lock().unwrap().is_some() {
        return Err("Cannot start wake-word listener while recording".into());
    }

    // Resolve the phrase + threshold up front so we can fail with a
    // friendly message before any cpal stream is opened.
    let phrase_id = state.wake_word_phrase.lock().unwrap().clone();
    let phrase = wake_word::WakePhrase::from_id(&phrase_id)
        .ok_or_else(|| format!("Unknown wake phrase '{phrase_id}'"))?;
    let threshold = *state.wake_word_threshold.lock().unwrap();

    let mut detector = wake_word::WakeWordDetector::new(phrase)
        .map_err(|e| format!("Failed to build wake-word detector: {e}"))?;

    let (frame_tx, mut frame_rx) = tokio::sync::mpsc::unbounded_channel::<Vec<f32>>();
    let device_name = state.selected_device.lock().unwrap().clone();
    let capture = audio_capture::stream_to_live(frame_tx, device_name)
        .map_err(|e| format!("Failed to start wake-word capture: {e}"))?;

    let ah = app_handle.clone();
    tokio::spawn(async move {
        let mut last_emit = std::time::Instant::now();
        let _ = ah.emit("wake-word-threshold", threshold);
        // Warmup: still useful as a UI signal even though OpenWakeWord
        // does its own first-5-zero suppression. Covers the WASAPI
        // settling window and any post-recording paste/keyboard
        // artifacts when the listener re-arms after a dictation.
        let warmup_until = std::time::Instant::now()
            + std::time::Duration::from_millis(2500);
        let mut warmup_done_emitted = false;
        while let Some(samples) = frame_rx.recv().await {
            // OpenWakeWord pipeline runs on edge cases; wrap so that a
            // panic in ort or our buffering math leaves a clean exit
            // rather than a dead task with no logging.
            let result = std::panic::catch_unwind(
                std::panic::AssertUnwindSafe(|| detector.push_samples(&samples)),
            );
            let score_opt = match result {
                Ok(Ok(s)) => s,
                Ok(Err(e)) => {
                    eprintln!("wake-word: inference error: {e}");
                    None
                }
                Err(_) => {
                    eprintln!("wake-word: pipeline panicked — disabling listener");
                    let st: State<AppState> = ah.state();
                    let old = {
                        let mut guard = st.wake_word_handle.lock().unwrap();
                        guard.take()
                    };
                    drop(old);
                    return;
                }
            };
            let now = std::time::Instant::now();
            if !warmup_done_emitted && now >= warmup_until {
                warmup_done_emitted = true;
                let _ = ah.emit("wake-word-warmed-up", ());
            }
            // Always emit the latest score (throttled) so the UI gauge
            // stays alive — even during warmup, so users can see the
            // pipeline is processing audio.
            if let Some(s) = score_opt {
                if last_emit.elapsed() > std::time::Duration::from_millis(200) {
                    let _ = ah.emit("wake-word-score", s);
                    last_emit = now;
                }
                // Suppress real fires until warmup is complete.
                if now < warmup_until {
                    continue;
                }
                if s >= threshold {
                    eprintln!("wake-word: matched (score {:.2})", s);
                    // Tear down BEFORE emitting — the drop blocks until
                    // cpal hands the WASAPI handle back, so by the time
                    // the UI receives the event the mic is free.
                    let st: State<AppState> = ah.state();
                    let old = {
                        let mut guard = st.wake_word_handle.lock().unwrap();
                        guard.take()
                    };
                    drop(old);
                    let _ = ah.emit("wake-word-triggered", ());
                    return;
                }
            }
        }
    });

    *state.wake_word_handle.lock().unwrap() = Some(WakeWordHandle { _capture: capture });
    eprintln!("wake-word: listener started ({})", phrase.id());
    Ok(())
}

/// Stop the wake-word listener (if running). Idempotent.
#[tauri::command]
fn stop_wake_word_listener(state: State<AppState>) {
    let handle = state.wake_word_handle.lock().unwrap().take();
    if handle.is_some() {
        eprintln!("wake-word: listener stopped");
    }
    // Drop happens at end of scope — releases mic, closes frame_tx →
    // matcher task exits when its receiver returns None.
}

#[tauri::command]
fn get_wake_word_listener_running(state: State<AppState>) -> bool {
    state.wake_word_handle.lock().unwrap().is_some()
}

#[tauri::command]
fn set_force_cpu(state: State<AppState>, force: bool) {
    // Idempotent: the UI calls this on startup to sync localStorage → backend,
    // and we don't want that sync to tear down a freshly-loaded model. Only
    // drop loaded models when the value is genuinely changing.
    {
        let mut current = state.force_cpu.lock().unwrap();
        if *current == force {
            return;
        }
        *current = force;
    }
    // Persist so the next launch's startup loader picks the right mode
    // without relying on the webview's localStorage sync to flip it.
    user_prefs::save_force_cpu(force);
    // The preference actually changed — clear loaded models so they reload
    // with the new GPU preference. VRAM is reclaimed immediately when
    // toggling ON since the GPU-backed models are the only strong refs.
    *state.transcriber.lock().unwrap() = None;
    *state.formatter.lock().unwrap() = None;
    *state.using_gpu.lock().unwrap() = None;
    *state.llm_using_gpu.lock().unwrap() = None;
    *state.llm_status.lock().unwrap() = None;
}

// ── Word bank (Whisper replacement dictionary) ──────────────────────────
// Loaded from / saved to disk on each call. The bank is tiny (dozens of
// entries at most) and edits happen in Settings, so there's no need to
// cache it in AppState — reading the JSON file costs nothing relative to
// a transcription run.

#[tauri::command]
fn get_word_bank() -> Vec<word_bank::Entry> {
    word_bank::load()
}

#[tauri::command]
fn set_word_bank(entries: Vec<word_bank::Entry>) -> Result<(), String> {
    // Trim whitespace on both sides so UI blank-row artifacts don't persist.
    let cleaned: Vec<word_bank::Entry> = entries
        .into_iter()
        .map(|e| word_bank::Entry {
            from: e.from.trim().to_string(),
            to: e.to.trim().to_string(),
        })
        .filter(|e| !e.from.is_empty()) // drop fully-blank rows
        .collect();
    word_bank::save(&cleaned)
}

// ── Strict cleanup notes (user guidance appended to the strict prompt) ──
// Small persistent string shown in Settings. Applied by format_text and
// reformat_selection when the active format is StrictCleanUp. Empty string
// = no change from the default prompt, which is the v0.4.0 behaviour.

fn strict_cleanup_notes_path() -> std::path::PathBuf {
    model_manager::models_dir().join("strict_cleanup_notes.txt")
}

fn load_strict_cleanup_notes() -> String {
    std::fs::read_to_string(strict_cleanup_notes_path()).unwrap_or_default()
}

#[tauri::command]
fn get_strict_cleanup_notes(state: State<AppState>) -> String {
    state.strict_cleanup_notes.lock().unwrap().clone()
}

#[tauri::command]
fn set_strict_cleanup_notes(state: State<AppState>, notes: String) -> Result<(), String> {
    // Trim only the outer whitespace — internal newlines (which the user may
    // use to separate points) are kept intentionally. We DO persist an empty
    // string (as "") rather than deleting the file, so load is stable.
    let value = notes.trim().to_string();
    std::fs::write(strict_cleanup_notes_path(), &value)
        .map_err(|e| format!("Failed to save strict cleanup notes: {e}"))?;
    *state.strict_cleanup_notes.lock().unwrap() = value;
    Ok(())
}

/// Merge the user's strict-cleanup notes into the system instruction, but
/// only when the format is `StrictCleanUp` and notes are non-empty. Returns
/// the owned instruction to pass into `format_text_custom` (or `None` to
/// fall back to whatever the caller would have used).
///
/// `base` is either the user's custom template override (if any) or `None`
/// meaning "use the default"; we materialise a string only when we actually
/// need to append notes.
///
/// **Why notes go first:** small local LLMs (Phi-3 Mini etc.) latch onto
/// early instructions more reliably than trailing ones, and the default
/// strict-cleanup prompt ends with a hard "IMPORTANT: your entire response
/// must be the rewritten text and nothing else" block. Appending user notes
/// after that made the model treat them as optional footnotes. Inverting
/// the order — user instructions first with explicit priority framing,
/// default prompt below — puts the user's intent in primary position and
/// visibly instructs the model to resolve conflicts in the user's favour.
fn apply_strict_notes(
    format_type: FormatType,
    base: Option<String>,
    notes: &str,
) -> Option<String> {
    let notes_trimmed = notes.trim();
    if format_type != FormatType::StrictCleanUp || notes_trimmed.is_empty() {
        return base;
    }
    let base_str = base.unwrap_or_else(|| format_type.default_instruction().to_string());
    Some(format!(
        "USER INSTRUCTIONS (these take priority over any conflicting defaults below):\n\
         {notes_trimmed}\n\n\
         ---\n\n\
         {base_str}"
    ))
}

// ── History (persistent transcription log) ──────────────────────────────
// The frontend calls add_history_entry at the end of each successful
// dictation flow, once it knows both the raw Whisper output (returned
// from stop_and_transcribe) and the final text (same as raw if no
// format was applied, otherwise the LLM-formatted result). Keeping this
// frontend-driven means we don't need to plumb format results back
// through the backend — the frontend already has everything.

#[tauri::command]
fn get_history() -> Vec<history::Entry> {
    history::load()
}

#[tauri::command]
fn add_history_entry(
    raw: String,
    final_text: String,
    format_type: String,
) -> Result<(), String> {
    history::append(&raw, &final_text, &format_type).map(|_| ())
}

#[tauri::command]
fn delete_history_entry(id: u64) -> Result<(), String> {
    history::delete(id)
}

#[tauri::command]
fn clear_history() -> Result<(), String> {
    history::clear()
}

/// Let the user re-enable GPU after a crash-recovery skip. Clears the guard's
/// session-disabled flag, unloads any loaded models, and kicks off a background
/// warm-up of the whisper model so the UI gets immediate feedback that GPU is
/// coming back online. Without the warm-up, the status row would stay in
/// "Loading..." limbo until the user's next dictation triggers the lazy reload.
///
/// The LLM is intentionally left to lazy-load on first format — warming it up
/// here would serialize two heavy loads and make the click feel slow.
#[tauri::command]
fn retry_gpu(app_handle: tauri::AppHandle, state: State<AppState>) {
    gpu_guard::allow_gpu_retry();
    *state.transcriber.lock().unwrap() = None;
    *state.formatter.lock().unwrap() = None;
    *state.using_gpu.lock().unwrap() = None;
    *state.llm_using_gpu.lock().unwrap() = None;
    *state.llm_status.lock().unwrap() = None;

    // Background warm-up — reload the whisper model now so the UI's
    // "Loading model..." actually corresponds to something loading.
    let transcriber = state.transcriber.clone();
    let using_gpu = state.using_gpu.clone();
    let force_cpu = state.force_cpu.clone();
    let handle = app_handle.clone();
    std::thread::spawn(move || {
        let model_path = match model_manager::find_whisper_model() {
            Some(p) => p,
            None => {
                eprintln!("retry_gpu warm-up: no whisper model found");
                return;
            }
        };
        let fc = *force_cpu.lock().unwrap();
        eprintln!(
            "retry_gpu warm-up: reloading whisper model from {}",
            model_path.display()
        );
        match transcription::Transcriber::new(&model_path, fc) {
            Ok(t) => {
                *using_gpu.lock().unwrap() = Some(t.is_using_gpu());
                *transcriber.lock().unwrap() = Some(t);
                let _ = handle.emit("model-ready", "ok");
            }
            Err(e) => {
                // Soft failure — stay on CPU. Intentionally do NOT emit
                // `model-ready` with an error payload: the UI's existing
                // listener would surface it as a generic error toast. The
                // frontend's status poll will pick up Some(false) and settle
                // on "Running on CPU" within a second.
                eprintln!("retry_gpu warm-up: failed to load model: {e}");
                *using_gpu.lock().unwrap() = Some(false);
            }
        }
    });
}

#[tauri::command]
async fn start_recording(
    state: State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    let mut rec_guard = state.recording.lock().unwrap();
    if rec_guard.is_some() {
        return Err("Already recording".into());
    }

    let stop_flag = Arc::new(AtomicBool::new(false));
    let samples: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let peak_level = Arc::new(AtomicU32::new(0));

    let stop_clone = stop_flag.clone();
    let samples_clone = samples.clone();
    let peak_clone = peak_level.clone();
    let device_name = state.selected_device.lock().unwrap().clone();

    // Optionally spawn a live preview pipeline that runs alongside the
    // recording. Audio is forked from the same mic callback into the VAD;
    // preview segments are emitted as Tauri events for the UI to show
    // in the transcript area. The final transcription still comes from the
    // full-audio pass after recording stops — the preview is purely visual.
    let live_preview = *state.live_preview.lock().unwrap();
    let preview_tx = if live_preview {
        let cross_ctx = true; // preview always uses cross-segment context
        match live::spawn(state.transcriber.clone(), cross_ctx) {
            Ok(live::LiveSessionHandle { frame_tx, mut event_rx }) => {
                // Detach a drain task that forwards preview events to the UI.
                let ah = app_handle.clone();
                tokio::spawn(async move {
                    while let Some(ev) = event_rx.recv().await {
                        match ev {
                            LiveEvent::SegmentStart => {
                                let _ = ah.emit("preview-segment-start", ());
                            }
                            LiveEvent::Segment { text, .. } => {
                                let _ = ah.emit("preview-segment", &text);
                            }
                            LiveEvent::SegmentError(e) => {
                                eprintln!("preview: segment error: {e}");
                            }
                            LiveEvent::Ended => break,
                        }
                    }
                });
                Some(frame_tx)
            }
            Err(e) => {
                eprintln!("preview: failed to start live pipeline: {e}");
                None
            }
        }
    } else {
        None
    };

    // Optionally spawn a silence-auto-stop watchdog. Audio is forked from
    // the mic callback into a dedicated VAD configured with the user's
    // silence threshold; once it sees end-of-speech it emits a Tauri event
    // that the UI handles by triggering the same stop path as a hotkey
    // press. Independent of the live-preview pipeline so users can run
    // either, both, or neither.
    let auto_stop_enabled = *state.auto_stop_enabled.lock().unwrap();
    let auto_stop_secs = *state.auto_stop_secs.lock().unwrap();
    let silence_tx = if auto_stop_enabled && auto_stop_secs > 0.0 {
        let chunks = dictator::vad::silence_chunks_for_secs(auto_stop_secs);
        match dictator::vad::Vad::with_silence_chunks(chunks) {
            Ok(mut watchdog) => {
                let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Vec<f32>>();
                let ah = app_handle.clone();
                // No-speech timeout: if the recording starts (e.g. via
                // wake word) but the user never actually says anything,
                // we want to clean up rather than sit there forever.
                // Heuristic: 2× the user's silence span, with a 2 s floor
                // so a 0.5 s slider doesn't make this trigger before the
                // user can draw breath.
                let no_speech_timeout = std::time::Duration::from_secs_f32(
                    (auto_stop_secs * 2.0).max(2.0),
                );
                tokio::spawn(async move {
                    use dictator::vad::VadEvent;
                    let mut speech_seen = false;
                    let started = std::time::Instant::now();
                    while let Some(samples) = rx.recv().await {
                        // Cancel if nothing was ever said within the
                        // grace window. Distinct event from auto-stop —
                        // the UI maps this to `cancel_recording`, not
                        // `stop_and_transcribe`, so no garbage gets
                        // transcribed and injected.
                        if !speech_seen && started.elapsed() > no_speech_timeout {
                            let _ = ah.emit("auto-cancel-recording", ());
                            return;
                        }
                        watchdog.push_samples(&samples);
                        for ev in watchdog.drain_events() {
                            match ev {
                                VadEvent::Start => speech_seen = true,
                                VadEvent::End { .. } => {
                                    if speech_seen {
                                        let _ = ah.emit("auto-stop-recording", ());
                                        return;
                                    }
                                }
                            }
                        }
                    }
                });
                Some(tx)
            }
            Err(e) => {
                eprintln!("auto-stop: VAD init failed: {e} — auto-stop disabled this session");
                None
            }
        }
    } else {
        None
    };

    let ptx = preview_tx.clone();
    let thread = std::thread::spawn(move || {
        match audio_capture::record_until_stopped(
            stop_clone,
            device_name,
            peak_clone,
            ptx,
            silence_tx,
        ) {
            Ok(recorded) => {
                *samples_clone.lock().unwrap() = recorded;
            }
            Err(e) => {
                eprintln!("Recording error: {e}");
            }
        }
    });

    *rec_guard = Some(RecordingSession {
        stop_flag,
        samples,
        peak_level,
        thread: Some(thread),
        preview_tx,
    });

    Ok(())
}

#[tauri::command]
fn stop_and_transcribe(state: State<AppState>, app_handle: tauri::AppHandle) -> Result<TranscriptionResult, String> {
    let session = {
        let mut guard = state.recording.lock().unwrap();
        guard.take().ok_or("Not recording")?
    };

    session.stop_flag.store(true, Ordering::Relaxed);

    if let Some(thread) = session.thread {
        thread.join().map_err(|_| "Recording thread panicked")?;
    }

    // Kill the preview pipeline (if any) before starting the full Whisper
    // pass. Dropping the sender causes the session to wind down. Any
    // in-flight preview segment finishes quickly and releases the Whisper
    // mutex.
    drop(session.preview_tx);

    let samples = Arc::try_unwrap(session.samples)
        .map_err(|_| "Failed to get samples")?
        .into_inner()
        .unwrap();

    let duration = samples.len() as f64 / 16000.0;

    // Gentle handling for too-short recordings
    if duration < 0.3 {
        return Err("hint:Too short \u{2014} hold to record".into());
    }

    // Lazy-reload the transcriber if it was cleared (e.g. power_events cleared
    // it on system resume to refresh the Vulkan context). Mirrors the LLM
    // formatter's lazy-load pattern in format_text.
    {
        let mut guard = state.transcriber.lock().unwrap();
        if guard.is_none() {
            let model_path = model_manager::find_whisper_model()
                .ok_or("No whisper model found")?;
            let fc = *state.force_cpu.lock().unwrap();
            eprintln!("Reloading whisper model (lazy) from: {}", model_path.display());
            match transcription::Transcriber::new(&model_path, fc) {
                Ok(t) => {
                    *state.using_gpu.lock().unwrap() = Some(t.is_using_gpu());
                    *guard = Some(t);
                    let _ = app_handle.emit("model-ready", "ok");
                }
                Err(e) => {
                    let msg = format!("Failed to load model: {e}");
                    eprintln!("{msg}");
                    *state.using_gpu.lock().unwrap() = Some(false);
                    return Err(msg);
                }
            }
        }
    }

    let transcriber_guard = state.transcriber.lock().unwrap();
    let transcriber = transcriber_guard.as_ref().ok_or("Model not loaded")?;

    let handle = app_handle.clone();
    let result = transcriber
        .transcribe_with_progress(&samples, move |pct| {
            let _ = handle.emit("transcribe-progress", pct);
        })
        .map_err(|e| format!("Transcription failed: {e}"))?;

    // Gentle handling for no speech
    if result.text.is_empty() {
        return Err("hint:No speech detected".into());
    }

    // Capture the true raw Whisper output BEFORE word-bank substitution.
    // The history panel exposes this so users can spot recurring mishears
    // to add to the word bank ("Whisper keeps hearing 'clod' — let me add
    // that rule"). Once word bank replaces "clod" with "Claude", the raw
    // evidence disappears unless we preserve it here.
    let raw_whisper = result.text.clone();

    // Apply the user's word-replacement bank before any downstream consumer
    // (LLM formatter, direct inject, auto-format). This is the single choke
    // point for Whisper output — doing it here means every flow benefits
    // without needing to remember it at each call site. Loaded fresh each
    // time so edits in Settings take effect immediately with no reload.
    let corrected = word_bank::apply(&result.text, &word_bank::load());
    if corrected != raw_whisper {
        eprintln!("word_bank: rewrote Whisper output");
    }

    Ok(TranscriptionResult {
        text: corrected,
        raw: raw_whisper,
    })
}

/// Return type for `stop_and_transcribe`. `text` is the post-word-bank
/// output the UI uses for display / inject / format; `raw` is the true
/// pre-word-bank Whisper output, exposed to the history panel so users
/// can review what Whisper actually heard.
#[derive(serde::Serialize)]
struct TranscriptionResult {
    text: String,
    raw: String,
}

#[tauri::command]
fn cancel_recording(state: State<AppState>) -> Result<(), String> {
    let session = {
        let mut guard = state.recording.lock().unwrap();
        guard.take().ok_or("Not recording")?
    };
    session.stop_flag.store(true, Ordering::Relaxed);
    drop(session.preview_tx); // kill preview pipeline if any
    if let Some(thread) = session.thread {
        let _ = thread.join();
    }
    Ok(())
}

// ── Live-mode commands (desktop mic) ────────────────────────────────────

/// Start a live-mode session using the desktop's local microphone.
///
/// Spawns a VAD + Whisper pipeline and connects the cpal mic stream to it.
/// A background tokio task drains live events and emits Tauri events
/// (`live-segment`, `live-ended`, etc.) so the frontend can show a running
/// transcript.
///
/// The caller should listen for `live-ended` before calling
/// `start_live_session` again.
#[tauri::command]
async fn start_live_session(
    state: State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    // Guard against double-start.
    {
        let guard = state.live_session.lock().unwrap();
        if guard.is_some() {
            return Err("Live session already running".into());
        }
    }

    // Also reject if a press-to-talk recording is in progress.
    {
        let guard = state.recording.lock().unwrap();
        if guard.is_some() {
            return Err("Cannot start live mode while recording".into());
        }
    }

    // Spawn the VAD + Whisper pipeline (cross-segment context always on).
    let live::LiveSessionHandle { frame_tx, mut event_rx } =
        live::spawn(state.transcriber.clone(), true)
            .map_err(|e| format!("Failed to start live session: {e}"))?;

    // Start mic capture, feeding audio into the pipeline.
    let device = state.selected_device.lock().unwrap().clone();
    let mic = audio_capture::stream_to_live(frame_tx.clone(), device)
        .map_err(|e| format!("Failed to open microphone: {e}"))?;

    // Stash the handles.
    {
        let mut guard = state.live_session.lock().unwrap();
        *guard = Some(DesktopLiveSession { frame_tx, mic });
    }

    // Snapshot format settings for the lifetime of this session.
    let auto_format = *state.auto_format.lock().unwrap();
    let format_type: FormatType = {
        let s = state.auto_format_type.lock().unwrap().clone();
        serde_json::from_value(serde_json::Value::String(s)).unwrap_or(FormatType::None)
    };
    let auto_space = *state.auto_space.lock().unwrap();
    let injection_mode_str = state.injection_mode.lock().unwrap().clone();
    let formatter = state.formatter.clone();
    let force_cpu = *state.force_cpu.lock().unwrap();
    let will_format = auto_format && format_type != FormatType::None;

    // Background task: drain live events → Tauri events + inject.
    tokio::spawn(async move {
        let mut segments: Vec<String> = Vec::new();
        let mut seg_idx: u32 = 0;

        while let Some(event) = event_rx.recv().await {
            match event {
                LiveEvent::SegmentStart => {
                    let _ = app_handle.emit("live-segment-start", ());
                }
                LiveEvent::Segment { text, duration_ms } => {
                    let bank = word_bank::load();
                    let corrected = word_bank::apply(&text, &bank);
                    let trimmed = corrected.trim().to_string();
                    if trimmed.is_empty() {
                        continue;
                    }

                    let idx = seg_idx;
                    seg_idx += 1;
                    segments.push(trimmed.clone());

                    eprintln!("Live segment #{idx}: \"{trimmed}\" ({duration_ms} ms)");

                    let _ = app_handle.emit(
                        "live-segment",
                        serde_json::json!({
                            "index": idx,
                            "text": trimmed,
                            "duration_ms": duration_ms,
                        }),
                    );

                    if !will_format {
                        let mut out = trimmed;
                        if auto_space {
                            out.push(' ');
                        }
                        let mode = match injection_mode_str.as_str() {
                            "type" => InjectionMode::Type,
                            _ => InjectionMode::Paste,
                        };
                        if let Err(e) = injection::inject_text(&out, mode) {
                            eprintln!("Live segment inject failed: {e}");
                        }
                    }
                }
                LiveEvent::SegmentError(msg) => {
                    eprintln!("Live segment error: {msg}");
                    let _ = app_handle.emit("live-error", msg);
                }
                LiveEvent::Ended => {
                    let combined = segments.join(" ").trim().to_string();

                    let final_text = if will_format && !combined.is_empty() {
                        let _ = app_handle.emit("live-formatting", ());
                        let raw = combined.clone();
                        let ft = format_type;
                        let fmt = formatter.clone();
                        let fc = force_cpu;
                        let formatted = tokio::task::spawn_blocking(move || {
                            let mut guard = fmt.lock().unwrap();
                            if guard.is_none() {
                                if let Some(path) = llm::find_llm_model_path() {
                                    match llm::Formatter::new(&path, fc) {
                                        Ok(f) => *guard = Some(f),
                                        Err(e) => return Err(format!("{e}")),
                                    }
                                } else {
                                    return Err("No LLM model found".into());
                                }
                            }
                            guard
                                .as_ref()
                                .unwrap()
                                .format_text(&raw, ft)
                                .map_err(|e| format!("{e}"))
                        })
                        .await;

                        match formatted {
                            Ok(Ok(t)) => t,
                            Ok(Err(e)) => {
                                eprintln!("Live auto-format failed: {e}");
                                combined.clone()
                            }
                            Err(e) => {
                                eprintln!("Live auto-format task panicked: {e}");
                                combined.clone()
                            }
                        }
                    } else {
                        combined
                    };

                    if will_format && !final_text.is_empty() {
                        let mut out = final_text.clone();
                        if auto_space {
                            out.push(' ');
                        }
                        let mode = match injection_mode_str.as_str() {
                            "type" => InjectionMode::Type,
                            _ => InjectionMode::Paste,
                        };
                        if let Err(e) = injection::inject_text(&out, mode) {
                            eprintln!("Live final inject failed: {e}");
                        }
                    }

                    let _ = app_handle.emit(
                        "live-ended",
                        serde_json::json!({
                            "text": final_text,
                            "formatted": will_format,
                        }),
                    );
                    break;
                }
            }
        }
    });

    eprintln!("Live mode: desktop mic session started");
    Ok(())
}

/// Stop the running live-mode session.
///
/// Tears down the cpal mic stream and drops our frame_tx clone. Once
/// the cpal callback's clone is also gone (~100 ms), the live session
/// sees no remaining senders, flushes the VAD, drains any in-flight
/// Whisper jobs, and emits `LiveEvent::Ended`. The background drain
/// task converts that to a `live-ended` Tauri event.
#[tauri::command]
fn stop_live_session(state: State<AppState>) -> Result<(), String> {
    let session = {
        let mut guard = state.live_session.lock().unwrap();
        guard.take().ok_or("No live session running")?
    };

    // Stop mic capture first — the cpal stream is torn down within ~100 ms.
    session.mic.stop();
    // Drop our frame_tx so the session's frame_rx closes once the
    // callback's clone is also dropped.
    drop(session.frame_tx);

    eprintln!("Live mode: stop requested, waiting for drain task to emit live-ended");
    Ok(())
}

/// Returns the peak audio level since the last read and resets to zero.
/// The frontend polls this during recording to detect silence.
#[tauri::command]
fn get_recording_level(state: State<AppState>) -> f32 {
    let guard = state.recording.lock().unwrap();
    if let Some(session) = guard.as_ref() {
        f32::from_bits(session.peak_level.swap(0, Ordering::Relaxed))
    } else {
        0.0
    }
}

/// Dynamically register Escape as a global shortcut (while recording/TTS).
#[tauri::command]
fn register_escape_shortcut(app_handle: tauri::AppHandle) -> Result<(), String> {
    use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Shortcut};
    let sc = Shortcut::new(None, Code::Escape);
    if app_handle.global_shortcut().is_registered(sc) {
        return Ok(());
    }
    app_handle
        .global_shortcut()
        .register(sc)
        .map_err(|e| format!("{e}"))
}

/// Unregister the Escape global shortcut (when nothing to cancel).
#[tauri::command]
fn unregister_escape_shortcut(app_handle: tauri::AppHandle) -> Result<(), String> {
    use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Shortcut};
    let sc = Shortcut::new(None, Code::Escape);
    if !app_handle.global_shortcut().is_registered(sc) {
        return Ok(());
    }
    app_handle
        .global_shortcut()
        .unregister(sc)
        .map_err(|e| format!("{e}"))
}

#[tauri::command]
fn inject_text(state: State<AppState>, text: String) -> Result<(), String> {
    let mode_str = state.injection_mode.lock().unwrap().clone();
    let mode = match mode_str.as_str() {
        "type" => InjectionMode::Type,
        _ => InjectionMode::Paste,
    };

    // Auto-append space if enabled
    let final_text = if *state.auto_space.lock().unwrap() && !text.is_empty() {
        format!("{text} ")
    } else {
        text
    };

    std::thread::sleep(std::time::Duration::from_millis(200));

    injection::inject_text(&final_text, mode).map_err(|e| format!("Injection failed: {e}"))
}

#[tauri::command]
fn set_injection_mode(state: State<AppState>, mode: String) {
    *state.injection_mode.lock().unwrap() = mode;
}

#[tauri::command]
fn list_audio_devices() -> Vec<String> {
    audio_capture::list_input_devices()
}

#[tauri::command]
fn get_selected_device(state: State<AppState>) -> Option<String> {
    state.selected_device.lock().unwrap().clone()
}

#[tauri::command]
fn set_selected_device(state: State<AppState>, device: Option<String>) {
    *state.selected_device.lock().unwrap() = device;
}

#[tauri::command]
fn get_autostart() -> serde_json::Value {
    use std::os::windows::process::CommandExt;
    let output = std::process::Command::new("reg")
        .args([
            "query",
            r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run",
            "/v",
            "Dictator",
        ])
        .creation_flags(0x08000000) // CREATE_NO_WINDOW
        .output();

    let ok_output = match output {
        Ok(o) if o.status.success() => o,
        _ => return serde_json::json!({"enabled": false, "stale": false}),
    };

    // Parse the stored value from reg output ("    Dictator    REG_SZ    \"C:\path\to\exe\" --minimized")
    let stdout = String::from_utf8_lossy(&ok_output.stdout);
    let stored_raw = stdout
        .lines()
        .find(|l| l.contains("Dictator"))
        .and_then(|l| l.split("REG_SZ").nth(1))
        .map(|p| p.trim().to_string());

    // Extract just the exe path: strip quotes and any trailing flags like --minimized
    let stored_path = stored_raw.as_deref().map(|s| {
        let s = s.trim().trim_start_matches('"');
        // If quoted, take up to closing quote; otherwise take up to first space
        if let Some(end) = s.find('"') {
            s[..end].to_string()
        } else {
            s.split_whitespace().next().unwrap_or(s).to_string()
        }
    });

    let current_exe = std::env::current_exe()
        .ok()
        .and_then(|p| p.to_str().map(|s| s.to_string()));

    let stale = match (&stored_path, &current_exe) {
        (Some(stored), Some(current)) => !stored.eq_ignore_ascii_case(current),
        _ => false,
    };

    serde_json::json!({"enabled": true, "stale": stale})
}

#[tauri::command]
fn get_hotkey(state: State<AppState>) -> String {
    state.hotkey.lock().unwrap().clone()
}

#[tauri::command]
fn get_inject_hotkey(state: State<AppState>) -> String {
    state.inject_hotkey.lock().unwrap().clone()
}

/// Re-register all global shortcuts from current state.
fn register_all_hotkeys(
    app_handle: &tauri::AppHandle,
    rec_preset: &str,
    inject_preset: &str,
    tts_preset: &str,
    reformat_preset: &str,
    copy_preset: &str,
) -> Result<(), String> {
    use tauri_plugin_global_shortcut::GlobalShortcutExt;

    let _ = app_handle.global_shortcut().unregister_all();

    if let Some(sc) = parse_hotkey_preset(rec_preset) {
        app_handle
            .global_shortcut()
            .register(sc)
            .map_err(|e| format!("Failed to register recording hotkey: {e}"))?;
    }
    if let Some(sc) = parse_hotkey_preset(inject_preset) {
        app_handle
            .global_shortcut()
            .register(sc)
            .map_err(|e| format!("Failed to register inject hotkey: {e}"))?;
    }
    if let Some(sc) = parse_hotkey_preset(tts_preset) {
        app_handle
            .global_shortcut()
            .register(sc)
            .map_err(|e| format!("Failed to register read-aloud hotkey: {e}"))?;
    }
    if let Some(sc) = parse_hotkey_preset(reformat_preset) {
        app_handle
            .global_shortcut()
            .register(sc)
            .map_err(|e| format!("Failed to register reformat hotkey: {e}"))?;
    }
    if let Some(sc) = parse_hotkey_preset(copy_preset) {
        app_handle
            .global_shortcut()
            .register(sc)
            .map_err(|e| format!("Failed to register copy-to-clipboard hotkey: {e}"))?;
    }
    Ok(())
}

fn get_all_hotkey_presets(state: &AppState) -> (String, String, String, String, String) {
    let rec = state.hotkey.lock().unwrap().clone();
    let inject = state.inject_hotkey.lock().unwrap().clone();
    let tts = state.tts_hotkey.lock().unwrap().clone();
    let reformat = state.reformat_hotkey.lock().unwrap().clone();
    let copy = state.copy_hotkey.lock().unwrap().clone();
    (rec, inject, tts, reformat, copy)
}

#[tauri::command]
fn set_hotkey(
    app_handle: tauri::AppHandle,
    state: State<AppState>,
    preset: String,
) -> Result<(), String> {
    let (_, inject, tts, reformat, copy) = get_all_hotkey_presets(&state);
    register_all_hotkeys(&app_handle, &preset, &inject, &tts, &reformat, &copy)?;
    *state.hotkey.lock().unwrap() = preset;
    Ok(())
}

#[tauri::command]
fn set_inject_hotkey(
    app_handle: tauri::AppHandle,
    state: State<AppState>,
    preset: String,
) -> Result<(), String> {
    let (rec, _, tts, reformat, copy) = get_all_hotkey_presets(&state);
    register_all_hotkeys(&app_handle, &rec, &preset, &tts, &reformat, &copy)?;
    *state.inject_hotkey.lock().unwrap() = preset;
    Ok(())
}

#[tauri::command]
fn get_tts_hotkey(state: State<AppState>) -> String {
    state.tts_hotkey.lock().unwrap().clone()
}

#[tauri::command]
fn set_tts_hotkey(
    app_handle: tauri::AppHandle,
    state: State<AppState>,
    preset: String,
) -> Result<(), String> {
    let (rec, inject, _, reformat, copy) = get_all_hotkey_presets(&state);
    register_all_hotkeys(&app_handle, &rec, &inject, &preset, &reformat, &copy)?;
    *state.tts_hotkey.lock().unwrap() = preset;
    Ok(())
}

#[tauri::command]
fn get_reformat_hotkey(state: State<AppState>) -> String {
    state.reformat_hotkey.lock().unwrap().clone()
}

#[tauri::command]
fn set_reformat_hotkey(
    app_handle: tauri::AppHandle,
    state: State<AppState>,
    preset: String,
) -> Result<(), String> {
    let (rec, inject, tts, _, copy) = get_all_hotkey_presets(&state);
    register_all_hotkeys(&app_handle, &rec, &inject, &tts, &preset, &copy)?;
    *state.reformat_hotkey.lock().unwrap() = preset;
    Ok(())
}

#[tauri::command]
fn get_copy_hotkey(state: State<AppState>) -> String {
    state.copy_hotkey.lock().unwrap().clone()
}

#[tauri::command]
fn set_copy_hotkey(
    app_handle: tauri::AppHandle,
    state: State<AppState>,
    preset: String,
) -> Result<(), String> {
    let (rec, inject, tts, reformat, _) = get_all_hotkey_presets(&state);
    register_all_hotkeys(&app_handle, &rec, &inject, &tts, &reformat, &preset)?;
    *state.copy_hotkey.lock().unwrap() = preset;
    Ok(())
}

/// Put arbitrary text onto the system clipboard. Used by the "copy last
/// transcription" hotkey for apps like RustDesk that block SendInput
/// injection but accept manual Ctrl+V paste.
#[tauri::command]
fn copy_text_to_clipboard(text: String) -> Result<(), String> {
    use arboard::Clipboard;
    let mut cb = Clipboard::new().map_err(|e| format!("Clipboard error: {e}"))?;
    cb.set_text(text)
        .map_err(|e| format!("Clipboard error: {e}"))?;
    Ok(())
}

fn parse_hotkey_preset(
    preset: &str,
) -> Option<tauri_plugin_global_shortcut::Shortcut> {
    use tauri_plugin_global_shortcut::{Code, Modifiers, Shortcut};

    if preset == "none" || preset.is_empty() {
        return None;
    }

    let parts: Vec<&str> = preset.split('+').collect();
    let mut modifiers = Modifiers::empty();
    let mut key_part: Option<&str> = None;

    for part in &parts {
        match part.to_lowercase().as_str() {
            "ctrl" | "control" => modifiers |= Modifiers::CONTROL,
            "shift" => modifiers |= Modifiers::SHIFT,
            "alt" => modifiers |= Modifiers::ALT,
            "super" | "meta" | "win" => modifiers |= Modifiers::SUPER,
            _ => key_part = Some(part),
        }
    }

    let key_str = key_part?;
    let code = match key_str.to_lowercase().as_str() {
        "space" => Code::Space,
        "enter" | "return" => Code::Enter,
        "tab" => Code::Tab,
        "backspace" => Code::Backspace,
        "delete" => Code::Delete,
        "insert" => Code::Insert,
        "home" => Code::Home,
        "end" => Code::End,
        "pageup" => Code::PageUp,
        "pagedown" => Code::PageDown,
        "up" | "arrowup" => Code::ArrowUp,
        "down" | "arrowdown" => Code::ArrowDown,
        "left" | "arrowleft" => Code::ArrowLeft,
        "right" | "arrowright" => Code::ArrowRight,
        "f1" => Code::F1, "f2" => Code::F2, "f3" => Code::F3, "f4" => Code::F4,
        "f5" => Code::F5, "f6" => Code::F6, "f7" => Code::F7, "f8" => Code::F8,
        "f9" => Code::F9, "f10" => Code::F10, "f11" => Code::F11, "f12" => Code::F12,
        "a" => Code::KeyA, "b" => Code::KeyB, "c" => Code::KeyC, "d" => Code::KeyD,
        "e" => Code::KeyE, "f" => Code::KeyF, "g" => Code::KeyG, "h" => Code::KeyH,
        "i" => Code::KeyI, "j" => Code::KeyJ, "k" => Code::KeyK, "l" => Code::KeyL,
        "m" => Code::KeyM, "n" => Code::KeyN, "o" => Code::KeyO, "p" => Code::KeyP,
        "q" => Code::KeyQ, "r" => Code::KeyR, "s" => Code::KeyS, "t" => Code::KeyT,
        "u" => Code::KeyU, "v" => Code::KeyV, "w" => Code::KeyW, "x" => Code::KeyX,
        "y" => Code::KeyY, "z" => Code::KeyZ,
        "0" => Code::Digit0, "1" => Code::Digit1, "2" => Code::Digit2, "3" => Code::Digit3,
        "4" => Code::Digit4, "5" => Code::Digit5, "6" => Code::Digit6, "7" => Code::Digit7,
        "8" => Code::Digit8, "9" => Code::Digit9,
        "`" | "backquote" => Code::Backquote,
        "-" | "minus" => Code::Minus,
        "=" | "equal" => Code::Equal,
        "[" | "bracketleft" => Code::BracketLeft,
        "]" | "bracketright" => Code::BracketRight,
        "\\" | "backslash" => Code::Backslash,
        ";" | "semicolon" => Code::Semicolon,
        "'" | "quote" => Code::Quote,
        "," | "comma" => Code::Comma,
        "." | "period" => Code::Period,
        "/" | "slash" => Code::Slash,
        _ => return None,
    };

    let mods = if modifiers.is_empty() { None } else { Some(modifiers) };
    Some(Shortcut::new(mods, code))
}

#[tauri::command]
fn set_autostart(enabled: bool) -> Result<(), String> {
    use std::os::windows::process::CommandExt;
    if enabled {
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
        let exe_str = exe.to_str().ok_or("Invalid exe path")?;
        // Quote the path and append --minimized so the app starts hidden to tray
        let reg_value = format!("\"{}\" --minimized", exe_str);
        let output = std::process::Command::new("reg")
            .args([
                "add",
                r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run",
                "/v",
                "Dictator",
                "/t",
                "REG_SZ",
                "/d",
                &reg_value,
                "/f",
            ])
            .creation_flags(0x08000000)
            .output()
            .map_err(|e| e.to_string())?;
        if !output.status.success() {
            return Err("Failed to enable autostart".into());
        }
    } else {
        let output = std::process::Command::new("reg")
            .args([
                "delete",
                r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run",
                "/v",
                "Dictator",
                "/f",
            ])
            .creation_flags(0x08000000)
            .output()
            .map_err(|e| e.to_string())?;
        if !output.status.success() {
            return Err("Failed to disable autostart".into());
        }
    }
    Ok(())
}

/// Re-register autostart with the current exe path (repairs stale entries).
#[tauri::command]
fn fix_autostart() -> Result<(), String> {
    set_autostart(true)
}

// ── Uninstall / Cleanup ────────────────────────────────────────────────

/// Remove all app data so the user can cleanly delete the executable.
///
/// Cleans up:
///   1. models/ directory (Whisper, LLM, Piper exe, Piper voices)
///   2. %TEMP%/dictator-certs/ (Tailscale certificates)
///   3. Autostart registry entry (if present)
///
/// Returns a JSON summary of what was removed and any errors encountered.
#[tauri::command]
fn uninstall_cleanup() -> serde_json::Value {
    use std::os::windows::process::CommandExt;
    let mut removed = Vec::<String>::new();
    let mut errors = Vec::<String>::new();

    // 1. Delete models directory
    let models = model_manager::models_dir();
    if models.exists() {
        match std::fs::remove_dir_all(&models) {
            Ok(_) => removed.push(format!("Models directory: {}", models.display())),
            Err(e) => errors.push(format!("Failed to delete models dir: {e}")),
        }
    }

    // 2. Delete Tailscale cert directory
    let cert_dir = std::env::temp_dir().join("dictator-certs");
    if cert_dir.exists() {
        match std::fs::remove_dir_all(&cert_dir) {
            Ok(_) => removed.push(format!("Certificates: {}", cert_dir.display())),
            Err(e) => errors.push(format!("Failed to delete cert dir: {e}")),
        }
    }

    // 3. Remove autostart registry entry (ignore if not present)
    let reg_output = std::process::Command::new("reg")
        .args([
            "delete",
            r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run",
            "/v",
            "Dictator",
            "/f",
        ])
        .creation_flags(0x08000000) // CREATE_NO_WINDOW
        .output();
    match reg_output {
        Ok(o) if o.status.success() => {
            removed.push("Autostart registry entry".into());
        }
        _ => {
            // Not an error — entry may simply not exist
        }
    }

    serde_json::json!({
        "removed": removed,
        "errors": errors,
    })
}

// ── Version Check ───────────────────────────────────────────────────────

const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

#[tauri::command]
fn get_app_version() -> String {
    APP_VERSION.to_string()
}

/// Version check endpoint. Currently uses a public Gist (works while repo is private).
/// When the repo goes public, switch to GitHub Releases API:
///   https://api.github.com/repos/yuanbaca/Dictator/releases/latest
const VERSION_CHECK_URL: &str =
    "https://gist.githubusercontent.com/yuanbaca/0a44d0642f6e774ab3776bda7fd870a1/raw/version.json";

/// Compare two semver strings (e.g. "0.3.1" vs "0.3.0").
/// Returns true if `remote` is strictly newer than `local`.
fn is_newer_version(remote: &str, local: &str) -> bool {
    let parse = |s: &str| -> (u32, u32, u32) {
        let mut parts = s.split('.').filter_map(|p| p.parse::<u32>().ok());
        (
            parts.next().unwrap_or(0),
            parts.next().unwrap_or(0),
            parts.next().unwrap_or(0),
        )
    };
    parse(remote) > parse(local)
}

#[tauri::command]
async fn check_for_updates() -> serde_json::Value {
    let current = APP_VERSION;

    let result = reqwest::Client::new()
        .get(VERSION_CHECK_URL)
        .header("Cache-Control", "no-cache")
        .send()
        .await;

    match result {
        Ok(resp) if resp.status().is_success() => {
            if let Ok(json) = resp.json::<serde_json::Value>().await {
                let latest = json["version"].as_str().unwrap_or("");
                let url = json["url"].as_str().unwrap_or("");
                return serde_json::json!({
                    "current": current,
                    "latest": latest,
                    "update_available": !latest.is_empty() && is_newer_version(latest, current),
                    "url": url,
                });
            }
        }
        _ => {}
    }

    serde_json::json!({
        "current": current,
        "latest": null,
        "update_available": false,
        "url": null,
    })
}

// ── Model Management commands ───────────────────────────────────────────

#[tauri::command]
fn has_llm_model() -> bool {
    model_manager::has_llm_model()
}

#[tauri::command]
fn list_llm_models() -> Vec<serde_json::Value> {
    model_manager::installed_llm_models()
        .into_iter()
        .map(|(filename, name)| serde_json::json!({"filename": filename, "name": name}))
        .collect()
}

#[tauri::command]
fn list_whisper_models() -> Vec<serde_json::Value> {
    model_manager::WHISPER_MODELS
        .iter()
        .map(|m| {
            serde_json::json!({
                "id": m.id,
                "name": m.name,
                "filename": m.filename,
                "size_mb": m.size_bytes / 1_000_000,
                "installed": model_manager::models_dir().join(m.filename).exists(),
            })
        })
        .collect()
}

#[tauri::command]
async fn download_whisper_model(
    app_handle: tauri::AppHandle,
    model_id: String,
) -> Result<String, String> {
    let handle = app_handle.clone();
    model_manager::download_whisper_model(&model_id, move |downloaded, total| {
        let pct = if total > 0 {
            (downloaded as f64 / total as f64 * 100.0) as u32
        } else {
            0
        };
        let _ = handle.emit("whisper-download-progress", serde_json::json!({
            "downloaded": downloaded,
            "total": total,
            "percent": pct,
        }));
    })
    .await
    .map(|p| p.to_string_lossy().to_string())
}

#[tauri::command]
fn load_whisper_model(state: State<AppState>, app_handle: tauri::AppHandle) -> Result<(), String> {
    let model_path = model_manager::find_whisper_model()
        .ok_or("No whisper model found")?;

    *state.no_model.lock().unwrap() = false;
    let transcriber_handle = state.transcriber.clone();
    let gpu_handle = state.using_gpu.clone();
    let force_cpu = *state.force_cpu.lock().unwrap();

    std::thread::spawn(move || {
        eprintln!("Loading whisper model from: {}", model_path.display());
        match transcription::Transcriber::new(&model_path, force_cpu) {
            Ok(t) => {
                let gpu = t.is_using_gpu();
                *gpu_handle.lock().unwrap() = Some(gpu);
                *transcriber_handle.lock().unwrap() = Some(t);
                eprintln!("Model loaded successfully ({})", if gpu { "GPU" } else { "CPU" });
                let _ = app_handle.emit("model-ready", "ok");
            }
            Err(e) => {
                let msg = format!("Failed to load model: {e}");
                eprintln!("{msg}");
                let _ = app_handle.emit("model-ready", &msg);
            }
        }
    });

    Ok(())
}

#[tauri::command]
fn list_available_models() -> Vec<serde_json::Value> {
    let dir = model_manager::models_dir();
    let known_filenames: std::collections::HashSet<&str> = model_manager::AVAILABLE_MODELS
        .iter()
        .map(|m| m.filename)
        .collect();

    // Start with known models (downloadable)
    let mut results: Vec<serde_json::Value> = model_manager::AVAILABLE_MODELS
        .iter()
        .map(|m| {
            serde_json::json!({
                "id": m.id,
                "name": m.name,
                "filename": m.filename,
                "size_mb": m.size_bytes / 1_000_000,
                "installed": dir.join(m.filename).exists(),
                "custom": false,
            })
        })
        .collect();

    // Discover custom GGUF files not in the known list
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "gguf") {
                let filename = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                if !known_filenames.contains(filename.as_str()) {
                    let size_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                    let display_name = path.file_stem()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();
                    results.push(serde_json::json!({
                        "id": format!("custom:{}", filename),
                        "name": display_name,
                        "filename": filename,
                        "size_mb": size_bytes / 1_000_000,
                        "installed": true,
                        "custom": true,
                    }));
                }
            }
        }
    }

    results
}

#[tauri::command]
async fn download_llm_model(
    app_handle: tauri::AppHandle,
    model_id: String,
) -> Result<String, String> {
    let handle = app_handle.clone();
    model_manager::download_model(&model_id, move |downloaded, total| {
        let pct = if total > 0 {
            (downloaded as f64 / total as f64 * 100.0) as u32
        } else {
            0
        };
        let _ = handle.emit("llm-download-progress", serde_json::json!({
            "downloaded": downloaded,
            "total": total,
            "percent": pct,
        }));
    })
    .await
    .map(|p| p.to_string_lossy().to_string())
}

#[tauri::command]
fn delete_llm_model(state: State<AppState>, filename: String) -> Result<(), String> {
    // Unload the formatter first — on Windows the loaded model locks the file
    *state.formatter.lock().unwrap() = None;
    *state.llm_status.lock().unwrap() = None;

    // Retry a few times — Windows may take a moment to release memory-mapped handles
    let mut last_err = String::new();
    for attempt in 0..5 {
        if attempt > 0 {
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
        match model_manager::delete_model(&filename) {
            Ok(()) => return Ok(()),
            Err(e) => last_err = e,
        }
    }
    Err(last_err)
}

#[tauri::command]
fn get_models_dir() -> String {
    let path = model_manager::models_dir()
        .canonicalize()
        .unwrap_or_else(|_| model_manager::models_dir())
        .to_string_lossy()
        .to_string();
    // Strip Windows UNC prefix (\\?\) that canonicalize adds — explorer doesn't like it
    path.strip_prefix(r"\\?\").unwrap_or(&path).to_string()
}

/// Validate a GGUF model by checking magic bytes — does NOT load the model,
/// so the file stays unlocked and deletable on Windows.
#[tauri::command]
fn validate_llm_model(filename: String) -> serde_json::Value {
    let model_path = model_manager::models_dir().join(&filename);
    if !model_path.exists() {
        return serde_json::json!({
            "valid": false,
            "error": "File not found",
            "chat_format": null,
        });
    }

    // Detect the chat format from filename (with user override check)
    let mut detected = ChatFormat::detect(&model_path.to_string_lossy());
    if let Some(override_str) = model_manager::get_chat_format_override(&filename) {
        if let Ok(fmt) = serde_json::from_value::<ChatFormat>(
            serde_json::Value::String(override_str),
        ) {
            detected = fmt;
        }
    }
    let fmt_key = detected.key();

    // Quick validation: check GGUF magic bytes (0x47 0x47 0x55 0x46 = "GGUF")
    // This avoids loading the full model which memory-maps the file and blocks deletion on Windows.
    let valid = match std::fs::File::open(&model_path).and_then(|mut f| {
        use std::io::Read;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        Ok(magic)
    }) {
        Ok(magic) => &magic == b"GGUF",
        Err(_) => false,
    };

    let error = if valid { None } else { Some("Not a valid GGUF file".to_string()) };

    serde_json::json!({
        "valid": valid,
        "error": error,
        "chat_format": fmt_key,
    })
}

/// Set the chat format override for a model file, then reset the loaded formatter
/// so the next format_text call picks up the change.
#[tauri::command]
fn set_model_chat_format(state: State<AppState>, filename: String, format: String) -> Result<(), String> {
    // Validate the format string parses
    let _: ChatFormat = serde_json::from_value(
        serde_json::Value::String(format.clone()),
    )
    .map_err(|_| format!("Unknown chat format: {format}"))?;

    model_manager::set_chat_format_override(&filename, Some(&format));

    // Clear the loaded formatter so it reloads with the new format on next use
    *state.formatter.lock().unwrap() = None;
    *state.llm_status.lock().unwrap() = None;
    *state.llm_error.lock().unwrap() = None;

    Ok(())
}

// ── TTS commands ────────────────────────────────────────────────────────

#[tauri::command]
fn speak_text(state: State<AppState>, text: String) -> Result<(), String> {
    let engine = state.tts_engine.lock().unwrap().clone();
    if engine == "piper" {
        state.piper.speak(&text)
    } else if engine == "minimax" {
        let api_key = state.minimax_api_key.lock().unwrap().clone();
        let voice_id = state.minimax_voice.lock().unwrap().clone();
        let playing = state.minimax_playing.clone();
        let sink = state.minimax_sink.clone();
        // Run on a dedicated thread to avoid tokio runtime conflicts
        let text = text.clone();
        std::thread::spawn(move || {
            let _ = tts::minimax_speak(&api_key, &voice_id, &text, &playing, &sink, || {});
        });
        Ok(())
    } else {
        let mut guard = state.speaker.lock().unwrap();
        if guard.is_none() {
            *guard = Some(tts::SapiSpeaker::new()?);
        }
        guard.as_ref().unwrap().speak(&text)
    }
}

#[tauri::command]
fn stop_speaking(state: State<AppState>) -> Result<(), String> {
    let engine = state.tts_engine.lock().unwrap().clone();
    if engine == "piper" {
        state.piper.stop();
        Ok(())
    } else if engine == "minimax" {
        let sink = state.minimax_sink.lock().unwrap();
        if let Some(s) = sink.as_ref() {
            s.stop();
        }
        state.minimax_playing.store(false, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    } else {
        let guard = state.speaker.lock().unwrap();
        if let Some(speaker) = guard.as_ref() {
            speaker.stop()?;
        }
        Ok(())
    }
}

#[tauri::command]
fn read_selected_text(state: State<AppState>, app_handle: tauri::AppHandle) -> Result<(), String> {
    let engine = state.tts_engine.lock().unwrap().clone();

    // Cycle: speaking → pause, paused → resume, idle → capture & speak
    if engine == "piper" {
        if state.piper.is_speaking() {
            if state.piper.is_paused() {
                state.piper.resume();
                let _ = app_handle.emit("tts-state", "speaking");
            } else {
                state.piper.pause();
                let _ = app_handle.emit("tts-state", "paused");
            }
            return Ok(());
        }
    } else if engine == "minimax" {
        if state.minimax_playing.load(std::sync::atomic::Ordering::Relaxed) {
            let sink = state.minimax_sink.lock().unwrap();
            if let Some(s) = sink.as_ref() { s.stop(); }
            state.minimax_playing.store(false, std::sync::atomic::Ordering::Relaxed);
            let _ = app_handle.emit("tts-state", "idle");
            return Ok(());
        }
    } else {
        let guard = state.speaker.lock().unwrap();
        if let Some(speaker) = guard.as_ref() {
            if speaker.is_speaking() {
                let _ = speaker.stop();
                let _ = app_handle.emit("tts-state", "idle");
                return Ok(());
            }
        }
    }

    let text = tts::capture_selected_text()?;

    let _ = app_handle.emit("tts-text", text.as_str());
    // Minimax has a loading phase (API call) before audio plays
    let initial_state = if engine == "minimax" { "loading" } else { "speaking" };
    let _ = app_handle.emit("tts-state", initial_state);

    if engine == "piper" {
        let piper = state.piper.clone();
        let handle = app_handle.clone();
        std::thread::spawn(move || {
            let result = piper.speak(&text);
            let _ = handle.emit("tts-state", "idle");
            if let Err(e) = result {
                eprintln!("TTS error: {e}");
            }
        });
        Ok(())
    } else if engine == "minimax" {
        let api_key = state.minimax_api_key.lock().unwrap().clone();
        let voice_id = state.minimax_voice.lock().unwrap().clone();
        let playing = state.minimax_playing.clone();
        let sink = state.minimax_sink.clone();
        let handle = app_handle.clone();
        std::thread::spawn(move || {
            let result = tts::minimax_speak(&api_key, &voice_id, &text, &playing, &sink, {
                let h = handle.clone();
                move || { let _ = h.emit("tts-state", "speaking"); }
            });
            let _ = handle.emit("tts-state", "idle");
            if let Err(e) = result {
                eprintln!("Minimax TTS error: {e}");
            }
        });
        Ok(())
    } else {
        let mut guard = state.speaker.lock().unwrap();
        if guard.is_none() {
            *guard = Some(tts::SapiSpeaker::new()?);
        }
        let result = guard.as_ref().unwrap().speak(&text);
        let _ = app_handle.emit("tts-state", "idle");
        result
    }
}

#[tauri::command]
fn stop_tts(state: State<AppState>, app_handle: tauri::AppHandle) -> Result<(), String> {
    let engine = state.tts_engine.lock().unwrap().clone();
    if engine == "piper" {
        state.piper.stop();
    } else if engine == "minimax" {
        let sink = state.minimax_sink.lock().unwrap();
        if let Some(s) = sink.as_ref() { s.stop(); }
        state.minimax_playing.store(false, std::sync::atomic::Ordering::Relaxed);
    } else {
        let guard = state.speaker.lock().unwrap();
        if let Some(speaker) = guard.as_ref() {
            speaker.stop()?;
        }
    }
    let _ = app_handle.emit("tts-state", "idle");
    Ok(())
}

#[tauri::command]
fn list_tts_voices(state: State<AppState>) -> Result<Vec<String>, String> {
    let mut speaker_guard = state.speaker.lock().unwrap();
    if speaker_guard.is_none() {
        *speaker_guard = Some(tts::SapiSpeaker::new()?);
    }
    Ok(speaker_guard.as_ref().unwrap().list_voices())
}

#[tauri::command]
fn set_tts_voice(state: State<AppState>, name: String) -> Result<(), String> {
    let mut speaker_guard = state.speaker.lock().unwrap();
    if speaker_guard.is_none() {
        *speaker_guard = Some(tts::SapiSpeaker::new()?);
    }
    speaker_guard.as_ref().unwrap().set_voice(&name)
}

#[tauri::command]
fn set_tts_rate(state: State<AppState>, rate: f32) -> Result<(), String> {
    let mut speaker_guard = state.speaker.lock().unwrap();
    if speaker_guard.is_none() {
        *speaker_guard = Some(tts::SapiSpeaker::new()?);
    }
    speaker_guard.as_ref().unwrap().set_rate(rate)
}

// ── Piper TTS commands ──────────────────────────────────────────────────

#[tauri::command]
fn get_tts_engine(state: State<AppState>) -> String {
    state.tts_engine.lock().unwrap().clone()
}

#[tauri::command]
fn set_tts_engine(state: State<AppState>, engine: String) -> Result<(), String> {
    if engine != "sapi" && engine != "piper" && engine != "minimax" {
        return Err(format!("Unknown engine: {engine}"));
    }
    *state.tts_engine.lock().unwrap() = engine;
    Ok(())
}

#[tauri::command]
fn get_minimax_api_key(state: State<AppState>) -> String {
    state.minimax_api_key.lock().unwrap().clone()
}

#[tauri::command]
fn set_minimax_api_key(state: State<AppState>, key: String) {
    *state.minimax_api_key.lock().unwrap() = key;
}

#[tauri::command]
fn get_minimax_voice(state: State<AppState>) -> String {
    state.minimax_voice.lock().unwrap().clone()
}

#[tauri::command]
fn set_minimax_voice(state: State<AppState>, voice_id: String) {
    *state.minimax_voice.lock().unwrap() = voice_id;
}

#[tauri::command]
fn list_minimax_voices() -> Vec<serde_json::Value> {
    tts::MINIMAX_VOICES.iter().map(|(id, name)| {
        serde_json::json!({ "id": id, "name": name })
    }).collect()
}

/// Test the Minimax API key by synthesizing a short phrase.
/// Runs on a spawned thread to avoid tokio runtime conflicts.
#[tauri::command]
fn test_minimax_api(state: State<AppState>) -> Result<String, String> {
    let api_key = state.minimax_api_key.lock().unwrap().clone();
    let voice_id = state.minimax_voice.lock().unwrap().clone();
    if api_key.is_empty() {
        return Err("No API key set".to_string());
    }
    let playing = state.minimax_playing.clone();
    let sink = state.minimax_sink.clone();

    // Spawn a thread for the blocking HTTP call, then join to get the result
    let handle = std::thread::spawn(move || {
        tts::minimax_speak(&api_key, &voice_id, "Hello, this is a test of the Minimax voice.", &playing, &sink, || {})
    });
    match handle.join() {
        Ok(Ok(())) => Ok("success".to_string()),
        Ok(Err(e)) => Err(e),
        Err(_) => Err("Thread panicked".to_string()),
    }
}

#[tauri::command]
fn has_piper(state: State<AppState>) -> bool {
    state.piper.has_piper_exe()
}

#[tauri::command]
fn list_piper_voices(state: State<AppState>) -> Vec<serde_json::Value> {
    let mut voices: Vec<serde_json::Value> = tts::PIPER_VOICES
        .iter()
        .map(|v| {
            serde_json::json!({
                "id": v.id,
                "name": v.name,
                "size_mb": v.size_bytes / 1_000_000,
                "installed": state.piper.has_voice(v.id),
                "custom": false,
            })
        })
        .collect();

    // Append custom voices discovered on disk
    for cv in state.piper.discover_custom_voices() {
        voices.push(serde_json::json!({
            "id": cv.id,
            "name": cv.display_name,
            "size_mb": cv.size_bytes / 1_000_000,
            "installed": true,
            "custom": true,
        }));
    }

    voices
}

#[tauri::command]
fn get_piper_voice_dir(state: State<AppState>) -> String {
    state.piper.voice_dir().to_string_lossy().to_string()
}

#[tauri::command]
fn set_piper_voice(state: State<AppState>, voice_id: String) {
    state.piper.set_voice(&voice_id);
}

#[tauri::command]
async fn download_piper(
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    let state: State<AppState> = app_handle.state();
    let piper_dir = state.piper.piper_dir().to_path_buf();
    let handle = app_handle.clone();

    tts::download_piper_exe(&piper_dir, |downloaded, total| {
        let percent = if total > 0 {
            (downloaded * 100 / total) as u32
        } else {
            0
        };
        let _ = handle.emit("piper-download-progress", serde_json::json!({
            "downloaded": downloaded,
            "total": total,
            "percent": percent,
        }));
    })
    .await
}

#[tauri::command]
async fn download_piper_voice(
    app_handle: tauri::AppHandle,
    voice_id: String,
) -> Result<(), String> {
    let state: State<AppState> = app_handle.state();
    let voice_dir = state.piper.voice_dir().to_path_buf();
    let handle = app_handle.clone();

    tts::download_piper_voice(&voice_dir, &voice_id, |downloaded, total| {
        let percent = if total > 0 {
            (downloaded * 100 / total) as u32
        } else {
            0
        };
        let _ = handle.emit("piper-voice-download-progress", serde_json::json!({
            "downloaded": downloaded,
            "total": total,
            "percent": percent,
        }));
    })
    .await
}

#[tauri::command]
fn delete_piper_voice(state: State<AppState>, voice_id: String) -> Result<(), String> {
    state.piper.delete_voice(&voice_id)
}

// ── Utility commands ────────────────────────────────────────────────────

#[tauri::command]
fn open_path(path: String) -> Result<(), String> {
    std::process::Command::new("explorer")
        .arg(&path)
        .spawn()
        .map_err(|e| format!("Failed to open path: {e}"))?;
    Ok(())
}

#[tauri::command]
fn show_notification(title: String, body: String) -> Result<(), String> {
    use std::os::windows::process::CommandExt;
    let safe_title = title.replace('"', "&quot;").replace('<', "&lt;");
    let safe_body = body.replace('"', "&quot;").replace('<', "&lt;");

    let script = [
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null",
        "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom, ContentType = WindowsRuntime] | Out-Null",
        &format!(r#"$xml = '<toast duration="short"><visual><binding template="ToastGeneric"><text>{safe_title}</text><text>{safe_body}</text></binding></visual><audio src="ms-winsoundevent:Notification.Default"/></toast>'"#),
        "$doc = New-Object Windows.Data.Xml.Dom.XmlDocument",
        "$doc.LoadXml($xml)",
        "$toast = New-Object Windows.UI.Notifications.ToastNotification $doc",
        r#"[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe").Show($toast)"#,
    ].join("\n");

    let script_path = std::env::temp_dir().join("dictator_toast.ps1");
    std::fs::write(&script_path, &script)
        .map_err(|e| format!("Failed to write notification script: {e}"))?;

    std::thread::spawn(move || {
        let _ = std::process::Command::new("powershell")
            .args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-File"])
            .arg(&script_path)
            .creation_flags(0x08000000) // CREATE_NO_WINDOW
            .spawn();
    });

    Ok(())
}

// ── LLM Formatting commands ─────────────────────────────────────────────

#[tauri::command]
fn list_formats() -> Vec<serde_json::Value> {
    FormatType::all()
        .iter()
        .map(|f| {
            serde_json::json!({
                "id": f,
                "label": f.label(),
            })
        })
        .collect()
}

/// Get the instruction text for a format type (custom override or default).
#[tauri::command]
fn get_template(state: State<AppState>, format_type: String) -> serde_json::Value {
    let ft: FormatType = serde_json::from_value(serde_json::Value::String(format_type.clone()))
        .unwrap_or(FormatType::None);
    let customs = state.custom_templates.lock().unwrap();
    let custom = customs.get(&format_type);
    serde_json::json!({
        "instruction": custom.cloned().unwrap_or_else(|| ft.default_instruction().to_string()),
        "is_custom": custom.is_some(),
        "default": ft.default_instruction(),
    })
}

/// Save a custom instruction for a format type.
#[tauri::command]
fn set_template(state: State<AppState>, format_type: String, instruction: String) {
    let mut customs = state.custom_templates.lock().unwrap();
    customs.insert(format_type, instruction);
    dictator::templates::save_custom_templates(&customs);
}

/// Reset a format type back to its default instruction.
#[tauri::command]
fn reset_template(state: State<AppState>, format_type: String) {
    let mut customs = state.custom_templates.lock().unwrap();
    customs.remove(&format_type);
    dictator::templates::save_custom_templates(&customs);
}

#[tauri::command]
fn get_llm_status(state: State<AppState>) -> serde_json::Value {
    let status = *state.llm_status.lock().unwrap();
    let error = state.llm_error.lock().unwrap().clone();
    serde_json::json!({
        "loaded": status,
        "error": error,
    })
}

/// Lazy-load the LLM and format text. First call triggers model loading.
#[tauri::command]
fn format_text(
    state: State<AppState>,
    text: String,
    format: String,
) -> Result<String, String> {
    let format_type: FormatType =
        serde_json::from_value(serde_json::Value::String(format))
            .map_err(|_| "Unknown format type".to_string())?;

    if format_type == FormatType::None {
        return Ok(text);
    }

    // Lazy-load the LLM on first format request
    {
        let mut guard = state.formatter.lock().unwrap();
        if guard.is_none() {
            let model_path = llm::find_llm_model_path()
                .ok_or("No LLM model found. Place a .gguf file in the models/ folder.")?;
            let force_cpu = *state.force_cpu.lock().unwrap();

            match llm::Formatter::new(&model_path, force_cpu) {
                Ok(f) => {
                    *state.llm_using_gpu.lock().unwrap() = Some(f.is_using_gpu());
                    *state.llm_status.lock().unwrap() = Some(true);
                    *guard = Some(f);
                }
                Err(e) => {
                    let msg = format!("{e}");
                    *state.llm_using_gpu.lock().unwrap() = Some(false);
                    *state.llm_status.lock().unwrap() = Some(false);
                    *state.llm_error.lock().unwrap() = Some(msg.clone());
                    return Err(msg);
                }
            }
        }
    }

    let guard = state.formatter.lock().unwrap();
    let formatter = guard.as_ref().ok_or("LLM not loaded")?;

    // Check for a custom template override, then layer the user's strict-
    // cleanup notes on top (only applied when format_type == StrictCleanUp).
    let customs = state.custom_templates.lock().unwrap();
    let custom_instruction = customs.get(format_type.key()).cloned();
    drop(customs);

    let notes = state.strict_cleanup_notes.lock().unwrap().clone();
    let merged = apply_strict_notes(format_type, custom_instruction, &notes);

    formatter
        .format_text_custom(&text, format_type, merged.as_deref())
        .map_err(|e| format!("{e}"))
}

// ── Reformat selection in-place ─────────────────────────────────────────

/// Grab highlighted text via Ctrl+C, format with LLM, paste back via Ctrl+V.
#[tauri::command]
fn reformat_selection(state: State<AppState>) -> Result<String, String> {
    use arboard::Clipboard;
    use windows::Win32::UI::Input::KeyboardAndMouse::{
        SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYBD_EVENT_FLAGS,
        KEYEVENTF_KEYUP, VIRTUAL_KEY,
    };

    let vk_control = VIRTUAL_KEY(0x11);
    let vk_shift = VIRTUAL_KEY(0x10);
    let vk_alt = VIRTUAL_KEY(0x12);
    let vk_c = VIRTUAL_KEY(0x43);
    let vk_v = VIRTUAL_KEY(0x56);

    fn key_up(vk: VIRTUAL_KEY) -> INPUT {
        INPUT {
            r#type: INPUT_KEYBOARD,
            Anonymous: INPUT_0 {
                ki: KEYBDINPUT {
                    wVk: vk, wScan: 0, dwFlags: KEYEVENTF_KEYUP, time: 0, dwExtraInfo: 0,
                },
            },
        }
    }
    fn key_down(vk: VIRTUAL_KEY) -> INPUT {
        INPUT {
            r#type: INPUT_KEYBOARD,
            Anonymous: INPUT_0 {
                ki: KEYBDINPUT {
                    wVk: vk, wScan: 0, dwFlags: KEYBD_EVENT_FLAGS(0), time: 0, dwExtraInfo: 0,
                },
            },
        }
    }

    // Brief delay so the user's hotkey modifier keys (Ctrl+Shift+F) are
    // physically released before we simulate Ctrl+C.
    std::thread::sleep(std::time::Duration::from_millis(150));

    // 1. Save current clipboard, then clear it so we can detect Ctrl+C
    //    reliably. The old comparison (`selected == old_text`) silently
    //    fails when the clipboard already holds the same text as the
    //    selection — common after a paste-mode dictation.
    let old_text = {
        let mut cb = Clipboard::new().map_err(|e| format!("Clipboard error: {e}"))?;
        let saved = cb.get_text().unwrap_or_default();
        let _ = cb.set_text(String::new()); // clear
        saved
    };

    // 2. Ctrl+C to grab selection
    let copy_inputs = [
        key_up(vk_shift), key_up(vk_alt), key_up(vk_control),
        key_down(vk_control), key_down(vk_c), key_up(vk_c), key_up(vk_control),
    ];
    unsafe { SendInput(&copy_inputs, std::mem::size_of::<INPUT>() as i32); }
    std::thread::sleep(std::time::Duration::from_millis(300));

    // 3. Read selected text — clipboard was cleared, so any content
    //    means Ctrl+C succeeded regardless of what was there before.
    let selected = {
        let mut cb = Clipboard::new().map_err(|e| format!("Clipboard error: {e}"))?;
        cb.get_text().unwrap_or_default()
    };
    if selected.is_empty() {
        // Ctrl+C didn't capture anything — restore original clipboard
        if !old_text.is_empty() {
            if let Ok(mut cb) = Clipboard::new() { let _ = cb.set_text(&old_text); }
        }
        return Err("No text selected".to_string());
    }

    // 4. Format with LLM — uses whatever format type is currently selected
    //    (cleanup, email, bullet points, etc.)
    let format_type_key = state.auto_format_type.lock().unwrap().clone();
    let format_type: FormatType =
        serde_json::from_value(serde_json::Value::String(format_type_key.clone()))
            .unwrap_or(FormatType::CleanUp);

    // Lazy-load the LLM
    {
        let mut guard = state.formatter.lock().unwrap();
        if guard.is_none() {
            let model_path = llm::find_llm_model_path().ok_or_else(|| {
                if let Ok(mut cb) = Clipboard::new() { let _ = cb.set_text(&old_text); }
                "No LLM model found".to_string()
            })?;
            let force_cpu = *state.force_cpu.lock().unwrap();
            match llm::Formatter::new(&model_path, force_cpu) {
                Ok(f) => {
                    *state.llm_using_gpu.lock().unwrap() = Some(f.is_using_gpu());
                    *state.llm_status.lock().unwrap() = Some(true);
                    *guard = Some(f);
                }
                Err(e) => {
                    if let Ok(mut cb) = Clipboard::new() { let _ = cb.set_text(&old_text); }
                    *state.llm_using_gpu.lock().unwrap() = Some(false);
                    *state.llm_status.lock().unwrap() = Some(false);
                    return Err(format!("{e}"));
                }
            }
        }
    }

    let guard = state.formatter.lock().unwrap();
    let formatter = guard.as_ref().ok_or("LLM not loaded")?;
    let customs = state.custom_templates.lock().unwrap();
    let custom_instruction = customs.get(format_type.key()).cloned();
    drop(customs);

    // Layer the user's strict-cleanup notes on top of any custom override.
    // Only fires when format_type == StrictCleanUp; other cleanup tiers
    // passed through this hotkey get their default / custom prompt unchanged.
    let notes = state.strict_cleanup_notes.lock().unwrap().clone();
    let merged = apply_strict_notes(format_type, custom_instruction, &notes);

    let formatted = match formatter.format_text_custom(&selected, format_type, merged.as_deref()) {
        Ok(text) => text,
        Err(e) => {
            drop(guard);
            if let Ok(mut cb) = Clipboard::new() { let _ = cb.set_text(&old_text); }
            return Err(format!("{e}"));
        }
    };
    drop(guard);

    // 5. Set clipboard to formatted text and Ctrl+V to paste
    {
        let mut cb = Clipboard::new().map_err(|e| format!("Clipboard error: {e}"))?;
        cb.set_text(&formatted).map_err(|e| format!("Clipboard error: {e}"))?;
    }
    std::thread::sleep(std::time::Duration::from_millis(100));

    let paste_inputs = [
        key_up(vk_shift), key_up(vk_alt), key_up(vk_control),
        key_down(vk_control), key_down(vk_v), key_up(vk_v), key_up(vk_control),
    ];
    unsafe { SendInput(&paste_inputs, std::mem::size_of::<INPUT>() as i32); }
    std::thread::sleep(std::time::Duration::from_millis(300));

    // 6. Restore original clipboard
    if !old_text.is_empty() {
        if let Ok(mut cb) = Clipboard::new() { let _ = cb.set_text(&old_text); }
    }

    Ok(formatted)
}

// ── Single-instance enforcement ─────────────────────────────────────────

/// Try to acquire a system-wide named mutex. Returns the handle on success.
/// If another instance already holds it, shows a message and exits.
fn enforce_single_instance() -> windows::Win32::Foundation::HANDLE {
    use windows::core::w;
    use windows::Win32::Foundation::{GetLastError, ERROR_ALREADY_EXISTS, HANDLE};
    use windows::Win32::System::Threading::CreateMutexW;

    let handle: HANDLE = unsafe {
        CreateMutexW(None, true, w!("Global\\DictatorSingleInstance"))
            .unwrap_or(HANDLE::default())
    };

    if handle.is_invalid() || unsafe { GetLastError() } == ERROR_ALREADY_EXISTS {
        use windows::Win32::UI::WindowsAndMessaging::{MessageBoxW, MB_OK, MB_ICONINFORMATION};
        unsafe {
            MessageBoxW(
                None,
                w!("Dictator is already running.\nCheck the system tray (bottom-right of the taskbar)."),
                w!("Dictator"),
                MB_OK | MB_ICONINFORMATION,
            );
        }
        std::process::exit(0);
    }

    handle
}

// ── App entry point ─────────────────────────────────────────────────────

fn main() {
    // Install a panic hook that appends to `panic.log` next to the exe
    // before forwarding to the default hook (which writes to stderr that
    // a packaged Windows GUI app discards). Crashes in background threads
    // — VAD, wake-word listener, cpal callbacks — used to be invisible;
    // now we can read the log to see what happened.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                let log_path = dir.join("panic.log");
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open(&log_path)
                {
                    use std::io::Write;
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    let _ = writeln!(f, "[{now}] {info}");
                    let _ = writeln!(f, "{}", std::backtrace::Backtrace::force_capture());
                    let _ = writeln!(f, "---");
                }
            }
        }
        default_hook(info);
    }));

    // Install the TLS crypto provider before anything else — both ring and
    // aws-lc-rs features are activated by our dependency tree (ureq + axum-server),
    // so rustls can't auto-detect which one to use.
    let _ = rustls::crypto::ring::default_provider().install_default();

    // Check for a leftover GPU crash marker from the previous session. If found,
    // GPU will be skipped this session and the UI will offer a "Try GPU Again"
    // affordance. Must run before any GPU work is attempted.
    gpu_guard::init();

    let _mutex = enforce_single_instance();
    let model_path = model_manager::find_whisper_model();
    let transcriber: Arc<Mutex<Option<transcription::Transcriber>>> = Arc::new(Mutex::new(None));
    let formatter: Arc<Mutex<Option<llm::Formatter>>> = Arc::new(Mutex::new(None));
    let injection_mode: Arc<Mutex<String>> = Arc::new(Mutex::new("paste".into()));
    let auto_space: Arc<Mutex<bool>> = Arc::new(Mutex::new(true)); // on by default
    let auto_format: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));
    let auto_format_type: Arc<Mutex<String>> = Arc::new(Mutex::new("clean_up".into()));
    let live_context: Arc<Mutex<bool>> = Arc::new(Mutex::new(true)); // on by default
    let live_preview: Arc<Mutex<bool>> = Arc::new(Mutex::new(false)); // off by default — GPU recommended
    let connected_phones: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
    let using_gpu: Arc<Mutex<Option<bool>>> = Arc::new(Mutex::new(None));
    let llm_using_gpu: Arc<Mutex<Option<bool>>> = Arc::new(Mutex::new(None));
    let llm_status: Arc<Mutex<Option<bool>>> = Arc::new(Mutex::new(None));
    // Read from disk so the startup whisper loader below picks the
    // right GPU/CPU mode on the first try. Without this, we'd default to
    // `false`, load GPU, then have the webview's localStorage sync flip
    // us to `true` and clear the just-loaded model — UI ends up stuck on
    // "Loading whisper model…". See `user_prefs` module for the full
    // race description.
    let force_cpu: Arc<Mutex<bool>> = Arc::new(Mutex::new(user_prefs::load_force_cpu()));
    let selected_device: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    // Hands-free recording features. Both default off; the UI restores
    // saved preferences from localStorage on startup. Defaults chosen so
    // existing users see no behaviour change unless they opt in.
    let auto_stop_enabled: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));
    let auto_stop_secs: Arc<Mutex<f32>> = Arc::new(Mutex::new(2.0));
    let wake_word_phrase: Arc<Mutex<String>> =
        Arc::new(Mutex::new(wake_word::WakePhrase::HeyJarvis.id().to_string()));
    let wake_word_threshold: Arc<Mutex<f32>> = Arc::new(Mutex::new(0.50));

    // Layer 2 of GPU resilience: subscribe to Windows suspend/resume events so
    // we can unload the models before they attempt to run on a stale Vulkan
    // context. Failure here is non-fatal — Layer 1 (gpu_guard crash marker)
    // still protects against crash loops even without proactive notifications.
    if let Err(e) = power_events::register(
        transcriber.clone(),
        formatter.clone(),
        using_gpu.clone(),
        llm_using_gpu.clone(),
        llm_status.clone(),
    ) {
        eprintln!("power_events: registration failed ({e}) — continuing without sleep/resume detection");
    }

    // ── Networking strategy ──
    //
    // Two servers, both opt-in via Settings toggles:
    //   Port 3456 — LAN (self-signed cert, same WiFi only)
    //   Port 3457 — Tailscale (trusted cert, works from anywhere)
    //
    // The app defaults to fully local on a fresh install. The UI restores
    // saved preferences on startup and invokes start_lan_server /
    // refresh_tailscale if those were previously enabled.

    tauri::Builder::default()
        .plugin(tauri_plugin_notification::init())
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, shortcut, event| {
                    if event.state == tauri_plugin_global_shortcut::ShortcutState::Pressed {
                        if let Some(win) = app.get_webview_window("main") {
                            // Check if this is the dynamically registered Escape key
                            use tauri_plugin_global_shortcut::{Code, Shortcut as GShortcut};
                            let escape_sc = GShortcut::new(None, Code::Escape);
                            if *shortcut == escape_sc {
                                let _ = win.emit("escape-pressed", ());
                                return;
                            }

                            let state = app.try_state::<AppState>();
                            let inject_key = state.as_ref().and_then(|s| {
                                parse_hotkey_preset(&s.inject_hotkey.lock().unwrap())
                            });
                            let tts_key = state.as_ref().and_then(|s| {
                                parse_hotkey_preset(&s.tts_hotkey.lock().unwrap())
                            });
                            let reformat_key = state.as_ref().and_then(|s| {
                                parse_hotkey_preset(&s.reformat_hotkey.lock().unwrap())
                            });
                            let copy_key = state.as_ref().and_then(|s| {
                                parse_hotkey_preset(&s.copy_hotkey.lock().unwrap())
                            });

                            if inject_key.is_some_and(|k| k == *shortcut) {
                                let _ = win.emit("inject-at-cursor", ());
                            } else if tts_key.is_some_and(|k| k == *shortcut) {
                                let _ = win.emit("tts-read-aloud", ());
                            } else if reformat_key.is_some_and(|k| k == *shortcut) {
                                let _ = win.emit("reformat-selection", ());
                            } else if copy_key.is_some_and(|k| k == *shortcut) {
                                let _ = win.emit("copy-last-to-clipboard", ());
                            } else {
                                let _ = win.emit("toggle-recording", ());
                            }
                        }
                    }
                })
                .build(),
        )
        .manage(AppState {
            transcriber: transcriber.clone(),
            recording: Mutex::new(None),
            injection_mode: injection_mode.clone(),
            auto_space: auto_space.clone(),
            auto_format: auto_format.clone(),
            auto_format_type: auto_format_type.clone(),
            model_error: Mutex::new(None),
            server_url: Mutex::new(None),
            connected_phones: connected_phones.clone(),
            cert_type: Mutex::new("self-signed".to_string()),
            tailscale_url: Mutex::new(None),
            using_gpu: using_gpu.clone(),
            llm_using_gpu: llm_using_gpu.clone(),
            force_cpu: force_cpu.clone(),
            selected_device: selected_device.clone(),
            hotkey: Mutex::new("Ctrl+Shift+Space".into()),
            inject_hotkey: Mutex::new("none".into()),
            tts_hotkey: Mutex::new("none".into()),
            reformat_hotkey: Mutex::new("none".into()),
            copy_hotkey: Mutex::new("none".into()),
            speaker: Arc::new(Mutex::new(None)),
            piper: Arc::new(tts::PiperSpeaker::new(
                model_manager::models_dir().join("piper"),
                model_manager::models_dir().join("piper-voices"),
            )),
            tts_engine: Mutex::new("sapi".into()),
            minimax_api_key: Mutex::new(String::new()),
            minimax_voice: Mutex::new("English_Graceful_Lady".into()),
            minimax_playing: Arc::new(AtomicBool::new(false)),
            minimax_sink: Arc::new(Mutex::new(None)),
            formatter: formatter.clone(),
            llm_status: llm_status.clone(),
            llm_error: Mutex::new(None),
            no_model: Mutex::new(false),
            tailscale_server_running: Arc::new(AtomicBool::new(false)),
            lan_server_handle: Mutex::new(None),
            server_shared: Mutex::new(None),
            tailscale_cert_generated: Mutex::new(None),
            custom_templates: Arc::new(Mutex::new(
                dictator::templates::load_custom_templates(),
            )),
            strict_cleanup_notes: Arc::new(Mutex::new(load_strict_cleanup_notes())),
            live_context: live_context.clone(),
            live_preview: live_preview.clone(),
            live_session: Mutex::new(None),
            auto_stop_enabled: auto_stop_enabled.clone(),
            auto_stop_secs: auto_stop_secs.clone(),
            wake_word_phrase: wake_word_phrase.clone(),
            wake_word_threshold: wake_word_threshold.clone(),
            wake_word_handle: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![
            get_status,
            get_server_url,
            get_connected_phones,
            get_cert_type,
            get_tailscale_url,
            refresh_tailscale,
            start_lan_server,
            stop_lan_server,
            get_lan_server_running,
            get_tailscale_status,
            get_auto_space,
            set_auto_space,
            get_auto_format,
            set_auto_format,
            get_auto_format_type,
            set_auto_format_type,
            get_live_context,
            set_live_context,
            get_live_preview,
            set_live_preview,
            get_gpu_status,
            get_force_cpu,
            set_force_cpu,
            get_auto_stop,
            set_auto_stop,
            get_auto_stop_secs,
            set_auto_stop_secs,
            list_wake_word_phrases,
            get_wake_word_phrase_id,
            set_wake_word_phrase_id,
            get_wake_word_threshold,
            set_wake_word_threshold,
            start_wake_word_listener,
            stop_wake_word_listener,
            get_wake_word_listener_running,
            get_word_bank,
            set_word_bank,
            get_history,
            add_history_entry,
            delete_history_entry,
            clear_history,
            get_strict_cleanup_notes,
            set_strict_cleanup_notes,
            retry_gpu,
            list_audio_devices,
            get_selected_device,
            set_selected_device,
            start_recording,
            stop_and_transcribe,
            cancel_recording,
            get_recording_level,
            start_live_session,
            stop_live_session,
            register_escape_shortcut,
            unregister_escape_shortcut,
            inject_text,
            set_injection_mode,
            get_autostart,
            set_autostart,
            fix_autostart,
            get_hotkey,
            set_hotkey,
            get_inject_hotkey,
            set_inject_hotkey,
            get_tts_hotkey,
            set_tts_hotkey,
            get_reformat_hotkey,
            set_reformat_hotkey,
            get_copy_hotkey,
            set_copy_hotkey,
            copy_text_to_clipboard,
            reformat_selection,
            speak_text,
            stop_speaking,
            read_selected_text,
            stop_tts,
            list_tts_voices,
            set_tts_voice,
            set_tts_rate,
            get_tts_engine,
            set_tts_engine,
            has_piper,
            list_piper_voices,
            get_piper_voice_dir,
            set_piper_voice,
            download_piper,
            download_piper_voice,
            delete_piper_voice,
            get_minimax_api_key,
            set_minimax_api_key,
            get_minimax_voice,
            set_minimax_voice,
            list_minimax_voices,
            test_minimax_api,
            open_path,
            show_notification,
            list_formats,
            get_template,
            set_template,
            reset_template,
            get_llm_status,
            format_text,
            has_llm_model,
            list_llm_models,
            list_whisper_models,
            download_whisper_model,
            load_whisper_model,
            list_available_models,
            download_llm_model,
            delete_llm_model,
            get_models_dir,
            validate_llm_model,
            set_model_chat_format,
            get_app_version,
            check_for_updates,
            uninstall_cleanup,
        ])
        .setup(move |app| {
            let transcriber_handle = transcriber.clone();
            let gpu_handle = using_gpu.clone();
            let force_cpu_handle = force_cpu.clone();
            let app_handle = app.handle().clone();

            // Load model in background (or signal no model available)
            if let Some(model_path) = model_path {
                std::thread::spawn(move || {
                    let fc = *force_cpu_handle.lock().unwrap();
                    eprintln!("Loading whisper model from: {}", model_path.display());

                    match transcription::Transcriber::new(&model_path, fc) {
                        Ok(t) => {
                            let gpu = t.is_using_gpu();
                            *gpu_handle.lock().unwrap() = Some(gpu);
                            *transcriber_handle.lock().unwrap() = Some(t);
                            eprintln!(
                                "Model loaded successfully ({})",
                                if gpu { "GPU" } else { "CPU" }
                            );
                            let _ = app_handle.emit("model-ready", "ok");
                        }
                        Err(e) => {
                            *gpu_handle.lock().unwrap() = Some(false);
                            let msg = format!("Failed to load model: {e}");
                            eprintln!("{msg}");
                            let _ = app_handle.emit("model-ready", &msg);
                            let state: State<AppState> = app_handle.state();
                            *state.model_error.lock().unwrap() = Some(msg);
                        }
                    }
                });
            } else {
                eprintln!("No whisper model found — waiting for user to download one");
                {
                    let state: State<AppState> = app_handle.state();
                    *state.no_model.lock().unwrap() = true;
                }
                let _ = app_handle.emit("model-ready", "no-model");
            }

            // Start the phone companion HTTPS servers
            // LAN on port 3456 (always, self-signed) + Tailscale on 3457 (if available)
            let server_transcriber = transcriber.clone();
            let server_formatter = formatter.clone();
            let server_injection = injection_mode.clone();
            let server_phones = connected_phones.clone();
            let server_auto_space = auto_space.clone();
            let server_auto_format = auto_format.clone();
            let server_auto_format_type = auto_format_type.clone();
            let server_gpu = using_gpu.clone();

            // Channel for phone to trigger TTS read-aloud on desktop
            let (tts_tx, tts_rx) = std::sync::mpsc::channel::<()>();
            {
                let app_handle_tts = app.handle().clone();
                std::thread::spawn(move || {
                    while tts_rx.recv().is_ok() {
                        let state: State<AppState> = app_handle_tts.state();
                        let engine = state.tts_engine.lock().unwrap().clone();
                        // Cycle: speaking → pause, paused → resume, idle → speak
                        if engine == "piper" {
                            if state.piper.is_speaking() {
                                if state.piper.is_paused() {
                                    state.piper.resume();
                                    let _ = app_handle_tts.emit("tts-state", "speaking");
                                } else {
                                    state.piper.pause();
                                    let _ = app_handle_tts.emit("tts-state", "paused");
                                }
                                continue;
                            }
                        } else if engine == "minimax" {
                            if state.minimax_playing.load(std::sync::atomic::Ordering::Relaxed) {
                                let sink = state.minimax_sink.lock().unwrap();
                                if let Some(s) = sink.as_ref() { s.stop(); }
                                state.minimax_playing.store(false, std::sync::atomic::Ordering::Relaxed);
                                let _ = app_handle_tts.emit("tts-state", "idle");
                                continue;
                            }
                        } else {
                            let guard = state.speaker.lock().unwrap();
                            if let Some(s) = guard.as_ref() {
                                if s.is_speaking() {
                                    let _ = s.stop();
                                    let _ = app_handle_tts.emit("tts-state", "idle");
                                    continue;
                                }
                            }
                        }
                        match tts::capture_selected_text() {
                            Ok(text) => {
                                let _ = app_handle_tts.emit("tts-text", text.as_str());
                                let _ = app_handle_tts.emit("tts-state", "speaking");
                                let res = if engine == "piper" {
                                    let result = state.piper.speak(&text);
                                    let _ = app_handle_tts.emit("tts-state", "idle");
                                    result
                                } else if engine == "minimax" {
                                    let api_key = state.minimax_api_key.lock().unwrap().clone();
                                    let voice_id = state.minimax_voice.lock().unwrap().clone();
                                    let h = app_handle_tts.clone();
                                    let result = tts::minimax_speak(
                                        &api_key, &voice_id, &text,
                                        &state.minimax_playing, &state.minimax_sink,
                                        move || { let _ = h.emit("tts-state", "speaking"); },
                                    );
                                    let _ = app_handle_tts.emit("tts-state", "idle");
                                    result
                                } else {
                                    let mut guard = state.speaker.lock().unwrap();
                                    if guard.is_none() {
                                        match tts::SapiSpeaker::new() {
                                            Ok(s) => { *guard = Some(s); }
                                            Err(e) => { eprintln!("SAPI init failed: {e}"); continue; }
                                        }
                                    }
                                    let result = guard.as_ref().unwrap().speak(&text);
                                    let _ = app_handle_tts.emit("tts-state", "idle");
                                    result
                                };
                                if let Err(e) = res {
                                    eprintln!("TTS speak failed: {e}");
                                }
                            }
                            Err(e) => eprintln!("Capture selected text failed: {e}"),
                        }
                    }
                });
            }

            // Store shared server state so refresh_tailscale can start the Tailscale server on demand
            {
                let state: State<AppState> = app.handle().state();
                *state.server_shared.lock().unwrap() = Some(ServerShared {
                    transcriber: server_transcriber.clone(),
                    formatter: server_formatter.clone(),
                    injection_mode: server_injection.clone(),
                    connected_phones: server_phones.clone(),
                    auto_space: server_auto_space.clone(),
                    auto_format: server_auto_format.clone(),
                    auto_format_type: server_auto_format_type.clone(),
                    using_gpu: server_gpu.clone(),
                    force_cpu: force_cpu.clone(),
                    live_context: live_context.clone(),
                    tts_trigger: tts_tx.clone(),
                });
            }

            // Daily cert renewal check — if Tailscale is enabled and cert is nearing expiry
            {
                let app_handle_renew = app.handle().clone();
                std::thread::spawn(move || {
                    loop {
                        // Check once per day
                        std::thread::sleep(std::time::Duration::from_secs(86400));
                        let state: State<AppState> = app_handle_renew.state();
                        let running = state.tailscale_server_running.load(Ordering::Relaxed);
                        if !running { continue; }

                        let needs_renewal = {
                            let cert_time = state.tailscale_cert_generated.lock().unwrap();
                            match *cert_time {
                                Some(t) => {
                                    let now = std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .unwrap_or_default().as_secs();
                                    let days = (now.saturating_sub(t)) / 86400;
                                    days >= 60 // Renew when 60 days old (30 days before expiry)
                                }
                                None => true, // No record — try to renew
                            }
                        };

                        if needs_renewal {
                            eprintln!("Tailscale cert nearing expiry — attempting renewal");
                            match invoke_refresh_tailscale(&app_handle_renew) {
                                Ok(msg) => eprintln!("Cert renewal: {msg}"),
                                Err(e) => eprintln!("Cert renewal failed: {e}"),
                            }
                        }
                    }
                });
            }

            // Both LAN and Tailscale servers are opt-in. The UI will invoke
            // start_lan_server on startup if the user's saved preference says
            // so — otherwise the app runs purely locally with no open ports.
            // Consume the server_* bindings so Rust doesn't warn about unused
            // variables (clones live on inside server_shared).
            let _ = (
                server_transcriber,
                server_formatter,
                server_injection,
                server_phones,
                server_auto_space,
                server_auto_format,
                server_auto_format_type,
                server_gpu,
                tts_tx,
            );

            // ── System tray — auto-format toggle, cancel recording, show/quit ───
            use tauri::menu::PredefinedMenuItem;

            // Read current format type from state so tray checkmark matches saved preference
            let current_fmt = auto_format_type.lock().unwrap().clone();
            let fmt_label = |id: &str, display: &str| -> String {
                if id == current_fmt { format!("  \u{2713} {display}") } else { format!("  {display}") }
            };

            let autoformat_item = tauri::menu::MenuItem::with_id(
                app, "autoformat", "Auto Format: Off", true, None::<&str>,
            )?;
            let fmt_light = tauri::menu::MenuItem::with_id(app, "fmt_light_clean_up", &fmt_label("light_clean_up", "Light Cleanup"), true, None::<&str>)?;
            let fmt_clean = tauri::menu::MenuItem::with_id(app, "fmt_clean_up", &fmt_label("clean_up", "Clean Up"), true, None::<&str>)?;
            let fmt_strict = tauri::menu::MenuItem::with_id(app, "fmt_strict_clean_up", &fmt_label("strict_clean_up", "Strict Cleanup"), true, None::<&str>)?;
            let fmt_casual_email = tauri::menu::MenuItem::with_id(app, "fmt_casual_email", &fmt_label("casual_email", "Casual Email"), true, None::<&str>)?;
            let fmt_pro_email = tauri::menu::MenuItem::with_id(app, "fmt_professional_email", &fmt_label("professional_email", "Professional Email"), true, None::<&str>)?;
            let fmt_letter = tauri::menu::MenuItem::with_id(app, "fmt_formal_letter", &fmt_label("formal_letter", "Formal Letter"), true, None::<&str>)?;
            let fmt_bullet = tauri::menu::MenuItem::with_id(app, "fmt_bullet_summary", &fmt_label("bullet_summary", "Bullet Summary"), true, None::<&str>)?;
            let fmt_notes = tauri::menu::MenuItem::with_id(app, "fmt_meeting_notes", &fmt_label("meeting_notes", "Meeting Notes"), true, None::<&str>)?;
            let fmt_docs = tauri::menu::MenuItem::with_id(app, "fmt_documentation", &fmt_label("documentation", "Documentation"), true, None::<&str>)?;
            let fmt_msg = tauri::menu::MenuItem::with_id(app, "fmt_message", &fmt_label("message", "Message"), true, None::<&str>)?;

            let cleanup_sub = tauri::menu::Submenu::with_items(app, "Cleanup", true, &[
                &fmt_light, &fmt_clean, &fmt_strict,
            ])?;
            let email_sub = tauri::menu::Submenu::with_items(app, "Email", true, &[
                &fmt_casual_email, &fmt_pro_email,
            ])?;

            let sep1 = PredefinedMenuItem::separator(app)?;
            let cancel_rec_item = tauri::menu::MenuItem::with_id(
                app, "cancel_recording", "Cancel Recording", true, None::<&str>,
            )?;
            // Free GPU toggle — mirrors the main-screen toggle and Settings
            // "Force CPU mode" setting. Label is updated dynamically to show
            // the current state ("Free GPU: On" / "Free GPU: Off").
            let sep_gpu = PredefinedMenuItem::separator(app)?;
            let free_gpu_item = tauri::menu::MenuItem::with_id(
                app, "free_gpu", "Free GPU: Off", true, None::<&str>,
            )?;
            let sep2 = PredefinedMenuItem::separator(app)?;
            let show_item = tauri::menu::MenuItem::with_id(app, "show", "Show Dictator", true, None::<&str>)?;
            let quit_item = tauri::menu::MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;

            // Cancel Recording is inserted/removed dynamically via recording-state-changed.
            // Format items (Cleanup, Email, Formal Letter, etc.) are also dynamic:
            // they appear only when Auto Format is ON, to keep the menu compact
            // for users who don't use auto-formatting. Default state is OFF, so
            // the initial menu excludes them; the UI emits sync-tray-autoformat
            // on startup and the listener below inserts them if saved=true.
            let tray_menu = tauri::menu::Menu::with_items(app, &[
                &autoformat_item,
                &sep_gpu,
                &free_gpu_item,
                &sep2,
                &show_item,
                &quit_item,
            ])?;

            // Tracks whether the format items are currently present in the menu,
            // so the insert/remove logic stays correct across multiple toggles
            // and the recording handler knows where sep2 lives.
            let fmt_items_visible = Arc::new(std::sync::atomic::AtomicBool::new(false));

            // Clones for the various closures that need menu item access
            let autoformat_item2 = autoformat_item.clone();
            let autoformat_item3 = autoformat_item.clone();
            let fmt_light2 = fmt_light.clone();
            let fmt_clean2 = fmt_clean.clone();
            let fmt_strict2 = fmt_strict.clone();
            let fmt_casual_email2 = fmt_casual_email.clone();
            let fmt_pro_email2 = fmt_pro_email.clone();
            let fmt_letter2 = fmt_letter.clone();
            let fmt_bullet2 = fmt_bullet.clone();
            let fmt_notes2 = fmt_notes.clone();
            let fmt_docs2 = fmt_docs.clone();
            let fmt_msg2 = fmt_msg.clone();
            // free_gpu: one clone used inside on_menu_event, another inside the
            // sync listener so the menu label reflects UI-initiated changes.
            let free_gpu_item2 = free_gpu_item.clone();
            let free_gpu_item3 = free_gpu_item.clone();

            // Shared helper: insert or remove the format items from the tray
            // menu based on the new Auto Format state. Used by the tray menu
            // click handler, the left-click handler, and the sync listener so
            // all three entry points keep the menu in the same shape.
            //
            // Layout when fmt items are visible:
            //   0 autoformat, 1 cleanup_sub, 2 email_sub,
            //   3..7 fmt_letter/bullet/notes/docs/msg,
            //   8 sep_gpu, 9 free_gpu, 10 sep2, 11 show, 12 quit
            // Layout when hidden:
            //   0 autoformat, 1 sep_gpu, 2 free_gpu, 3 sep2, 4 show, 5 quit
            let set_fmt_visible: Arc<dyn Fn(bool) + Send + Sync> = {
                let cleanup_sub = cleanup_sub.clone();
                let email_sub = email_sub.clone();
                let fmt_letter = fmt_letter.clone();
                let fmt_bullet = fmt_bullet.clone();
                let fmt_notes = fmt_notes.clone();
                let fmt_docs = fmt_docs.clone();
                let fmt_msg = fmt_msg.clone();
                let tray_menu = tray_menu.clone();
                let fmt_items_visible = fmt_items_visible.clone();
                Arc::new(move |enabled: bool| {
                    let was_visible = fmt_items_visible.load(std::sync::atomic::Ordering::SeqCst);
                    if enabled && !was_visible {
                        let _ = tray_menu.insert(&cleanup_sub, 1);
                        let _ = tray_menu.insert(&email_sub, 2);
                        let _ = tray_menu.insert(&fmt_letter, 3);
                        let _ = tray_menu.insert(&fmt_bullet, 4);
                        let _ = tray_menu.insert(&fmt_notes, 5);
                        let _ = tray_menu.insert(&fmt_docs, 6);
                        let _ = tray_menu.insert(&fmt_msg, 7);
                        fmt_items_visible.store(true, std::sync::atomic::Ordering::SeqCst);
                    } else if !enabled && was_visible {
                        let _ = tray_menu.remove(&cleanup_sub);
                        let _ = tray_menu.remove(&email_sub);
                        let _ = tray_menu.remove(&fmt_letter);
                        let _ = tray_menu.remove(&fmt_bullet);
                        let _ = tray_menu.remove(&fmt_notes);
                        let _ = tray_menu.remove(&fmt_docs);
                        let _ = tray_menu.remove(&fmt_msg);
                        fmt_items_visible.store(false, std::sync::atomic::Ordering::SeqCst);
                    }
                })
            };
            let set_fmt_visible_menu = set_fmt_visible.clone();
            let set_fmt_visible_tray = set_fmt_visible.clone();
            let set_fmt_visible_fmtpick = set_fmt_visible.clone();
            let set_fmt_visible_sync = set_fmt_visible.clone();
            // Used by the recording-state listener to compute where the cancel
            // block should insert (sep2 is at index 3 when fmt items are hidden,
            // index 10 when visible).
            let fmt_items_visible_rec = fmt_items_visible.clone();

            // Precompute idle (green) and recording (red) tray icons. The red
            // variant is generated from the default icon by swapping the R and
            // G channels per pixel — the cap goes green→red while yellow
            // accents and transparency stay intact.
            //
            // We rebuild both as owned `'static` images so they can be moved
            // into the recording-state-changed listener closure without
            // dragging `app`'s lifetime with them.
            let default_icon_ref = app
                .default_window_icon()
                .expect("Missing app icon");
            let icon_width = default_icon_ref.width();
            let icon_height = default_icon_ref.height();
            let idle_rgba: Vec<u8> = default_icon_ref.rgba().to_vec();
            let idle_icon: tauri::image::Image<'static> =
                tauri::image::Image::new_owned(idle_rgba.clone(), icon_width, icon_height);
            let recording_icon: tauri::image::Image<'static> = {
                let mut rgba = idle_rgba;
                for px in rgba.chunks_exact_mut(4) {
                    px.swap(0, 1);
                }
                tauri::image::Image::new_owned(rgba, icon_width, icon_height)
            };

            let tray = tauri::tray::TrayIconBuilder::new()
                .icon(idle_icon.clone())
                .tooltip("Dictator")
                .menu(&tray_menu)
                .on_menu_event(move |app, event| {
                    let id = event.id().as_ref().to_string();
                    match id.as_str() {
                        "autoformat" => {
                            let state: State<AppState> = app.state();
                            let currently_on = *state.auto_format.lock().unwrap();

                            if !currently_on && !model_manager::has_llm_model() {
                                // No model yet — show the app and trigger download
                                if let Some(win) = app.get_webview_window("main") {
                                    let _ = win.show();
                                    let _ = win.set_focus();
                                    let _ = win.emit("tray-autoformat-needs-model", ());
                                }
                                return;
                            }

                            // Toggle auto-format on/off
                            let mut af = state.auto_format.lock().unwrap();
                            *af = !*af;
                            let enabled = *af;
                            drop(af);
                            if let Some(win) = app.get_webview_window("main") {
                                let _ = win.emit("autoformat-changed", enabled);
                            }
                            let label = if enabled { "Auto Format: On" } else { "Auto Format: Off" };
                            let _ = autoformat_item.set_text(label);
                            // Show/hide the format submenus + items to match.
                            set_fmt_visible_menu(enabled);
                        }
                        "cancel_recording" => {
                            if let Some(win) = app.get_webview_window("main") {
                                let _ = win.emit("cancel-recording", ());
                            }
                        }
                        "show" => {
                            if let Some(win) = app.get_webview_window("main") {
                                let _ = win.show();
                                let _ = win.set_focus();
                            }
                        }
                        "quit" => {
                            // Clear the GPU crash marker — we're exiting cleanly,
                            // so next launch should be allowed to try GPU again.
                            gpu_guard::disarm();
                            app.exit(0);
                        }
                        "free_gpu" => {
                            // Toggle Force CPU mode from the tray. Mirrors the
                            // main-screen "Free GPU" toggle and Settings "Force
                            // CPU mode" — they all read/write the same state.
                            let state: State<AppState> = app.state();
                            let new_state = {
                                let mut fc = state.force_cpu.lock().unwrap();
                                *fc = !*fc;
                                *fc
                            };
                            // Drop loaded models so the new preference takes
                            // effect on next use (VRAM is reclaimed for games
                            // immediately when toggling ON).
                            *state.transcriber.lock().unwrap() = None;
                            *state.formatter.lock().unwrap() = None;
                            *state.using_gpu.lock().unwrap() = None;
                            *state.llm_using_gpu.lock().unwrap() = None;
                            *state.llm_status.lock().unwrap() = None;

                            let label = if new_state { "Free GPU: On" } else { "Free GPU: Off" };
                            let _ = free_gpu_item2.set_text(label);

                            // Tell the UI so its toggles, localStorage, and
                            // status row stay in sync.
                            if let Some(win) = app.get_webview_window("main") {
                                let _ = win.emit("freegpu-changed", new_state);
                            }
                        }
                        _ if id.starts_with("fmt_") => {
                            // Format type selected — check model first
                            if !model_manager::has_llm_model() {
                                if let Some(win) = app.get_webview_window("main") {
                                    let _ = win.show();
                                    let _ = win.set_focus();
                                    let _ = win.emit("tray-autoformat-needs-model", ());
                                }
                                return;
                            }

                            let fmt_type = id.strip_prefix("fmt_").unwrap().to_string();
                            let state: State<AppState> = app.state();
                            *state.auto_format_type.lock().unwrap() = fmt_type.clone();
                            // Also enable auto-format
                            *state.auto_format.lock().unwrap() = true;
                            if let Some(win) = app.get_webview_window("main") {
                                let _ = win.emit("autoformat-changed", true);
                                let _ = win.emit("autoformat-type-changed", fmt_type.as_str());
                            }
                            let _ = autoformat_item.set_text("Auto Format: On");
                            // Ensure fmt items are visible (they must be, since
                            // the user just clicked one — this is defensive).
                            set_fmt_visible_fmtpick(true);
                            // Update checkmarks
                            let types = [
                                (&fmt_light, "light_clean_up", "Light Cleanup"),
                                (&fmt_clean, "clean_up", "Clean Up"),
                                (&fmt_strict, "strict_clean_up", "Strict Cleanup"),
                                (&fmt_casual_email, "casual_email", "Casual Email"),
                                (&fmt_pro_email, "professional_email", "Professional Email"),
                                (&fmt_letter, "formal_letter", "Formal Letter"),
                                (&fmt_bullet, "bullet_summary", "Bullet Summary"),
                                (&fmt_notes, "meeting_notes", "Meeting Notes"),
                                (&fmt_docs, "documentation", "Documentation"),
                                (&fmt_msg, "message", "Message"),
                            ];
                            for (item, name, display) in &types {
                                let prefix = if *name == fmt_type { "  ✓ " } else { "  " };
                                let _ = item.set_text(&format!("{prefix}{display}"));
                            }
                        }
                        _ => {}
                    }
                })
                .on_tray_icon_event(move |tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        let app = tray.app_handle();
                        let state: State<AppState> = app.state();
                        let currently_on = *state.auto_format.lock().unwrap();

                        if !currently_on && !model_manager::has_llm_model() {
                            // No model yet — show the app and trigger download
                            if let Some(win) = app.get_webview_window("main") {
                                let _ = win.show();
                                let _ = win.set_focus();
                                let _ = win.emit("tray-autoformat-needs-model", ());
                            }
                            return;
                        }

                        // Left-click: toggle auto-format
                        let mut af = state.auto_format.lock().unwrap();
                        *af = !*af;
                        let enabled = *af;
                        drop(af);
                        if let Some(win) = app.get_webview_window("main") {
                            let _ = win.emit("autoformat-changed", enabled);
                        }
                        let label = if enabled { "Auto Format: On" } else { "Auto Format: Off" };
                        let _ = autoformat_item2.set_text(label);
                        // Show/hide fmt items to match.
                        set_fmt_visible_tray(enabled);
                    }
                })
                .build(app)?;

            // Sync tray checkmarks when Settings changes the format type
            {
                use tauri::Listener;
                app.listen("sync-tray-format", move |event| {
                    let fmt_type = event.payload().trim_matches('"');
                    let items: [(&tauri::menu::MenuItem<tauri::Wry>, &str, &str); 10] = [
                        (&fmt_light2, "light_clean_up", "Light Cleanup"),
                        (&fmt_clean2, "clean_up", "Clean Up"),
                        (&fmt_strict2, "strict_clean_up", "Strict Cleanup"),
                        (&fmt_casual_email2, "casual_email", "Casual Email"),
                        (&fmt_pro_email2, "professional_email", "Professional Email"),
                        (&fmt_letter2, "formal_letter", "Formal Letter"),
                        (&fmt_bullet2, "bullet_summary", "Bullet Summary"),
                        (&fmt_notes2, "meeting_notes", "Meeting Notes"),
                        (&fmt_docs2, "documentation", "Documentation"),
                        (&fmt_msg2, "message", "Message"),
                    ];
                    for (item, id, display) in &items {
                        let prefix = if *id == fmt_type { "  \u{2713} " } else { "  " };
                        let _ = item.set_text(&format!("{prefix}{display}"));
                    }
                });

                app.listen("sync-tray-autoformat", move |event| {
                    let enabled = event.payload().trim_matches('"') == "true";
                    let label = if enabled { "Auto Format: On" } else { "Auto Format: Off" };
                    let _ = autoformat_item3.set_text(label);
                    // Show/hide fmt items to match (lets saved state on startup
                    // rebuild the menu shape from what the UI reports).
                    set_fmt_visible_sync(enabled);
                });

                // UI (or init) tells the tray what the Free GPU state is so
                // the menu label stays accurate without us querying AppState.
                app.listen("sync-tray-freegpu", move |event| {
                    let enabled = event.payload().trim_matches('"') == "true";
                    let label = if enabled { "Free GPU: On" } else { "Free GPU: Off" };
                    let _ = free_gpu_item3.set_text(label);
                });

                // Dynamic "Cancel Recording" — insert/remove based on recording state.
                //
                // The cancel block (sep1 + cancel_rec_item) goes just before
                // sep2. Because the fmt items are also dynamic, sep2's index
                // depends on Auto Format state:
                //   fmt hidden  -> sep2 at index 3  (af, sep_gpu, free_gpu, sep2, ...)
                //   fmt visible -> sep2 at index 10 (af, 7 fmt items, sep_gpu, free_gpu, sep2, ...)
                let cancel_sep_dyn = sep1.clone();
                let cancel_rec_dyn = cancel_rec_item.clone();
                let tray_menu_dyn = tray_menu.clone();
                let tray_for_icon = tray.clone();
                let idle_icon_for_listener = idle_icon.clone();
                let recording_icon_for_listener = recording_icon.clone();
                app.listen("recording-state-changed", move |event| {
                    let recording = event.payload().trim_matches('"') == "true";
                    // Swap tray icon: red while recording, green while idle.
                    let icon = if recording {
                        recording_icon_for_listener.clone()
                    } else {
                        idle_icon_for_listener.clone()
                    };
                    let _ = tray_for_icon.set_icon(Some(icon));
                    if recording {
                        let fmt_visible = fmt_items_visible_rec
                            .load(std::sync::atomic::Ordering::SeqCst);
                        let sep_idx = if fmt_visible { 10 } else { 3 };
                        let _ = tray_menu_dyn.insert(&cancel_sep_dyn, sep_idx);
                        let _ = tray_menu_dyn.insert(&cancel_rec_dyn, sep_idx + 1);
                    } else {
                        let _ = tray_menu_dyn.remove(&cancel_rec_dyn);
                        let _ = tray_menu_dyn.remove(&cancel_sep_dyn);
                    }
                });
            }

            // Show the window unless launched with --minimized (autostart).
            // The window starts hidden (visible: false in tauri.conf.json) so
            // that autostart launches can go straight to the system tray.
            let window = app.get_webview_window("main").unwrap();
            let start_minimized = std::env::args().any(|a| a == "--minimized");
            if !start_minimized {
                let _ = window.show();
                let _ = window.set_focus();
            }

            // Hide to tray when the X button is clicked (instead of quitting)
            let win_hide = window.clone();
            window.on_window_event(move |event| {
                if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                    api.prevent_close();
                    let _ = win_hide.hide();
                }
            });

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running Dictator");
}

