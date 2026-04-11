#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use dictator::audio_capture;
use dictator::injection::{self, InjectionMode};
use dictator::llm;
use dictator::model_manager;
use dictator::server;
use dictator::tailscale;
use dictator::templates::FormatType;
use dictator::transcription;
use dictator::tts;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconEvent};
use tauri::{Emitter, Manager, State};

/// Recording session: a background thread captures audio into a shared buffer.
struct RecordingSession {
    stop_flag: Arc<AtomicBool>,
    samples: Arc<Mutex<Vec<f32>>>,
    thread: Option<std::thread::JoinHandle<()>>,
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
    /// None while loading, Some(true) for GPU, Some(false) for CPU
    using_gpu: Arc<Mutex<Option<bool>>>,
    /// Selected audio input device name (None = system default)
    selected_device: Arc<Mutex<Option<String>>>,
    /// Current hotkey preset name (e.g. "Ctrl+Shift+Space")
    hotkey: Mutex<String>,
    /// Inject-at-cursor hotkey preset
    inject_hotkey: Mutex<String>,
    /// Read-aloud hotkey preset
    tts_hotkey: Mutex<String>,
    /// SAPI TTS speaker (lazy-loaded on first speak)
    speaker: Arc<Mutex<Option<tts::SapiSpeaker>>>,
    /// Piper TTS speaker (neural voices)
    piper: Arc<tts::PiperSpeaker>,
    /// TTS engine: "sapi" or "piper"
    tts_engine: Mutex<String>,
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
    /// Shared server state pieces needed to start Tailscale server dynamically
    server_shared: Mutex<Option<ServerShared>>,
    /// When the Tailscale cert was last generated (epoch secs), for expiry tracking
    tailscale_cert_generated: Mutex<Option<u64>>,
    /// Custom template instructions (overrides defaults in templates.rs)
    custom_templates: Arc<Mutex<HashMap<String, String>>>,
}

/// Shared references needed to construct a ServerState for the dynamic Tailscale server.
struct ServerShared {
    transcriber: Arc<Mutex<Option<transcription::Transcriber>>>,
    formatter: Arc<Mutex<Option<llm::Formatter>>>,
    injection_mode: Arc<Mutex<String>>,
    connected_phones: Arc<Mutex<u32>>,
    auto_space: Arc<Mutex<bool>>,
    auto_format: Arc<Mutex<bool>>,
    auto_format_type: Arc<Mutex<String>>,
    using_gpu: Arc<Mutex<Option<bool>>>,
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

    let ts_info = tailscale::detect();

    let ts = match ts_info {
        Some(ts) => ts,
        None => {
            *state.tailscale_url.lock().unwrap() = None;
            *state.tailscale_cert_generated.lock().unwrap() = None;
            let _ = app_handle.emit("tailscale-changed", "");
            return Err("Tailscale is not installed or not connected. Install Tailscale on both your PC and phone, then try again.".to_string());
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
fn get_gpu_status(state: State<AppState>) -> serde_json::Value {
    let gpu = *state.using_gpu.lock().unwrap();
    let compiled = transcription::Transcriber::gpu_compiled();
    serde_json::json!({
        "compiled": compiled,
        "active": gpu,
    })
}

#[tauri::command]
fn start_recording(state: State<AppState>) -> Result<(), String> {
    let mut rec_guard = state.recording.lock().unwrap();
    if rec_guard.is_some() {
        return Err("Already recording".into());
    }

    let stop_flag = Arc::new(AtomicBool::new(false));
    let samples: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));

    let stop_clone = stop_flag.clone();
    let samples_clone = samples.clone();
    let device_name = state.selected_device.lock().unwrap().clone();

    let thread = std::thread::spawn(move || {
        match audio_capture::record_until_stopped(stop_clone, device_name) {
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
        thread: Some(thread),
    });

    Ok(())
}

#[tauri::command]
fn stop_and_transcribe(state: State<AppState>, app_handle: tauri::AppHandle) -> Result<String, String> {
    let session = {
        let mut guard = state.recording.lock().unwrap();
        guard.take().ok_or("Not recording")?
    };

    session.stop_flag.store(true, Ordering::Relaxed);

    if let Some(thread) = session.thread {
        thread.join().map_err(|_| "Recording thread panicked")?;
    }

    let samples = Arc::try_unwrap(session.samples)
        .map_err(|_| "Failed to get samples")?
        .into_inner()
        .unwrap();

    let duration = samples.len() as f64 / 16000.0;

    // Gentle handling for too-short recordings
    if duration < 0.3 {
        return Err("hint:Too short \u{2014} hold to record".into());
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

    Ok(result.text)
}

#[tauri::command]
fn cancel_recording(state: State<AppState>) -> Result<(), String> {
    let session = {
        let mut guard = state.recording.lock().unwrap();
        guard.take().ok_or("Not recording")?
    };
    session.stop_flag.store(true, Ordering::Relaxed);
    if let Some(thread) = session.thread {
        let _ = thread.join();
    }
    Ok(())
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

    // Parse the stored path from reg output ("    Dictator    REG_SZ    C:\path\to\exe")
    let stdout = String::from_utf8_lossy(&ok_output.stdout);
    let stored_path = stdout
        .lines()
        .find(|l| l.contains("Dictator"))
        .and_then(|l| l.split("REG_SZ").nth(1))
        .map(|p| p.trim().to_string());

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
    Ok(())
}

fn get_all_hotkey_presets(state: &AppState) -> (String, String, String) {
    let rec = state.hotkey.lock().unwrap().clone();
    let inject = state.inject_hotkey.lock().unwrap().clone();
    let tts = state.tts_hotkey.lock().unwrap().clone();
    (rec, inject, tts)
}

#[tauri::command]
fn set_hotkey(
    app_handle: tauri::AppHandle,
    state: State<AppState>,
    preset: String,
) -> Result<(), String> {
    let (_, inject, tts) = get_all_hotkey_presets(&state);
    register_all_hotkeys(&app_handle, &preset, &inject, &tts)?;
    *state.hotkey.lock().unwrap() = preset;
    Ok(())
}

#[tauri::command]
fn set_inject_hotkey(
    app_handle: tauri::AppHandle,
    state: State<AppState>,
    preset: String,
) -> Result<(), String> {
    let (rec, _, tts) = get_all_hotkey_presets(&state);
    register_all_hotkeys(&app_handle, &rec, &preset, &tts)?;
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
    let (rec, inject, _) = get_all_hotkey_presets(&state);
    register_all_hotkeys(&app_handle, &rec, &inject, &preset)?;
    *state.tts_hotkey.lock().unwrap() = preset;
    Ok(())
}

fn parse_hotkey_preset(
    preset: &str,
) -> Option<tauri_plugin_global_shortcut::Shortcut> {
    use tauri_plugin_global_shortcut::{Code, Modifiers, Shortcut};
    match preset {
        "Ctrl+Shift+Space" => Some(Shortcut::new(
            Some(Modifiers::CONTROL | Modifiers::SHIFT),
            Code::Space,
        )),
        "Ctrl+Shift+D" => Some(Shortcut::new(
            Some(Modifiers::CONTROL | Modifiers::SHIFT),
            Code::KeyD,
        )),
        "Ctrl+Alt+Space" => Some(Shortcut::new(
            Some(Modifiers::CONTROL | Modifiers::ALT),
            Code::Space,
        )),
        "F9" => Some(Shortcut::new(None, Code::F9)),
        "Ctrl+F9" => Some(Shortcut::new(Some(Modifiers::CONTROL), Code::F9)),
        "Ctrl+Shift+V" => Some(Shortcut::new(
            Some(Modifiers::CONTROL | Modifiers::SHIFT),
            Code::KeyV,
        )),
        "Ctrl+Shift+I" => Some(Shortcut::new(
            Some(Modifiers::CONTROL | Modifiers::SHIFT),
            Code::KeyI,
        )),
        "F10" => Some(Shortcut::new(None, Code::F10)),
        "Ctrl+F10" => Some(Shortcut::new(Some(Modifiers::CONTROL), Code::F10)),
        "Ctrl+Shift+R" => Some(Shortcut::new(
            Some(Modifiers::CONTROL | Modifiers::SHIFT),
            Code::KeyR,
        )),
        "Ctrl+Shift+T" => Some(Shortcut::new(
            Some(Modifiers::CONTROL | Modifiers::SHIFT),
            Code::KeyT,
        )),
        "F11" => Some(Shortcut::new(None, Code::F11)),
        _ => None,
    }
}

#[tauri::command]
fn set_autostart(enabled: bool) -> Result<(), String> {
    use std::os::windows::process::CommandExt;
    if enabled {
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
        let exe_str = exe.to_str().ok_or("Invalid exe path")?;
        let output = std::process::Command::new("reg")
            .args([
                "add",
                r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run",
                "/v",
                "Dictator",
                "/t",
                "REG_SZ",
                "/d",
                exe_str,
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

// ── Version Check ───────────────────────────────────────────────────────

const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

#[tauri::command]
fn get_app_version() -> String {
    APP_VERSION.to_string()
}

#[tauri::command]
async fn check_for_updates() -> serde_json::Value {
    let current = APP_VERSION;
    // Check GitHub releases API (replace with actual repo when published)
    let result = reqwest::get(
        "https://api.github.com/repos/yuanbaca/dictator/releases/latest",
    )
    .await;

    match result {
        Ok(resp) if resp.status().is_success() => {
            if let Ok(json) = resp.json::<serde_json::Value>().await {
                let json: serde_json::Value = json;
                let latest = json["tag_name"]
                    .as_str()
                    .unwrap_or("")
                    .trim_start_matches('v');
                let url = json["html_url"]
                    .as_str()
                    .unwrap_or("");
                return serde_json::json!({
                    "current": current,
                    "latest": latest,
                    "update_available": latest != current && !latest.is_empty(),
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

    std::thread::spawn(move || {
        eprintln!("Loading whisper model from: {}", model_path.display());
        match transcription::Transcriber::new(&model_path) {
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
    model_manager::AVAILABLE_MODELS
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
fn delete_llm_model(filename: String) -> Result<(), String> {
    model_manager::delete_model(&filename)
}

#[tauri::command]
fn get_models_dir() -> String {
    model_manager::models_dir()
        .canonicalize()
        .unwrap_or_else(|_| model_manager::models_dir())
        .to_string_lossy()
        .to_string()
}

// ── TTS commands ────────────────────────────────────────────────────────

#[tauri::command]
fn speak_text(state: State<AppState>, text: String) -> Result<(), String> {
    let engine = state.tts_engine.lock().unwrap().clone();
    if engine == "piper" {
        state.piper.speak(&text)
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
                return Ok(());
            } else {
                state.piper.pause();
                let _ = app_handle.emit("tts-state", "paused");
                return Ok(());
            }
        }
    } else {
        let guard = state.speaker.lock().unwrap();
        if let Some(speaker) = guard.as_ref() {
            if speaker.is_speaking() {
                // SAPI doesn't support pause — stop instead
                let _ = speaker.stop();
                let _ = app_handle.emit("tts-state", "idle");
                return Ok(());
            }
        }
    }

    let text = tts::capture_selected_text()?;

    // Emit the text to the frontend so it can display it
    let _ = app_handle.emit("tts-text", text.as_str());
    let _ = app_handle.emit("tts-state", "speaking");

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
    if engine != "sapi" && engine != "piper" {
        return Err(format!("Unknown engine: {engine}"));
    }
    *state.tts_engine.lock().unwrap() = engine;
    Ok(())
}

#[tauri::command]
fn has_piper(state: State<AppState>) -> bool {
    state.piper.has_piper_exe()
}

#[tauri::command]
fn list_piper_voices(state: State<AppState>) -> Vec<serde_json::Value> {
    tts::PIPER_VOICES
        .iter()
        .map(|v| {
            serde_json::json!({
                "id": v.id,
                "name": v.name,
                "size_mb": v.size_bytes / 1_000_000,
                "installed": state.piper.has_voice(v.id),
            })
        })
        .collect()
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

            match llm::Formatter::new(&model_path) {
                Ok(f) => {
                    *state.llm_status.lock().unwrap() = Some(true);
                    *guard = Some(f);
                }
                Err(e) => {
                    let msg = format!("{e}");
                    *state.llm_status.lock().unwrap() = Some(false);
                    *state.llm_error.lock().unwrap() = Some(msg.clone());
                    return Err(msg);
                }
            }
        }
    }

    let guard = state.formatter.lock().unwrap();
    let formatter = guard.as_ref().ok_or("LLM not loaded")?;

    // Check for a custom template override
    let customs = state.custom_templates.lock().unwrap();
    let custom_instruction = customs.get(format_type.key()).cloned();
    drop(customs);

    formatter
        .format_text_custom(&text, format_type, custom_instruction.as_deref())
        .map_err(|e| format!("{e}"))
}

// ── App entry point ─────────────────────────────────────────────────────

fn main() {
    let model_path = model_manager::find_whisper_model();
    let transcriber: Arc<Mutex<Option<transcription::Transcriber>>> = Arc::new(Mutex::new(None));
    let formatter: Arc<Mutex<Option<llm::Formatter>>> = Arc::new(Mutex::new(None));
    let injection_mode: Arc<Mutex<String>> = Arc::new(Mutex::new("paste".into()));
    let auto_space: Arc<Mutex<bool>> = Arc::new(Mutex::new(true)); // on by default
    let auto_format: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));
    let auto_format_type: Arc<Mutex<String>> = Arc::new(Mutex::new("clean_up".into()));
    let connected_phones: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
    let using_gpu: Arc<Mutex<Option<bool>>> = Arc::new(Mutex::new(None));
    let selected_device: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

    // Determine LAN IPs for fallback
    let ips = server::get_local_ips();

    // ── Networking strategy ──
    //
    // Two servers, user picks which URL to open on their phone:
    //   Port 3456 — LAN (self-signed cert, always runs, same WiFi only)
    //   Port 3457 — Tailscale (trusted cert, only if available, works from anywhere)
    //
    // This way Tailscale is never forced, and LAN always works as a fallback.

    let lan_url = format!("https://{}:{}", ips[0], LAN_PORT);

    // Tailscale is opt-in — the frontend enables it via refresh_tailscale when the user toggles it on.
    // At boot we only start the LAN server. Tailscale server starts dynamically on demand.
    eprintln!("LAN URL: {lan_url} (self-signed, same WiFi)");

    tauri::Builder::default()
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

                            if inject_key.is_some_and(|k| k == *shortcut) {
                                let _ = win.emit("inject-at-cursor", ());
                            } else if tts_key.is_some_and(|k| k == *shortcut) {
                                let _ = win.emit("tts-read-aloud", ());
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
            server_url: Mutex::new(Some(lan_url)),
            connected_phones: connected_phones.clone(),
            cert_type: Mutex::new("self-signed".to_string()),
            tailscale_url: Mutex::new(None),
            using_gpu: using_gpu.clone(),
            selected_device: selected_device.clone(),
            hotkey: Mutex::new("Ctrl+Shift+Space".into()),
            inject_hotkey: Mutex::new("none".into()),
            tts_hotkey: Mutex::new("none".into()),
            speaker: Arc::new(Mutex::new(None)),
            piper: Arc::new(tts::PiperSpeaker::new(
                model_manager::models_dir().join("piper"),
                model_manager::models_dir().join("piper-voices"),
            )),
            tts_engine: Mutex::new("sapi".into()),
            formatter: formatter.clone(),
            llm_status: Arc::new(Mutex::new(None)),
            llm_error: Mutex::new(None),
            no_model: Mutex::new(false),
            tailscale_server_running: Arc::new(AtomicBool::new(false)),
            server_shared: Mutex::new(None),
            tailscale_cert_generated: Mutex::new(None),
            custom_templates: Arc::new(Mutex::new(
                dictator::templates::load_custom_templates(),
            )),
        })
        .invoke_handler(tauri::generate_handler![
            get_status,
            get_server_url,
            get_connected_phones,
            get_cert_type,
            get_tailscale_url,
            refresh_tailscale,
            get_tailscale_status,
            get_auto_space,
            set_auto_space,
            get_auto_format,
            set_auto_format,
            get_auto_format_type,
            set_auto_format_type,
            get_gpu_status,
            list_audio_devices,
            get_selected_device,
            set_selected_device,
            start_recording,
            stop_and_transcribe,
            cancel_recording,
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
            set_piper_voice,
            download_piper,
            download_piper_voice,
            delete_piper_voice,
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
            get_app_version,
            check_for_updates,
        ])
        .setup(move |app| {
            let transcriber_handle = transcriber.clone();
            let gpu_handle = using_gpu.clone();
            let app_handle = app.handle().clone();

            // Load model in background (or signal no model available)
            if let Some(model_path) = model_path {
                std::thread::spawn(move || {
                    eprintln!("Loading whisper model from: {}", model_path.display());

                    match transcription::Transcriber::new(&model_path) {
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

            // Only the LAN server starts at boot. Tailscale is opt-in via Settings toggle.
            std::thread::spawn(move || {
                let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
                rt.block_on(async move {
                    let lan_state = Arc::new(server::ServerState {
                        transcriber: server_transcriber,
                        formatter: server_formatter,
                        injection_mode: server_injection,
                        connected_phones: server_phones,
                        auto_space: server_auto_space,
                        auto_format: server_auto_format,
                        auto_format_type: server_auto_format_type,
                        cert_type: "self-signed".to_string(),
                        using_gpu: server_gpu,
                        tts_trigger: Some(tts_tx),
                    });
                    server::run_server(
                        lan_state,
                        LAN_PORT,
                        server::TlsSource::SelfSigned { local_ips: ips },
                    )
                    .await;
                });
            });

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
            let fmt_clean = tauri::menu::MenuItem::with_id(app, "fmt_clean_up", &fmt_label("clean_up", "Clean Up"), true, None::<&str>)?;
            let fmt_email = tauri::menu::MenuItem::with_id(app, "fmt_email", &fmt_label("email", "Email"), true, None::<&str>)?;
            let fmt_notes = tauri::menu::MenuItem::with_id(app, "fmt_meeting_notes", &fmt_label("meeting_notes", "Meeting Notes"), true, None::<&str>)?;
            let fmt_docs = tauri::menu::MenuItem::with_id(app, "fmt_documentation", &fmt_label("documentation", "Documentation"), true, None::<&str>)?;
            let fmt_msg = tauri::menu::MenuItem::with_id(app, "fmt_message", &fmt_label("message", "Message"), true, None::<&str>)?;

            let sep1 = PredefinedMenuItem::separator(app)?;
            let cancel_rec_item = tauri::menu::MenuItem::with_id(
                app, "cancel_recording", "Cancel Recording", true, None::<&str>,
            )?;
            let sep2 = PredefinedMenuItem::separator(app)?;
            let show_item = tauri::menu::MenuItem::with_id(app, "show", "Show Dictator", true, None::<&str>)?;
            let quit_item = tauri::menu::MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;

            let tray_menu = tauri::menu::Menu::with_items(app, &[
                &autoformat_item,
                &fmt_clean, &fmt_email, &fmt_notes, &fmt_docs, &fmt_msg,
                &sep1,
                &cancel_rec_item,
                &sep2,
                &show_item,
                &quit_item,
            ])?;

            // Clones for the various closures that need menu item access
            let autoformat_item2 = autoformat_item.clone();
            let autoformat_item3 = autoformat_item.clone();
            let fmt_clean2 = fmt_clean.clone();
            let fmt_email2 = fmt_email.clone();
            let fmt_notes2 = fmt_notes.clone();
            let fmt_docs2 = fmt_docs.clone();
            let fmt_msg2 = fmt_msg.clone();

            let _tray = tauri::tray::TrayIconBuilder::new()
                .icon(app.default_window_icon().expect("Missing app icon").clone())
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
                            app.exit(0);
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
                            // Update checkmarks
                            let types = [
                                (&fmt_clean, "clean_up"),
                                (&fmt_email, "email"),
                                (&fmt_notes, "meeting_notes"),
                                (&fmt_docs, "documentation"),
                                (&fmt_msg, "message"),
                            ];
                            for (item, name) in &types {
                                let prefix = if *name == fmt_type { "  ✓ " } else { "  " };
                                let display = match *name {
                                    "clean_up" => "Clean Up",
                                    "email" => "Email",
                                    "meeting_notes" => "Meeting Notes",
                                    "documentation" => "Documentation",
                                    "message" => "Message",
                                    _ => name,
                                };
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
                    }
                })
                .build(app)?;

            // Sync tray checkmarks when Settings changes the format type
            {
                use tauri::Listener;
                app.listen("sync-tray-format", move |event| {
                    let fmt_type = event.payload().trim_matches('"');
                    let items: [(&tauri::menu::MenuItem<tauri::Wry>, &str, &str); 5] = [
                        (&fmt_clean2, "clean_up", "Clean Up"),
                        (&fmt_email2, "email", "Email"),
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
                });
            }

            // Hide to tray when the X button is clicked (instead of quitting)
            let window = app.get_webview_window("main").unwrap();
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

