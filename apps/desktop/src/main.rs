#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use deskmic::audio_capture;
use deskmic::injection::{self, InjectionMode};
use deskmic::server;
use deskmic::tailscale;
use deskmic::transcription;
use std::path::PathBuf;
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
}

const LAN_PORT: u16 = 3456;
const TAILSCALE_PORT: u16 = 3457;

// ── Tauri commands ──────────────────────────────────────────────────────

#[tauri::command]
fn get_status(state: State<AppState>) -> String {
    if let Some(err) = state.model_error.lock().unwrap().as_ref() {
        return format!("error:{err}");
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

#[tauri::command]
fn get_auto_space(state: State<AppState>) -> bool {
    *state.auto_space.lock().unwrap()
}

#[tauri::command]
fn set_auto_space(state: State<AppState>, enabled: bool) {
    *state.auto_space.lock().unwrap() = enabled;
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
fn get_autostart() -> bool {
    use std::os::windows::process::CommandExt;
    let output = std::process::Command::new("reg")
        .args([
            "query",
            r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run",
            "/v",
            "DeskMic",
        ])
        .creation_flags(0x08000000) // CREATE_NO_WINDOW
        .output();
    matches!(output, Ok(o) if o.status.success())
}

#[tauri::command]
fn get_hotkey(state: State<AppState>) -> String {
    state.hotkey.lock().unwrap().clone()
}

#[tauri::command]
fn set_hotkey(
    app_handle: tauri::AppHandle,
    state: State<AppState>,
    preset: String,
) -> Result<(), String> {
    use tauri_plugin_global_shortcut::GlobalShortcutExt;

    // Unregister all existing hotkeys
    let _ = app_handle.global_shortcut().unregister_all();

    // Register new (unless "none")
    if let Some(sc) = parse_hotkey_preset(&preset) {
        app_handle
            .global_shortcut()
            .register(sc)
            .map_err(|e| format!("Failed to register hotkey: {e}"))?;
    }

    *state.hotkey.lock().unwrap() = preset;
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
                "DeskMic",
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
                "DeskMic",
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

// ── App entry point ─────────────────────────────────────────────────────

fn main() {
    let model_path = find_model_path();
    let transcriber: Arc<Mutex<Option<transcription::Transcriber>>> = Arc::new(Mutex::new(None));
    let injection_mode: Arc<Mutex<String>> = Arc::new(Mutex::new("paste".into()));
    let auto_space: Arc<Mutex<bool>> = Arc::new(Mutex::new(true)); // on by default
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

    let tailscale_info = tailscale::detect();
    let tailscale_cert: Option<(Vec<u8>, Vec<u8>, String)> =
        tailscale_info.as_ref().and_then(|ts| {
            let cert_dir = std::env::temp_dir().join("deskmic-certs");
            let _ = std::fs::create_dir_all(&cert_dir);
            let cert_path = cert_dir.join("cert.pem");
            let key_path = cert_dir.join("key.pem");

            match tailscale::generate_cert(&ts.cert_domain, &cert_path, &key_path) {
                Ok((cert_pem, key_pem)) => {
                    eprintln!("Tailscale cert OK for {}", ts.cert_domain);
                    Some((cert_pem, key_pem, ts.cert_domain.clone()))
                }
                Err(e) => {
                    eprintln!("Tailscale cert failed: {e}");
                    eprintln!(
                        "Hint: try running DeskMic as Administrator, or run in an admin terminal:"
                    );
                    eprintln!(
                        "  tailscale cert --cert-file=\"{}\" --key-file=\"{}\" {}",
                        cert_path.display(),
                        key_path.display(),
                        ts.cert_domain
                    );
                    None
                }
            }
        });

    let tailscale_url = tailscale_cert
        .as_ref()
        .map(|(_, _, domain)| format!("https://{}:{}", domain, TAILSCALE_PORT));

    let cert_type = if tailscale_cert.is_some() {
        "tailscale".to_string()
    } else {
        "self-signed".to_string()
    };

    eprintln!("LAN URL: {lan_url} (self-signed, same WiFi)");
    if let Some(ref ts_url) = tailscale_url {
        eprintln!("Tailscale URL: {ts_url} (trusted cert, works anywhere)");
    }

    tauri::Builder::default()
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, _shortcut, event| {
                    if event.state == tauri_plugin_global_shortcut::ShortcutState::Pressed {
                        if let Some(win) = app.get_webview_window("main") {
                            let _ = win.emit("toggle-recording", ());
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
            model_error: Mutex::new(None),
            server_url: Mutex::new(Some(lan_url)),
            connected_phones: connected_phones.clone(),
            cert_type: Mutex::new(cert_type),
            tailscale_url: Mutex::new(tailscale_url),
            using_gpu: using_gpu.clone(),
            selected_device: selected_device.clone(),
            hotkey: Mutex::new("Ctrl+Shift+Space".into()),
        })
        .invoke_handler(tauri::generate_handler![
            get_status,
            get_server_url,
            get_connected_phones,
            get_cert_type,
            get_tailscale_url,
            get_auto_space,
            set_auto_space,
            get_gpu_status,
            list_audio_devices,
            get_selected_device,
            set_selected_device,
            start_recording,
            stop_and_transcribe,
            cancel_recording,
            inject_text,
            set_injection_mode,
            get_autostart,
            set_autostart,
            get_hotkey,
            set_hotkey,
        ])
        .setup(move |app| {
            let transcriber_handle = transcriber.clone();
            let gpu_handle = using_gpu.clone();
            let app_handle = app.handle().clone();

            // Load model in background
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

            // Start the phone companion HTTPS servers
            // LAN on port 3456 (always, self-signed) + Tailscale on 3457 (if available)
            let server_transcriber = transcriber.clone();
            let server_injection = injection_mode.clone();
            let server_phones = connected_phones.clone();
            let server_auto_space = auto_space.clone();
            let server_gpu = using_gpu.clone();

            std::thread::spawn(move || {
                let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
                rt.block_on(async move {
                    // LAN server — always runs, self-signed cert, same WiFi only
                    let lan_state = Arc::new(server::ServerState {
                        transcriber: server_transcriber.clone(),
                        injection_mode: server_injection.clone(),
                        connected_phones: server_phones.clone(),
                        auto_space: server_auto_space.clone(),
                        cert_type: "self-signed".to_string(),
                        using_gpu: server_gpu.clone(),
                    });
                    let lan = tokio::spawn(async move {
                        server::run_server(
                            lan_state,
                            LAN_PORT,
                            server::TlsSource::SelfSigned { local_ips: ips },
                        )
                        .await;
                    });

                    // Tailscale server — trusted cert, works from anywhere
                    if let Some((cert_pem, key_pem, _domain)) = tailscale_cert {
                        let ts_state = Arc::new(server::ServerState {
                            transcriber: server_transcriber,
                            injection_mode: server_injection,
                            connected_phones: server_phones,
                            auto_space: server_auto_space,
                            cert_type: "tailscale".to_string(),
                            using_gpu: server_gpu,
                        });
                        let ts = tokio::spawn(async move {
                            server::run_server(
                                ts_state,
                                TAILSCALE_PORT,
                                server::TlsSource::Tailscale { cert_pem, key_pem },
                            )
                            .await;
                        });
                        let _ = tokio::join!(lan, ts);
                    } else {
                        let _ = lan.await;
                    }
                });
            });

            // ── System tray — hides to tray on close, quit from menu ───
            let show_item =
                tauri::menu::MenuItem::with_id(app, "show", "Show DeskMic", true, None::<&str>)?;
            let quit_item =
                tauri::menu::MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let tray_menu =
                tauri::menu::Menu::with_items(app, &[&show_item, &quit_item])?;

            let _tray = tauri::tray::TrayIconBuilder::new()
                .icon(app.default_window_icon().expect("Missing app icon").clone())
                .tooltip("DeskMic Dictation")
                .menu(&tray_menu)
                .on_menu_event(|app, event| {
                    match event.id().as_ref() {
                        "show" => {
                            if let Some(win) = app.get_webview_window("main") {
                                let _ = win.show();
                                let _ = win.set_focus();
                            }
                        }
                        "quit" => {
                            app.exit(0);
                        }
                        _ => {}
                    }
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        let app = tray.app_handle();
                        if let Some(win) = app.get_webview_window("main") {
                            let _ = win.show();
                            let _ = win.set_focus();
                        }
                    }
                })
                .build(app)?;

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
        .expect("error while running DeskMic");
}

fn find_model_path() -> PathBuf {
    const MODEL: &str = "models/ggml-base.en.bin";

    // Search relative to the working directory
    let cwd_candidates = [
        PathBuf::from(format!("../../{MODEL}")),
        PathBuf::from(MODEL),
        PathBuf::from(format!("../{MODEL}")),
    ];

    for p in &cwd_candidates {
        if p.exists() {
            eprintln!("Found model at: {}", p.display());
            return p.clone();
        }
    }

    // Search relative to the executable (works when double-clicking the exe)
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let exe_candidates = [
                exe_dir.join(format!("../../../../{MODEL}")), // from target/release/
                exe_dir.join(format!("../../{MODEL}")),       // from apps/desktop/
                exe_dir.join(format!("../{MODEL}")),
                exe_dir.join(MODEL),                          // model next to exe
            ];
            for p in &exe_candidates {
                if p.exists() {
                    eprintln!("Found model at: {}", p.display());
                    return p.clone();
                }
            }
        }
    }

    eprintln!("WARNING: Could not find whisper model file!");
    eprintln!("Place ggml-base.en.bin in the models/ folder at the project root.");
    PathBuf::from(MODEL)
}
