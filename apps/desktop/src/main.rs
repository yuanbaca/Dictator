use deskmic::audio_capture;
use deskmic::injection::{self, InjectionMode};
use deskmic::server;
use deskmic::transcription;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
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
    model_error: Mutex<Option<String>>,
    server_url: Mutex<Option<String>>,
    connected_phones: Arc<Mutex<u32>>,
}

const SERVER_PORT: u16 = 3456;

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
fn start_recording(state: State<AppState>) -> Result<(), String> {
    let mut rec_guard = state.recording.lock().unwrap();
    if rec_guard.is_some() {
        return Err("Already recording".into());
    }

    let stop_flag = Arc::new(AtomicBool::new(false));
    let samples: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));

    let stop_clone = stop_flag.clone();
    let samples_clone = samples.clone();

    let thread = std::thread::spawn(move || {
        match audio_capture::record_until_stopped(stop_clone) {
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
fn stop_and_transcribe(state: State<AppState>) -> Result<String, String> {
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
    if duration < 0.3 {
        return Err(format!("Recording too short ({:.1}s)", duration));
    }

    let transcriber_guard = state.transcriber.lock().unwrap();
    let transcriber = transcriber_guard
        .as_ref()
        .ok_or("Model not loaded")?;

    let result = transcriber
        .transcribe(&samples)
        .map_err(|e| format!("Transcription failed: {e}"))?;

    if result.text.is_empty() {
        return Err("No speech detected".into());
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

    std::thread::sleep(std::time::Duration::from_millis(200));

    injection::inject_text(&text, mode).map_err(|e| format!("Injection failed: {e}"))
}

#[tauri::command]
fn set_injection_mode(state: State<AppState>, mode: String) {
    *state.injection_mode.lock().unwrap() = mode;
}

fn main() {
    let model_path = find_model_path();
    let transcriber: Arc<Mutex<Option<transcription::Transcriber>>> = Arc::new(Mutex::new(None));
    let injection_mode: Arc<Mutex<String>> = Arc::new(Mutex::new("paste".into()));
    let connected_phones: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));

    // Determine server URL
    let ips = server::get_local_ips();
    let server_url = format!("https://{}:{}", ips[0], SERVER_PORT);
    eprintln!("Phone companion will be at: {server_url}");

    tauri::Builder::default()
        .manage(AppState {
            transcriber: transcriber.clone(),
            recording: Mutex::new(None),
            injection_mode: injection_mode.clone(),
            model_error: Mutex::new(None),
            server_url: Mutex::new(Some(server_url)),
            connected_phones: connected_phones.clone(),
        })
        .invoke_handler(tauri::generate_handler![
            get_status,
            get_server_url,
            get_connected_phones,
            start_recording,
            stop_and_transcribe,
            cancel_recording,
            inject_text,
            set_injection_mode,
        ])
        .setup(move |app| {
            let transcriber_handle = transcriber.clone();
            let app_handle = app.handle().clone();

            // Load model in background
            std::thread::spawn(move || {
                eprintln!("Loading whisper model from: {}", model_path.display());

                match transcription::Transcriber::new(&model_path) {
                    Ok(t) => {
                        *transcriber_handle.lock().unwrap() = Some(t);
                        eprintln!("Model loaded successfully");
                        let _ = app_handle.emit("model-ready", "ok");
                    }
                    Err(e) => {
                        let msg = format!("Failed to load model: {e}");
                        eprintln!("{msg}");
                        let _ = app_handle.emit("model-ready", &msg);
                        let state: State<AppState> = app_handle.state();
                        *state.model_error.lock().unwrap() = Some(msg);
                    }
                }
            });

            // Start the phone companion web server
            let server_transcriber = transcriber.clone();
            let server_injection = injection_mode.clone();
            let server_phones = connected_phones.clone();

            std::thread::spawn(move || {
                let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
                rt.block_on(async {
                    let state = Arc::new(server::ServerState {
                        transcriber: server_transcriber,
                        injection_mode: server_injection,
                        connected_phones: server_phones,
                    });
                    server::run_server(state, SERVER_PORT, ips).await;
                });
            });

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running DeskMic");
}

fn find_model_path() -> PathBuf {
    let candidates = [
        "../../models/ggml-base.en.bin",
        "models/ggml-base.en.bin",
        "../models/ggml-base.en.bin",
    ];

    for path in &candidates {
        let p = PathBuf::from(path);
        if p.exists() {
            eprintln!("Found model at: {}", p.display());
            return p;
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let model = exe_dir.join("../../models/ggml-base.en.bin");
            if model.exists() {
                eprintln!("Found model at: {}", model.display());
                return model;
            }
        }
    }

    eprintln!("WARNING: Could not find whisper model file!");
    PathBuf::from("../../models/ggml-base.en.bin")
}
