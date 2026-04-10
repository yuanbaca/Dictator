//! Embedded web server that serves the phone companion and handles WebSocket audio.

use crate::injection::{self, InjectionMode};
use crate::llm;
use crate::templates::FormatType;
use crate::transcription;
use axum::extract::ws::{Message, WebSocket};
use axum::extract::WebSocketUpgrade;
use axum::response::{Html, Json};
use axum::routing::get;
use axum::Router;
use axum_server::tls_rustls::RustlsConfig;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex};
use tower_http::cors::CorsLayer;

/// TLS configuration source.
pub enum TlsSource {
    /// Trusted Let's Encrypt cert from Tailscale (no browser warnings).
    Tailscale { cert_pem: Vec<u8>, key_pem: Vec<u8> },
    /// Self-signed cert for LAN IPs (browser warning once, but iOS allows bypass for IPs).
    SelfSigned { local_ips: Vec<String> },
}

/// Shared state for the web server.
pub struct ServerState {
    pub transcriber: Arc<Mutex<Option<transcription::Transcriber>>>,
    pub formatter: Arc<Mutex<Option<llm::Formatter>>>,
    pub injection_mode: Arc<Mutex<String>>,
    pub connected_phones: Arc<Mutex<u32>>,
    pub auto_space: Arc<Mutex<bool>>,
    pub auto_format: Arc<Mutex<bool>>,
    pub auto_format_type: Arc<Mutex<String>>,
    pub cert_type: String,
    pub using_gpu: Arc<Mutex<Option<bool>>>,
    /// Channel for phone to trigger TTS read-aloud on the desktop.
    pub tts_trigger: Option<std::sync::mpsc::Sender<()>>,
}

/// The mobile companion HTML (embedded at compile time).
const COMPANION_HTML: &str = include_str!("../companion/index.html");

/// The HTTPS setup guide HTML (embedded at compile time).
const SETUP_HTML: &str = include_str!("../companion/setup.html");

/// Start the HTTPS server on the given port. Runs until the app exits.
pub async fn run_server(state: Arc<ServerState>, port: u16, tls: TlsSource) {
    let state_for_ws = state.clone();
    let state_for_api = state.clone();

    let app = Router::new()
        .route("/", get(|| async { Html(COMPANION_HTML) }))
        .route("/setup", get(|| async { Html(SETUP_HTML) }))
        .route(
            "/api/info",
            get(move || {
                let st = state_for_api.clone();
                async move {
                    let gpu = *st.using_gpu.lock().unwrap();
                    Json(serde_json::json!({
                        "cert_type": st.cert_type,
                        "gpu": gpu,
                        "gpu_compiled": transcription::Transcriber::gpu_compiled(),
                        "connected_phones": *st.connected_phones.lock().unwrap(),
                    }))
                }
            }),
        )
        .route(
            "/ws",
            get(move |ws: WebSocketUpgrade| {
                let st = state_for_ws.clone();
                async move { ws.on_upgrade(move |socket| handle_ws(socket, st)) }
            }),
        )
        .layer(CorsLayer::permissive());

    // Build TLS config
    let tls_config = match tls {
        TlsSource::Tailscale { cert_pem, key_pem } => {
            eprintln!("Using Tailscale HTTPS certificate (trusted by browsers)");
            RustlsConfig::from_pem(cert_pem, key_pem)
                .await
                .expect("Failed to load Tailscale TLS config")
        }
        TlsSource::SelfSigned { local_ips } => {
            let mut sans = local_ips;
            sans.push("localhost".to_string());
            eprintln!("Generating self-signed cert for SANs: {sans:?}");
            let ck = rcgen::generate_simple_self_signed(sans)
                .expect("Failed to generate self-signed certificate");
            RustlsConfig::from_pem(
                ck.cert.pem().into_bytes(),
                ck.key_pair.serialize_pem().into_bytes(),
            )
            .await
            .expect("Failed to create self-signed TLS config")
        }
    };

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    eprintln!("Phone companion server listening on https://0.0.0.0:{port}");

    axum_server::bind_rustls(addr, tls_config)
        .serve(app.into_make_service())
        .await
        .expect("Server error");
}

/// Handle a WebSocket connection from a phone.
async fn handle_ws(mut socket: WebSocket, state: Arc<ServerState>) {
    eprintln!("Phone connected via WebSocket");

    {
        let mut count = state.connected_phones.lock().unwrap();
        *count += 1;
    }

    // Send a welcome message
    let _ = socket
        .send(Message::Text(
            serde_json::json!({"type": "status", "state": "connected"})
                .to_string()
                .into(),
        ))
        .await;

    // Check if model is ready
    let model_ready = state.transcriber.lock().unwrap().is_some();
    let _ = socket
        .send(Message::Text(
            serde_json::json!({
                "type": "status",
                "state": if model_ready { "ready" } else { "loading" }
            })
            .to_string()
            .into(),
        ))
        .await;

    // Send current settings so phone UI can sync
    let auto_space_val = *state.auto_space.lock().unwrap();
    let _ = socket
        .send(Message::Text(
            serde_json::json!({"type": "setting", "key": "auto_space", "value": auto_space_val})
                .to_string()
                .into(),
        ))
        .await;

    let auto_format_val = *state.auto_format.lock().unwrap();
    let _ = socket
        .send(Message::Text(
            serde_json::json!({"type": "setting", "key": "auto_format", "value": auto_format_val})
                .to_string()
                .into(),
        ))
        .await;

    let auto_format_type_val = state.auto_format_type.lock().unwrap().clone();
    let _ = socket
        .send(Message::Text(
            serde_json::json!({"type": "setting", "key": "auto_format_type", "value": auto_format_type_val})
                .to_string()
                .into(),
        ))
        .await;

    while let Some(msg) = socket.recv().await {
        let msg = match msg {
            Ok(m) => m,
            Err(e) => {
                eprintln!("WebSocket error: {e}");
                break;
            }
        };

        match msg {
            Message::Binary(audio_data) => {
                let duration_secs = audio_data.len() as f64 / (16000.0 * 4.0);
                eprintln!(
                    "Received audio: {} bytes ({:.1}s at 16kHz)",
                    audio_data.len(),
                    duration_secs
                );

                // Audio is raw f32 PCM at 16kHz mono
                let samples = bytes_to_f32_samples(&audio_data);

                // --- Gentle handling for too-short recordings ---
                if samples.len() < (16000.0 * 0.3) as usize {
                    let _ = socket
                        .send(Message::Text(
                            serde_json::json!({
                                "type": "hint",
                                "message": "Too short \u{2014} hold to record"
                            })
                            .to_string()
                            .into(),
                        ))
                        .await;
                    // Return to ready state
                    let _ = socket
                        .send(Message::Text(
                            serde_json::json!({"type": "status", "state": "ready"})
                                .to_string()
                                .into(),
                        ))
                        .await;
                    continue;
                }

                // Notify phone we're transcribing
                let _ = socket
                    .send(Message::Text(
                        serde_json::json!({"type": "status", "state": "transcribing"})
                            .to_string()
                            .into(),
                    ))
                    .await;

                // Transcribe with progress reporting
                let progress = Arc::new(AtomicI32::new(0));
                let transcriber = state.transcriber.clone();
                let progress_cb = progress.clone();

                let mut transcribe_handle = tokio::task::spawn_blocking(move || {
                    let guard = transcriber.lock().unwrap();
                    match guard.as_ref() {
                        Some(t) => match t.transcribe_with_progress(&samples, move |pct| {
                            progress_cb.store(pct, Ordering::Relaxed);
                        }) {
                            Ok(result) if !result.text.is_empty() => Ok(result.text),
                            Ok(_) => Err("no_speech".to_string()),
                            Err(e) => Err(format!("Transcription failed: {e}")),
                        },
                        None => Err("model_loading".to_string()),
                    }
                });

                // Poll progress and forward to phone while transcription runs
                let mut last_pct = -1i32;
                let text = loop {
                    tokio::select! {
                        result = &mut transcribe_handle => {
                            // Send 100% before breaking
                            let _ = socket
                                .send(Message::Text(
                                    serde_json::json!({"type": "progress", "percent": 100})
                                        .to_string().into(),
                                ))
                                .await;
                            break result.unwrap_or(Err("Transcription thread panicked".to_string()));
                        }
                        _ = tokio::time::sleep(std::time::Duration::from_millis(200)) => {
                            let pct = progress.load(Ordering::Relaxed);
                            if pct != last_pct {
                                last_pct = pct;
                                let _ = socket
                                    .send(Message::Text(
                                        serde_json::json!({"type": "progress", "percent": pct})
                                            .to_string().into(),
                                    ))
                                    .await;
                            }
                        }
                    }
                };

                match text {
                    Ok(mut text) => {
                        eprintln!("Transcribed: {text}");

                        // Auto-format with LLM if enabled
                        let should_format = *state.auto_format.lock().unwrap();
                        if should_format {
                            let fmt_type_str = state.auto_format_type.lock().unwrap().clone();
                            let fmt_type: Option<FormatType> =
                                serde_json::from_value(serde_json::Value::String(fmt_type_str)).ok();

                            if let Some(ft) = fmt_type {
                                if ft != FormatType::None {
                                    // Tell phone we're formatting
                                    let _ = socket
                                        .send(Message::Text(
                                            serde_json::json!({"type": "status", "state": "formatting"})
                                                .to_string()
                                                .into(),
                                        ))
                                        .await;

                                    // Lazy-load LLM if needed, then format
                                    let formatter = state.formatter.clone();
                                    let raw = text.clone();
                                    let formatted = tokio::task::spawn_blocking(move || {
                                        let mut guard = formatter.lock().unwrap();
                                        if guard.is_none() {
                                            if let Some(path) = llm::find_llm_model_path() {
                                                match llm::Formatter::new(&path) {
                                                    Ok(f) => { *guard = Some(f); }
                                                    Err(e) => {
                                                        eprintln!("LLM load failed: {e}");
                                                        return Err(format!("{e}"));
                                                    }
                                                }
                                            } else {
                                                return Err("No LLM model found".into());
                                            }
                                        }
                                        guard.as_ref().unwrap().format_text(&raw, ft).map_err(|e| format!("{e}"))
                                    }).await;

                                    match formatted {
                                        Ok(Ok(formatted_text)) => {
                                            eprintln!("Auto-formatted: {} -> {} chars", text.len(), formatted_text.len());
                                            text = formatted_text;
                                        }
                                        Ok(Err(e)) => {
                                            eprintln!("Auto-format failed, using raw text: {e}");
                                        }
                                        Err(e) => {
                                            eprintln!("Auto-format task failed, using raw text: {e}");
                                        }
                                    }
                                }
                            }
                        }

                        // Auto-append space if enabled
                        if *state.auto_space.lock().unwrap() {
                            text.push(' ');
                        }

                        // Inject into focused window
                        let mode_str = state.injection_mode.lock().unwrap().clone();
                        let mode = match mode_str.as_str() {
                            "type" => InjectionMode::Type,
                            _ => InjectionMode::Paste,
                        };

                        let inject_result = injection::inject_text(&text, mode);

                        let _ = socket
                            .send(Message::Text(
                                serde_json::json!({
                                    "type": "result",
                                    "text": text.trim(),
                                    "injected": inject_result.is_ok(),
                                    "error": inject_result.err().map(|e| e.to_string()),
                                })
                                .to_string()
                                .into(),
                            ))
                            .await;
                    }
                    Err(ref err) if err == "no_speech" => {
                        // Gentle hint, not an error
                        let _ = socket
                            .send(Message::Text(
                                serde_json::json!({
                                    "type": "hint",
                                    "message": "No speech detected"
                                })
                                .to_string()
                                .into(),
                            ))
                            .await;
                    }
                    Err(ref err) if err == "model_loading" => {
                        let _ = socket
                            .send(Message::Text(
                                serde_json::json!({"type": "status", "state": "loading"})
                                    .to_string()
                                    .into(),
                            ))
                            .await;
                    }
                    Err(err) => {
                        // Actual error
                        let _ = socket
                            .send(Message::Text(
                                serde_json::json!({
                                    "type": "error",
                                    "message": err,
                                })
                                .to_string()
                                .into(),
                            ))
                            .await;
                    }
                }

                // Back to ready
                let _ = socket
                    .send(Message::Text(
                        serde_json::json!({"type": "status", "state": "ready"})
                            .to_string()
                            .into(),
                    ))
                    .await;
            }
            Message::Text(text) => {
                if let Ok(msg) = serde_json::from_str::<serde_json::Value>(&text) {
                    match msg.get("type").and_then(|v| v.as_str()) {
                        Some("ping") => {
                            let _ = socket
                                .send(Message::Text(
                                    serde_json::json!({"type": "pong"}).to_string().into(),
                                ))
                                .await;
                        }
                        Some("set_auto_space") => {
                            if let Some(enabled) = msg.get("enabled").and_then(|v| v.as_bool()) {
                                *state.auto_space.lock().unwrap() = enabled;
                                eprintln!("Auto-space set to {enabled} (from phone)");
                            }
                        }
                        Some("check_llm_model") => {
                            let has = crate::model_manager::has_llm_model();
                            let _ = socket
                                .send(Message::Text(
                                    serde_json::json!({"type": "setting", "key": "has_llm_model", "value": has})
                                        .to_string()
                                        .into(),
                                ))
                                .await;
                        }
                        Some("download_model") => {
                            let _ = socket
                                .send(Message::Text(
                                    serde_json::json!({"type": "download_status", "state": "downloading"})
                                        .to_string()
                                        .into(),
                                ))
                                .await;

                            // We need a reference to socket inside the async download,
                            // but we can't share it. Instead, poll from the download
                            // callback via an atomic and forward progress in a loop.
                            let progress_pct = Arc::new(AtomicI32::new(0));
                            let progress_clone = progress_pct.clone();

                            let mut dl_handle = tokio::spawn(async move {
                                crate::model_manager::download_model("phi3-mini", move |dl, total| {
                                    let pct = if total > 0 { (dl as f64 / total as f64 * 100.0) as i32 } else { 0 };
                                    progress_clone.store(pct, Ordering::Relaxed);
                                })
                                .await
                            });

                            let mut last_pct = -1i32;
                            let result = loop {
                                tokio::select! {
                                    res = &mut dl_handle => { break res; }
                                    _ = tokio::time::sleep(std::time::Duration::from_millis(500)) => {
                                        let pct = progress_pct.load(Ordering::Relaxed);
                                        if pct != last_pct {
                                            last_pct = pct;
                                            let _ = socket.send(Message::Text(
                                                serde_json::json!({"type": "download_progress", "percent": pct}).to_string().into(),
                                            )).await;
                                        }
                                    }
                                }
                            };

                            match result {
                                Ok(Ok(_)) => {
                                    let _ = socket.send(Message::Text(
                                        serde_json::json!({"type": "download_status", "state": "complete"}).to_string().into(),
                                    )).await;
                                }
                                _ => {
                                    let _ = socket.send(Message::Text(
                                        serde_json::json!({"type": "download_status", "state": "failed"}).to_string().into(),
                                    )).await;
                                }
                            }
                        }
                        Some("set_auto_format") => {
                            if let Some(enabled) = msg.get("enabled").and_then(|v| v.as_bool()) {
                                *state.auto_format.lock().unwrap() = enabled;
                                eprintln!("Auto-format set to {enabled} (from phone)");
                            }
                        }
                        Some("set_auto_format_type") => {
                            if let Some(val) = msg.get("value").and_then(|v| v.as_str()) {
                                *state.auto_format_type.lock().unwrap() = val.to_string();
                                eprintln!("Auto-format type set to {val} (from phone)");
                            }
                        }
                        Some("key") => {
                            if let Some(key) = msg.get("key").and_then(|v| v.as_str()) {
                                if let Err(e) = injection::send_key_by_name(key) {
                                    eprintln!("Key injection failed: {e}");
                                }
                            }
                        }
                        Some("read_aloud") => {
                            if let Some(tx) = &state.tts_trigger {
                                let _ = tx.send(());
                                eprintln!("TTS read-aloud triggered from phone");
                            }
                        }
                        _ => {}
                    }
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }

    eprintln!("Phone disconnected");
    let mut count = state.connected_phones.lock().unwrap();
    *count = count.saturating_sub(1);
}

/// Convert raw bytes (little-endian f32) to Vec<f32>.
fn bytes_to_f32_samples(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Get local IP addresses, with the best LAN IP first.
///
/// Parses `ipconfig` output on Windows to enumerate all interfaces, then
/// filters out virtual adapters (VMware, Hyper-V, VirtualBox, Tailscale)
/// and sorts so that real LAN IPs (192.168.x.x, 10.x.x.x, 172.16–31.x.x)
/// appear first.  All IPs are still returned so the self-signed cert covers
/// them all — the first element is just the one shown to the user.
pub fn get_local_ips() -> Vec<String> {
    if let Some(ips) = parse_ipconfig() {
        if !ips.is_empty() {
            return ips;
        }
    }

    // Fallback: UDP trick (may return Tailscale IP, better than nothing)
    if let Ok(socket) = std::net::UdpSocket::bind("0.0.0.0:0") {
        if socket.connect("8.8.8.8:80").is_ok() {
            if let Ok(addr) = socket.local_addr() {
                return vec![addr.ip().to_string()];
            }
        }
    }

    vec!["127.0.0.1".to_string()]
}

/// Parse `ipconfig` to get all IPv4 addresses grouped by adapter.
fn parse_ipconfig() -> Option<Vec<String>> {
    let output = std::process::Command::new("ipconfig")
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&output.stdout);

    let mut lan_ips: Vec<String> = Vec::new();
    let mut other_ips: Vec<String> = Vec::new();
    let mut current_adapter = String::new();

    for line in text.lines() {
        let trimmed = line.trim();

        // Adapter header lines look like:
        //   "Ethernet adapter Ethernet:"
        //   "Wireless LAN adapter Wi-Fi:"
        //   "Ethernet adapter vEthernet (WSL):"
        if !trimmed.is_empty() && line.ends_with(':') && !trimmed.contains('.') {
            current_adapter = trimmed.to_lowercase();
            continue;
        }

        // IPv4 Address line: "IPv4 Address. . . . . . . . . . . : 192.168.1.50"
        if trimmed.starts_with("IPv4 Address") || trimmed.starts_with("Autoconfiguration IPv4") {
            if let Some(ip_str) = trimmed.rsplit(':').next() {
                let ip = ip_str.trim().to_string();

                // Skip loopback, link-local, APIPA
                if ip.starts_with("127.") || ip.starts_with("169.254.") {
                    continue;
                }

                // Check if this is a virtual/tunnel adapter
                let is_virtual = current_adapter.contains("vmware")
                    || current_adapter.contains("virtualbox")
                    || current_adapter.contains("hyper-v")
                    || current_adapter.contains("vethernet")
                    || current_adapter.contains("loopback")
                    || current_adapter.contains("docker")
                    || current_adapter.contains("wsl");

                // Check if this is a Tailscale IP (CGNAT range 100.64.0.0/10)
                let is_tailscale = is_tailscale_ip(&ip);

                if is_virtual || is_tailscale {
                    other_ips.push(ip);
                } else {
                    lan_ips.push(ip);
                }
            }
        }
    }

    // LAN IPs first, then virtual/Tailscale (for cert SANs)
    lan_ips.append(&mut other_ips);
    Some(lan_ips)
}

/// Check if an IP is in the Tailscale CGNAT range (100.64.0.0/10).
fn is_tailscale_ip(ip: &str) -> bool {
    if let Some(rest) = ip.strip_prefix("100.") {
        if let Some(second_octet) = rest.split('.').next() {
            if let Ok(n) = second_octet.parse::<u8>() {
                return (64..=127).contains(&n);
            }
        }
    }
    false
}
