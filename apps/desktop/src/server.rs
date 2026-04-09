//! Embedded web server that serves the phone companion and handles WebSocket audio.

use crate::injection::{self, InjectionMode};
use crate::transcription;
use axum::extract::ws::{Message, WebSocket};
use axum::extract::WebSocketUpgrade;
use axum::response::Html;
use axum::routing::get;
use axum::Router;
use axum_server::tls_rustls::RustlsConfig;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use tower_http::cors::CorsLayer;

/// Shared state for the web server.
pub struct ServerState {
    pub transcriber: Arc<Mutex<Option<transcription::Transcriber>>>,
    pub injection_mode: Arc<Mutex<String>>,
    pub connected_phones: Arc<Mutex<u32>>,
}

/// The mobile companion HTML (embedded at compile time).
const COMPANION_HTML: &str = include_str!("../companion/index.html");

/// Start the web server on the given port with HTTPS (required for iPhone mic access).
pub async fn run_server(state: Arc<ServerState>, port: u16, local_ips: Vec<String>) {
    let state_for_ws = state.clone();

    let app = Router::new()
        .route(
            "/",
            get(|| async { Html(COMPANION_HTML) }),
        )
        .route(
            "/ws",
            get(move |ws: WebSocketUpgrade| {
                let st = state_for_ws.clone();
                async move { ws.on_upgrade(move |socket| handle_ws(socket, st)) }
            }),
        )
        .layer(CorsLayer::permissive());

    // Generate self-signed TLS cert (iOS requires HTTPS for microphone access)
    let mut sans = local_ips;
    sans.push("localhost".to_string());
    eprintln!("Generating self-signed cert for SANs: {sans:?}");

    let certified_key = rcgen::generate_simple_self_signed(sans)
        .expect("Failed to generate self-signed certificate");
    let cert_pem = certified_key.cert.pem();
    let key_pem = certified_key.key_pair.serialize_pem();

    let tls_config = RustlsConfig::from_pem(cert_pem.into_bytes(), key_pem.into_bytes())
        .await
        .expect("Failed to create TLS config");

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
            serde_json::json!({"type": "status", "state": "connected"}).to_string().into(),
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
            .to_string().into(),
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
                eprintln!(
                    "Received audio: {} bytes ({:.1}s at 16kHz)",
                    audio_data.len(),
                    audio_data.len() as f64 / (16000.0 * 4.0) // f32 = 4 bytes
                );

                // Audio is raw f32 PCM at 16kHz mono, sent as little-endian bytes
                let samples = bytes_to_f32_samples(&audio_data);

                if samples.len() < (16000.0 * 0.3) as usize {
                    let _ = socket
                        .send(Message::Text(
                            serde_json::json!({
                                "type": "error",
                                "message": "Recording too short"
                            })
                            .to_string().into(),
                        ))
                        .await;
                    continue;
                }

                // Notify phone we're transcribing
                let _ = socket
                    .send(Message::Text(
                        serde_json::json!({"type": "status", "state": "transcribing"})
                            .to_string().into(),
                    ))
                    .await;

                // Transcribe
                let text = {
                    let guard = state.transcriber.lock().unwrap();
                    match guard.as_ref() {
                        Some(t) => match t.transcribe(&samples) {
                            Ok(result) if !result.text.is_empty() => Ok(result.text),
                            Ok(_) => Err("No speech detected".to_string()),
                            Err(e) => Err(format!("Transcription failed: {e}")),
                        },
                        None => Err("Model not loaded yet".to_string()),
                    }
                };

                match text {
                    Ok(text) => {
                        eprintln!("Transcribed: {text}");

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
                                    "text": text,
                                    "injected": inject_result.is_ok(),
                                    "error": inject_result.err().map(|e| e.to_string()),
                                })
                                .to_string().into(),
                            ))
                            .await;
                    }
                    Err(err) => {
                        let _ = socket
                            .send(Message::Text(
                                serde_json::json!({
                                    "type": "error",
                                    "message": err,
                                })
                                .to_string().into(),
                            ))
                            .await;
                    }
                }

                // Back to ready
                let _ = socket
                    .send(Message::Text(
                        serde_json::json!({"type": "status", "state": "ready"}).to_string().into(),
                    ))
                    .await;
            }
            Message::Text(text) => {
                // Handle control messages from phone
                if let Ok(msg) = serde_json::from_str::<serde_json::Value>(&text) {
                    if msg.get("type").and_then(|v| v.as_str()) == Some("ping") {
                        let _ = socket
                            .send(Message::Text(
                                serde_json::json!({"type": "pong"}).to_string().into(),
                            ))
                            .await;
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

/// Get the local IP addresses to display to the user.
pub fn get_local_ips() -> Vec<String> {
    let mut ips = Vec::new();

    // Try to get IPs by binding a UDP socket (doesn't actually send anything)
    if let Ok(socket) = std::net::UdpSocket::bind("0.0.0.0:0") {
        // Connect to a public IP to determine which local interface would be used
        if socket.connect("8.8.8.8:80").is_ok() {
            if let Ok(addr) = socket.local_addr() {
                ips.push(addr.ip().to_string());
            }
        }
    }

    if ips.is_empty() {
        ips.push("127.0.0.1".to_string());
    }

    ips
}
