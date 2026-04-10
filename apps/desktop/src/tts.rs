use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

// ── SAPI Speaker (Windows built-in) ────────────────────────────────────

pub struct SapiSpeaker {
    engine: Mutex<tts::Tts>,
}

impl SapiSpeaker {
    pub fn new() -> Result<Self, String> {
        let engine = tts::Tts::default().map_err(|e| format!("Failed to init TTS: {e}"))?;
        Ok(Self {
            engine: Mutex::new(engine),
        })
    }

    pub fn speak(&self, text: &str) -> Result<(), String> {
        let mut engine = self.engine.lock().unwrap();
        engine
            .speak(text, true)
            .map_err(|e| format!("TTS speak failed: {e}"))?;
        Ok(())
    }

    pub fn stop(&self) -> Result<(), String> {
        let mut engine = self.engine.lock().unwrap();
        engine
            .stop()
            .map_err(|e| format!("TTS stop failed: {e}"))?;
        Ok(())
    }

    pub fn is_speaking(&self) -> bool {
        let engine = self.engine.lock().unwrap();
        engine.is_speaking().unwrap_or(false)
    }

    pub fn set_rate(&self, rate: f32) -> Result<(), String> {
        let mut engine = self.engine.lock().unwrap();
        engine
            .set_rate(rate)
            .map_err(|e| format!("TTS set rate failed: {e}"))?;
        Ok(())
    }

    pub fn get_rate(&self) -> f32 {
        let engine = self.engine.lock().unwrap();
        engine.get_rate().unwrap_or(1.0)
    }

    pub fn list_voices(&self) -> Vec<String> {
        let engine = self.engine.lock().unwrap();
        engine
            .voices()
            .unwrap_or_default()
            .into_iter()
            .map(|v| v.name().to_string())
            .collect()
    }

    pub fn set_voice(&self, name: &str) -> Result<(), String> {
        let mut engine = self.engine.lock().unwrap();
        let voices = engine.voices().map_err(|e| format!("TTS voices failed: {e}"))?;
        let voice = voices
            .into_iter()
            .find(|v| v.name() == name)
            .ok_or_else(|| format!("Voice not found: {name}"))?;
        engine
            .set_voice(&voice)
            .map_err(|e| format!("TTS set voice failed: {e}"))?;
        Ok(())
    }
}

// ── Piper Speaker (neural TTS via piper.exe) ───────────────────────────

/// Piper TTS voices available for download.
pub struct PiperVoice {
    pub id: &'static str,
    pub name: &'static str,
    pub model_url: &'static str,
    pub config_url: &'static str,
    pub model_filename: &'static str,
    pub config_filename: &'static str,
    pub size_bytes: u64,
    pub sample_rate: u32,
}

pub const PIPER_VOICES: &[PiperVoice] = &[
    // ── Female voices ──
    PiperVoice {
        id: "amy-medium",
        name: "Amy (Female, US)",
        model_url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx",
        config_url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
        model_filename: "en_US-amy-medium.onnx",
        config_filename: "en_US-amy-medium.onnx.json",
        size_bytes: 63_000_000,
        sample_rate: 22050,
    },
    PiperVoice {
        id: "lessac-medium",
        name: "Lessac (Female, US)",
        model_url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        config_url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        model_filename: "en_US-lessac-medium.onnx",
        config_filename: "en_US-lessac-medium.onnx.json",
        size_bytes: 63_000_000,
        sample_rate: 22050,
    },
    PiperVoice {
        id: "jenny-medium",
        name: "Jenny (Female, GB)",
        model_url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx",
        config_url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx.json",
        model_filename: "en_GB-jenny_dioco-medium.onnx",
        config_filename: "en_GB-jenny_dioco-medium.onnx.json",
        size_bytes: 63_000_000,
        sample_rate: 22050,
    },
    PiperVoice {
        id: "alba-medium",
        name: "Alba (Female, GB)",
        model_url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/en_GB-alba-medium.onnx",
        config_url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/en_GB-alba-medium.onnx.json",
        model_filename: "en_GB-alba-medium.onnx",
        config_filename: "en_GB-alba-medium.onnx.json",
        size_bytes: 63_000_000,
        sample_rate: 22050,
    },
    // ── Male voices ──
    PiperVoice {
        id: "ryan-medium",
        name: "Ryan (Male, US)",
        model_url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx",
        config_url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json",
        model_filename: "en_US-ryan-medium.onnx",
        config_filename: "en_US-ryan-medium.onnx.json",
        size_bytes: 63_000_000,
        sample_rate: 22050,
    },
    PiperVoice {
        id: "joe-medium",
        name: "Joe (Male, US)",
        model_url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/joe/medium/en_US-joe-medium.onnx",
        config_url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/joe/medium/en_US-joe-medium.onnx.json",
        model_filename: "en_US-joe-medium.onnx",
        config_filename: "en_US-joe-medium.onnx.json",
        size_bytes: 63_000_000,
        sample_rate: 22050,
    },
    PiperVoice {
        id: "alan-medium",
        name: "Alan (Male, GB)",
        model_url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx",
        config_url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json",
        model_filename: "en_GB-alan-medium.onnx",
        config_filename: "en_GB-alan-medium.onnx.json",
        size_bytes: 63_000_000,
        sample_rate: 22050,
    },
    // ── High quality ──
    PiperVoice {
        id: "libritts-high",
        name: "LibriTTS (Multi-speaker, high quality)",
        model_url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts/high/en_US-libritts-high.onnx",
        config_url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts/high/en_US-libritts-high.onnx.json",
        model_filename: "en_US-libritts-high.onnx",
        config_filename: "en_US-libritts-high.onnx.json",
        size_bytes: 75_000_000,
        sample_rate: 22050,
    },
    PiperVoice {
        id: "lessac-high",
        name: "Lessac (Female, US, high quality)",
        model_url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/high/en_US-lessac-high.onnx",
        config_url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/high/en_US-lessac-high.onnx.json",
        model_filename: "en_US-lessac-high.onnx",
        config_filename: "en_US-lessac-high.onnx.json",
        size_bytes: 75_000_000,
        sample_rate: 22050,
    },
    PiperVoice {
        id: "ryan-high",
        name: "Ryan (Male, US, high quality)",
        model_url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx",
        config_url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx.json",
        model_filename: "en_US-ryan-high.onnx",
        config_filename: "en_US-ryan-high.onnx.json",
        size_bytes: 75_000_000,
        sample_rate: 22050,
    },
];

/// URL for the piper.exe Windows release.
const PIPER_EXE_URL: &str = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip";

pub struct PiperSpeaker {
    piper_dir: PathBuf,
    voice_dir: PathBuf,
    current_voice: Mutex<String>,
    playing: AtomicBool,
    // rodio sink handle for stopping playback
    sink: Mutex<Option<rodio::Sink>>,
}

impl PiperSpeaker {
    pub fn new(piper_dir: PathBuf, voice_dir: PathBuf) -> Self {
        Self {
            piper_dir,
            voice_dir,
            current_voice: Mutex::new(String::new()),
            playing: AtomicBool::new(false),
            sink: Mutex::new(None),
        }
    }

    pub fn piper_dir(&self) -> &Path {
        &self.piper_dir
    }

    pub fn voice_dir(&self) -> &Path {
        &self.voice_dir
    }

    fn piper_exe(&self) -> PathBuf {
        self.piper_dir.join("piper").join("piper.exe")
    }

    pub fn has_piper_exe(&self) -> bool {
        self.piper_exe().exists()
    }

    /// Check if a voice model is installed.
    pub fn has_voice(&self, voice_id: &str) -> bool {
        if let Some(voice) = PIPER_VOICES.iter().find(|v| v.id == voice_id) {
            self.voice_dir.join(voice.model_filename).exists()
                && self.voice_dir.join(voice.config_filename).exists()
        } else {
            false
        }
    }

    /// Delete a downloaded voice model.
    pub fn delete_voice(&self, voice_id: &str) -> Result<(), String> {
        let voice = PIPER_VOICES
            .iter()
            .find(|v| v.id == voice_id)
            .ok_or_else(|| format!("Unknown voice: {voice_id}"))?;

        let model = self.voice_dir.join(voice.model_filename);
        let config = self.voice_dir.join(voice.config_filename);

        if model.exists() {
            std::fs::remove_file(&model)
                .map_err(|e| format!("Failed to delete model: {e}"))?;
        }
        if config.exists() {
            std::fs::remove_file(&config)
                .map_err(|e| format!("Failed to delete config: {e}"))?;
        }
        Ok(())
    }

    /// List installed voice IDs.
    pub fn installed_voices(&self) -> Vec<String> {
        PIPER_VOICES
            .iter()
            .filter(|v| self.has_voice(v.id))
            .map(|v| v.id.to_string())
            .collect()
    }

    pub fn set_voice(&self, voice_id: &str) {
        *self.current_voice.lock().unwrap() = voice_id.to_string();
    }

    /// Speak text using piper.exe. Generates WAV then plays via rodio.
    pub fn speak(&self, text: &str) -> Result<(), String> {
        let voice_id = self.current_voice.lock().unwrap().clone();
        if voice_id.is_empty() {
            return Err("No Piper voice selected".to_string());
        }

        let voice = PIPER_VOICES
            .iter()
            .find(|v| v.id == voice_id)
            .ok_or_else(|| format!("Unknown voice: {voice_id}"))?;

        let model_path = self.voice_dir.join(voice.model_filename);
        let config_path = self.voice_dir.join(voice.config_filename);

        if !model_path.exists() {
            return Err(format!("Voice model not found: {}", voice.model_filename));
        }

        let piper_exe = self.piper_exe();
        if !piper_exe.exists() {
            return Err("piper.exe not found — download it in settings".to_string());
        }

        // Run piper.exe: echo text | piper.exe --model X --config X --output_raw
        let output = std::process::Command::new(&piper_exe)
            .arg("--model")
            .arg(&model_path)
            .arg("--config")
            .arg(&config_path)
            .arg("--output_raw")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .creation_flags(0x08000000) // CREATE_NO_WINDOW
            .spawn()
            .and_then(|mut child| {
                use std::io::Write;
                if let Some(ref mut stdin) = child.stdin {
                    let _ = stdin.write_all(text.as_bytes());
                }
                drop(child.stdin.take());
                child.wait_with_output()
            })
            .map_err(|e| format!("Failed to run piper.exe: {e}"))?;

        if !output.status.success() {
            return Err("piper.exe failed".to_string());
        }

        let pcm_data = output.stdout;
        if pcm_data.is_empty() {
            return Err("Piper produced no audio".to_string());
        }

        // Play raw PCM (16-bit signed LE, mono) via rodio
        self.playing.store(true, Ordering::Relaxed);

        // Convert raw bytes to i16 samples
        let samples: Vec<i16> = pcm_data
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();

        let source = rodio::buffer::SamplesBuffer::new(1, voice.sample_rate, samples);

        let (_stream, stream_handle) =
            rodio::OutputStream::try_default().map_err(|e| format!("Audio output error: {e}"))?;

        let sink =
            rodio::Sink::try_new(&stream_handle).map_err(|e| format!("Audio sink error: {e}"))?;

        sink.append(source);

        // Store sink so we can stop it later
        *self.sink.lock().unwrap() = Some(sink);

        // Block until playback finishes (or sink is stopped)
        // Re-acquire lock each check to allow stop() from another thread
        loop {
            let done = {
                let guard = self.sink.lock().unwrap();
                match guard.as_ref() {
                    Some(s) => s.empty(),
                    None => true,
                }
            };
            if done {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        self.playing.store(false, Ordering::Relaxed);
        *self.sink.lock().unwrap() = None;

        Ok(())
    }

    pub fn stop(&self) {
        if let Some(sink) = self.sink.lock().unwrap().take() {
            sink.stop();
        }
        self.playing.store(false, Ordering::Relaxed);
    }

    pub fn is_speaking(&self) -> bool {
        self.playing.load(Ordering::Relaxed)
    }
}

// Add creation_flags support for Command on Windows
use std::os::windows::process::CommandExt;

// ── Piper download helpers ─────────────────────────────────────────────

/// Download and extract piper.exe from the GitHub release zip.
pub async fn download_piper_exe(
    piper_dir: &Path,
    on_progress: impl Fn(u64, u64),
) -> Result<(), String> {
    use futures_util::StreamExt;

    std::fs::create_dir_all(piper_dir)
        .map_err(|e| format!("Failed to create piper dir: {e}"))?;

    let zip_path = piper_dir.join("piper_windows.zip");

    // Download zip
    let resp = reqwest::get(PIPER_EXE_URL)
        .await
        .map_err(|e| format!("Download failed: {e}"))?;

    let total = resp.content_length().unwrap_or(0);
    let mut downloaded: u64 = 0;
    let mut bytes = Vec::new();
    let mut stream = resp.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Download error: {e}"))?;
        downloaded += chunk.len() as u64;
        bytes.extend_from_slice(&chunk);
        on_progress(downloaded, total);
    }

    std::fs::write(&zip_path, &bytes).map_err(|e| format!("Failed to save zip: {e}"))?;

    // Extract zip
    let file = std::fs::File::open(&zip_path).map_err(|e| format!("Failed to open zip: {e}"))?;
    let mut archive =
        zip::ZipArchive::new(file).map_err(|e| format!("Failed to read zip: {e}"))?;

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).map_err(|e| format!("Zip error: {e}"))?;
        let out_path = piper_dir.join(entry.name());
        if entry.is_dir() {
            std::fs::create_dir_all(&out_path).ok();
        } else {
            if let Some(parent) = out_path.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            let mut outfile =
                std::fs::File::create(&out_path).map_err(|e| format!("Extract error: {e}"))?;
            std::io::copy(&mut entry, &mut outfile)
                .map_err(|e| format!("Extract error: {e}"))?;
        }
    }

    // Clean up zip
    let _ = std::fs::remove_file(&zip_path);

    Ok(())
}

/// Download a Piper voice model and config.
pub async fn download_piper_voice(
    voice_dir: &Path,
    voice_id: &str,
    on_progress: impl Fn(u64, u64),
) -> Result<(), String> {
    let voice = PIPER_VOICES
        .iter()
        .find(|v| v.id == voice_id)
        .ok_or_else(|| format!("Unknown voice: {voice_id}"))?;

    std::fs::create_dir_all(voice_dir)
        .map_err(|e| format!("Failed to create voice dir: {e}"))?;

    // Download model file
    let resp = reqwest::get(voice.model_url)
        .await
        .map_err(|e| format!("Model download failed: {e}"))?;

    let total = resp.content_length().unwrap_or(voice.size_bytes);
    let mut downloaded: u64 = 0;

    use futures_util::StreamExt;
    let mut bytes = Vec::new();
    let mut stream = resp.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Download error: {e}"))?;
        downloaded += chunk.len() as u64;
        bytes.extend_from_slice(&chunk);
        on_progress(downloaded, total);
    }

    let model_path = voice_dir.join(voice.model_filename);
    std::fs::write(&model_path, &bytes).map_err(|e| format!("Failed to save model: {e}"))?;

    // Download config file (small, no progress needed)
    let config_bytes = reqwest::get(voice.config_url)
        .await
        .map_err(|e| format!("Config download failed: {e}"))?
        .bytes()
        .await
        .map_err(|e| format!("Config download error: {e}"))?;

    let config_path = voice_dir.join(voice.config_filename);
    std::fs::write(&config_path, &config_bytes)
        .map_err(|e| format!("Failed to save config: {e}"))?;

    Ok(())
}

// ── Capture selected text ──────────────────────────────────────────────

/// Capture currently highlighted/selected text from any application.
/// Simulates Ctrl+C, waits briefly, then reads the clipboard.
pub fn capture_selected_text() -> Result<String, String> {
    use arboard::Clipboard;
    use windows::Win32::UI::Input::KeyboardAndMouse::{
        SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYBD_EVENT_FLAGS,
        KEYEVENTF_KEYUP, VIRTUAL_KEY,
    };

    let vk_control = VIRTUAL_KEY(0x11); // VK_CONTROL
    let vk_shift = VIRTUAL_KEY(0x10); // VK_SHIFT
    let vk_alt = VIRTUAL_KEY(0x12); // VK_MENU (Alt)
    let vk_c = VIRTUAL_KEY(0x43); // VK_C

    // Save current clipboard contents, then DROP the handle before simulating Ctrl+C.
    // Windows only allows one process to hold the clipboard open at a time.
    let old_text = {
        let mut cb = Clipboard::new().map_err(|e| format!("Clipboard error: {e}"))?;
        cb.get_text().unwrap_or_default()
    }; // cb dropped here — clipboard is free for Ctrl+C

    fn key_up(vk: VIRTUAL_KEY) -> INPUT {
        INPUT {
            r#type: INPUT_KEYBOARD,
            Anonymous: INPUT_0 {
                ki: KEYBDINPUT {
                    wVk: vk,
                    wScan: 0,
                    dwFlags: KEYEVENTF_KEYUP,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        }
    }

    fn key_down(vk: VIRTUAL_KEY) -> INPUT {
        INPUT {
            r#type: INPUT_KEYBOARD,
            Anonymous: INPUT_0 {
                ki: KEYBDINPUT {
                    wVk: vk,
                    wScan: 0,
                    dwFlags: KEYBD_EVENT_FLAGS(0),
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        }
    }

    // Release any modifier keys that might still be held from the hotkey combo,
    // then send a clean Ctrl+C, then release Ctrl.
    let inputs = [
        key_up(vk_shift),
        key_up(vk_alt),
        key_up(vk_control),
        key_down(vk_control),
        key_down(vk_c),
        key_up(vk_c),
        key_up(vk_control),
    ];

    unsafe {
        SendInput(&inputs, std::mem::size_of::<INPUT>() as i32);
    }

    // Wait for clipboard to update — 300ms is reliable across apps
    std::thread::sleep(std::time::Duration::from_millis(300));

    // Read clipboard, then restore the old contents
    let new_text = {
        let mut cb = Clipboard::new().map_err(|e| format!("Clipboard error: {e}"))?;
        let text = cb.get_text().unwrap_or_default();
        // Restore old clipboard so we don't clobber the user's copy buffer
        if text != old_text && !old_text.is_empty() {
            let _ = cb.set_text(&old_text);
        }
        text
    };

    if new_text.is_empty() {
        return Err("No text selected".to_string());
    }

    Ok(new_text)
}
