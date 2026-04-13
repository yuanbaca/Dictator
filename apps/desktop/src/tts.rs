use std::collections::HashSet;
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

/// A custom Piper voice discovered on disk (not in the built-in list).
pub struct CustomPiperVoice {
    pub id: String,
    pub display_name: String,
    pub model_filename: String,
    pub config_filename: String,
    pub size_bytes: u64,
    pub sample_rate: u32,
}

/// Read the sample rate from a Piper voice config JSON file.
fn read_piper_sample_rate(config_path: &Path) -> Option<u32> {
    let data = std::fs::read_to_string(config_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&data).ok()?;
    json.get("audio")?.get("sample_rate")?.as_u64().map(|r| r as u32)
}

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
        if voice_id.starts_with("custom:") {
            let stem = &voice_id["custom:".len()..];
            let model = self.voice_dir.join(format!("{stem}.onnx"));
            let config = self.voice_dir.join(format!("{stem}.onnx.json"));
            model.exists() && config.exists()
        } else if let Some(voice) = PIPER_VOICES.iter().find(|v| v.id == voice_id) {
            self.voice_dir.join(voice.model_filename).exists()
                && self.voice_dir.join(voice.config_filename).exists()
        } else {
            false
        }
    }

    /// Delete a downloaded voice model.
    pub fn delete_voice(&self, voice_id: &str) -> Result<(), String> {
        let (model, config) = if voice_id.starts_with("custom:") {
            let stem = &voice_id["custom:".len()..];
            (
                self.voice_dir.join(format!("{stem}.onnx")),
                self.voice_dir.join(format!("{stem}.onnx.json")),
            )
        } else {
            let voice = PIPER_VOICES
                .iter()
                .find(|v| v.id == voice_id)
                .ok_or_else(|| format!("Unknown voice: {voice_id}"))?;
            (
                self.voice_dir.join(voice.model_filename),
                self.voice_dir.join(voice.config_filename),
            )
        };

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

    /// List installed voice IDs (built-in + custom).
    pub fn installed_voices(&self) -> Vec<String> {
        let mut voices: Vec<String> = PIPER_VOICES
            .iter()
            .filter(|v| self.has_voice(v.id))
            .map(|v| v.id.to_string())
            .collect();
        for cv in self.discover_custom_voices() {
            voices.push(cv.id);
        }
        voices
    }

    pub fn set_voice(&self, voice_id: &str) {
        *self.current_voice.lock().unwrap() = voice_id.to_string();
    }

    /// Discover custom voice files in the voice directory.
    /// Returns voices that have both `.onnx` and `.onnx.json` and aren't built-in.
    pub fn discover_custom_voices(&self) -> Vec<CustomPiperVoice> {
        let mut custom = Vec::new();
        let entries = match std::fs::read_dir(&self.voice_dir) {
            Ok(e) => e,
            Err(_) => return custom,
        };

        // Collect built-in model filenames for exclusion
        let builtin_files: HashSet<&str> =
            PIPER_VOICES.iter().map(|v| v.model_filename).collect();

        for entry in entries.flatten() {
            let path = entry.path();
            let filename = match path.file_name().and_then(|f| f.to_str()) {
                Some(f) => f.to_string(),
                None => continue,
            };

            // Only look at .onnx files (skip .onnx.json)
            if !filename.ends_with(".onnx") || filename.ends_with(".onnx.json") {
                continue;
            }

            // Skip built-in voices
            if builtin_files.contains(filename.as_str()) {
                continue;
            }

            // Check matching config exists
            let config_filename = format!("{filename}.json");
            let config_path = self.voice_dir.join(&config_filename);
            if !config_path.exists() {
                continue;
            }

            // Read sample rate from config
            let sample_rate = read_piper_sample_rate(&config_path).unwrap_or(22050);

            // File size
            let size_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

            // Generate display name from filename stem
            let stem = filename.trim_end_matches(".onnx");
            let display_name = stem
                .replace(['-', '_'], " ")
                .split_whitespace()
                .map(|w| {
                    let mut c = w.chars();
                    match c.next() {
                        None => String::new(),
                        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                    }
                })
                .collect::<Vec<_>>()
                .join(" ");

            let id = format!("custom:{stem}");

            custom.push(CustomPiperVoice {
                id,
                display_name,
                model_filename: filename,
                config_filename,
                size_bytes,
                sample_rate,
            });
        }

        custom
    }

    /// Speak text using piper.exe. Generates WAV then plays via rodio.
    pub fn speak(&self, text: &str) -> Result<(), String> {
        let voice_id = self.current_voice.lock().unwrap().clone();
        if voice_id.is_empty() {
            return Err("No Piper voice selected".to_string());
        }

        // Resolve model path, config path, and sample rate for built-in or custom voices
        let (model_path, config_path, sample_rate) = if voice_id.starts_with("custom:") {
            let stem = &voice_id["custom:".len()..];
            let model = self.voice_dir.join(format!("{stem}.onnx"));
            let config = self.voice_dir.join(format!("{stem}.onnx.json"));
            if !model.exists() {
                return Err(format!("Custom voice model not found: {stem}.onnx"));
            }
            if !config.exists() {
                return Err(format!("Custom voice config not found: {stem}.onnx.json"));
            }
            let sr = read_piper_sample_rate(&config).unwrap_or(22050);
            (model, config, sr)
        } else {
            let voice = PIPER_VOICES
                .iter()
                .find(|v| v.id == voice_id)
                .ok_or_else(|| format!("Unknown voice: {voice_id}"))?;
            let model = self.voice_dir.join(voice.model_filename);
            let config = self.voice_dir.join(voice.config_filename);
            if !model.exists() {
                return Err(format!("Voice model not found: {}", voice.model_filename));
            }
            (model, config, voice.sample_rate)
        };

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

        let source = rodio::buffer::SamplesBuffer::new(1, sample_rate, samples);

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

    pub fn pause(&self) {
        let guard = self.sink.lock().unwrap();
        if let Some(ref sink) = *guard {
            sink.pause();
        }
    }

    pub fn resume(&self) {
        let guard = self.sink.lock().unwrap();
        if let Some(ref sink) = *guard {
            sink.play();
        }
    }

    pub fn is_paused(&self) -> bool {
        let guard = self.sink.lock().unwrap();
        match guard.as_ref() {
            Some(sink) => sink.is_paused(),
            None => false,
        }
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

// ── Minimax Cloud TTS ─────────────────────────────────────────────────

/// Known Minimax voice IDs and display names.
pub const MINIMAX_VOICES: &[(&str, &str)] = &[
    ("English_Graceful_Lady", "Graceful Lady (Female)"),
    ("English_Insightful_Speaker", "Insightful Speaker (Male)"),
    ("English_radiant_girl", "Radiant Girl (Female)"),
    ("English_Persuasive_Man", "Persuasive Man (Male)"),
    ("English_expressive_narrator", "Expressive Narrator"),
    ("Wise_Woman", "Wise Woman (Female)"),
    ("Friendly_Person", "Friendly Person"),
    ("Inspirational_girl", "Inspirational Girl (Female)"),
    ("Deep_Voice_Man", "Deep Voice Man (Male)"),
    ("Calm_Woman", "Calm Woman (Female)"),
    ("Casual_Guy", "Casual Guy (Male)"),
    ("Lively_Girl", "Lively Girl (Female)"),
    ("Patient_Man", "Patient Man (Male)"),
    ("Young_Knight", "Young Knight (Male)"),
    ("Lovely_Girl", "Lovely Girl (Female)"),
    ("Sweet_Girl_2", "Sweet Girl (Female)"),
    ("Elegant_Man", "Elegant Man (Male)"),
    ("English_Lucky_Robot", "Lucky Robot"),
];

/// Speak text via Minimax cloud TTS (blocking HTTP call + rodio playback).
/// Must NOT be called from within a tokio runtime — use std::thread::spawn.
/// `on_audio_ready` is called once the API returns and audio is about to play.
pub fn minimax_speak(
    api_key: &str,
    voice_id: &str,
    text: &str,
    playing: &AtomicBool,
    sink_holder: &Mutex<Option<rodio::Sink>>,
    on_audio_ready: impl FnOnce(),
) -> Result<(), String> {
    if api_key.is_empty() {
        return Err("Minimax API key not set".to_string());
    }

    // Build request JSON
    let body = serde_json::json!({
        "model": "speech-02-hd",
        "text": text,
        "stream": false,
        "language_boost": "English",
        "voice_setting": {
            "voice_id": voice_id,
            "speed": 1.0,
            "vol": 1.0,
        },
        "audio_setting": {
            "sample_rate": 24000,
            "format": "mp3",
            "channel": 1,
        }
    });

    // Blocking HTTP POST — must run outside tokio runtime
    let mut resp = ureq::post("https://api.minimax.io/v1/t2a_v2")
        .header("Authorization", &format!("Bearer {api_key}"))
        .send_json(&body)
        .map_err(|e| format!("Minimax API error: {e}"))?;

    let json: serde_json::Value = resp.body_mut().read_json()
        .map_err(|e| format!("Minimax response parse error: {e}"))?;

    // Check for API errors
    let status_code = json["base_resp"]["status_code"].as_i64().unwrap_or(-1);
    if status_code != 0 {
        let msg = json["base_resp"]["status_msg"].as_str().unwrap_or("Unknown error");
        return Err(format!("Minimax API error: {msg}"));
    }

    let hex_audio = json["data"]["audio"].as_str()
        .ok_or("No audio data in Minimax response")?;

    // Decode hex to bytes
    let audio_bytes: Vec<u8> = (0..hex_audio.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex_audio[i..i + 2], 16)
            .map_err(|e| format!("Hex decode error: {e}")))
        .collect::<Result<Vec<u8>, String>>()?;

    // Play MP3 via rodio
    on_audio_ready();
    playing.store(true, Ordering::Relaxed);

    let cursor = std::io::Cursor::new(audio_bytes);
    let source = rodio::Decoder::new(cursor)
        .map_err(|e| format!("Audio decode error: {e}"))?;

    let (_stream, stream_handle) =
        rodio::OutputStream::try_default().map_err(|e| format!("Audio output error: {e}"))?;
    let sink =
        rodio::Sink::try_new(&stream_handle).map_err(|e| format!("Audio sink error: {e}"))?;

    sink.append(source);
    *sink_holder.lock().unwrap() = Some(sink);

    // Wait for playback to finish (or be stopped externally)
    loop {
        {
            let guard = sink_holder.lock().unwrap();
            match guard.as_ref() {
                Some(s) if s.empty() => break,
                None => break,
                _ => {}
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    playing.store(false, Ordering::Relaxed);
    *sink_holder.lock().unwrap() = None;

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
