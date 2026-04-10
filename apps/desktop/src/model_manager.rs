//! Model download and management.

use futures_util::StreamExt;
use std::path::PathBuf;
use tokio::io::AsyncWriteExt;

/// Known LLM models available for download.
pub struct ModelInfo {
    pub id: &'static str,
    pub name: &'static str,
    pub url: &'static str,
    pub filename: &'static str,
    pub size_bytes: u64, // approximate
}

pub const AVAILABLE_MODELS: &[ModelInfo] = &[
    ModelInfo {
        id: "phi3-mini",
        name: "Phi-3 Mini 3.8B (recommended)",
        url: "https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF/resolve/main/Phi-3-mini-4k-instruct-Q4_K_M.gguf",
        filename: "Phi-3-mini-4k-instruct-Q4_K_M.gguf",
        size_bytes: 2_394_805_248,
    },
    ModelInfo {
        id: "llama32-3b",
        name: "Llama 3.2 3B (smaller, fast)",
        url: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        filename: "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        size_bytes: 2_019_377_152,
    },
];

/// Whisper speech-to-text models available for download.
pub const WHISPER_MODELS: &[ModelInfo] = &[
    ModelInfo {
        id: "tiny-en",
        name: "Tiny (English, fastest)",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
        filename: "ggml-tiny.en.bin",
        size_bytes: 77_704_715,
    },
    ModelInfo {
        id: "base-en",
        name: "Base (English, recommended)",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
        filename: "ggml-base.en.bin",
        size_bytes: 147_964_211,
    },
    ModelInfo {
        id: "small-en",
        name: "Small (English, higher accuracy)",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin",
        filename: "ggml-small.en.bin",
        size_bytes: 487_601_967,
    },
    ModelInfo {
        id: "medium-en",
        name: "Medium (English, best accuracy)",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin",
        filename: "ggml-medium.en.bin",
        size_bytes: 1_533_774_781,
    },
];

/// Find the first installed Whisper model, preferring base > small > tiny > medium.
pub fn find_whisper_model() -> Option<std::path::PathBuf> {
    let dir = models_dir();
    // Preference order: base (good balance), small, tiny, medium, then any .bin
    let preferred = ["ggml-base.en.bin", "ggml-small.en.bin", "ggml-tiny.en.bin", "ggml-medium.en.bin"];
    for name in &preferred {
        let p = dir.join(name);
        if p.exists() {
            return Some(p);
        }
    }
    // Check for any whisper model file
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("ggml-") && name.ends_with(".bin") {
                    return Some(path);
                }
            }
        }
    }
    None
}

/// Download a Whisper model by its ID.
pub async fn download_whisper_model<F>(
    model_id: &str,
    on_progress: F,
) -> Result<PathBuf, String>
where
    F: Fn(u64, u64) + Send + 'static,
{
    let model = WHISPER_MODELS
        .iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| format!("Unknown whisper model: {model_id}"))?;

    let dir = models_dir();
    let dest = dir.join(model.filename);
    let temp = dir.join(format!("{}.part", model.filename));

    eprintln!("Downloading whisper model {} to {}", model.name, dest.display());

    let response = reqwest::get(model.url)
        .await
        .map_err(|e| format!("Download request failed: {e}"))?;

    if !response.status().is_success() {
        return Err(format!("Download failed: HTTP {}", response.status()));
    }

    let total = response.content_length().unwrap_or(model.size_bytes);

    let mut file = tokio::fs::File::create(&temp)
        .await
        .map_err(|e| format!("Failed to create file: {e}"))?;

    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Download error: {e}"))?;
        file.write_all(&chunk)
            .await
            .map_err(|e| format!("Write error: {e}"))?;
        downloaded += chunk.len() as u64;
        on_progress(downloaded, total);
    }

    file.flush().await.map_err(|e| format!("Flush error: {e}"))?;
    drop(file);

    tokio::fs::rename(&temp, &dest)
        .await
        .map_err(|e| format!("Rename error: {e}"))?;

    eprintln!("Whisper model download complete: {}", dest.display());
    Ok(dest)
}

/// Get the models directory path, creating it if needed.
///
/// Search order:
///   1. Existing models/ dirs relative to exe (for dev builds with deep target/ paths)
///   2. Existing models/ dirs relative to CWD
///   3. Default: models/ next to the exe (stable for portable installs)
pub fn models_dir() -> PathBuf {
    // Try exe-relative first (for deployed and dev builds)
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let candidates = [
                exe_dir.join("../../../../models"), // from target/release/ in dev
                exe_dir.join("../../models"),
                exe_dir.join("models"),             // portable: models/ next to exe
            ];
            for c in &candidates {
                if c.exists() {
                    return c.clone();
                }
            }
        }
    }

    // Try CWD-relative
    let cwd_candidates = [
        PathBuf::from("../../models"),
        PathBuf::from("models"),
    ];
    for c in &cwd_candidates {
        if c.exists() {
            return c.clone();
        }
    }

    // Default: create models/ next to the exe (portable-friendly)
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let default = exe_dir.join("models");
            let _ = std::fs::create_dir_all(&default);
            return default;
        }
    }

    // Final fallback: CWD
    let default = PathBuf::from("models");
    let _ = std::fs::create_dir_all(&default);
    default
}

/// Check which LLM models are present in the models directory.
pub fn installed_llm_models() -> Vec<(String, String)> {
    let dir = models_dir();
    let mut results = Vec::new();

    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "gguf") {
                let filename = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                // Match to known model
                let name = AVAILABLE_MODELS
                    .iter()
                    .find(|m| m.filename == filename)
                    .map(|m| m.name.to_string())
                    .unwrap_or_else(|| filename.clone());
                results.push((filename, name));
            }
        }
    }

    results
}

/// Check if any LLM model is installed.
pub fn has_llm_model() -> bool {
    !installed_llm_models().is_empty()
}

/// Download a model by its ID. Calls `on_progress(bytes_downloaded, total_bytes)`.
pub async fn download_model<F>(
    model_id: &str,
    on_progress: F,
) -> Result<PathBuf, String>
where
    F: Fn(u64, u64) + Send + 'static,
{
    let model = AVAILABLE_MODELS
        .iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| format!("Unknown model: {model_id}"))?;

    let dir = models_dir();
    let dest = dir.join(model.filename);
    let temp = dir.join(format!("{}.part", model.filename));

    eprintln!("Downloading {} to {}", model.name, dest.display());

    let response = reqwest::get(model.url)
        .await
        .map_err(|e| format!("Download request failed: {e}"))?;

    if !response.status().is_success() {
        return Err(format!("Download failed: HTTP {}", response.status()));
    }

    let total = response.content_length().unwrap_or(model.size_bytes);

    let mut file = tokio::fs::File::create(&temp)
        .await
        .map_err(|e| format!("Failed to create file: {e}"))?;

    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Download error: {e}"))?;
        file.write_all(&chunk)
            .await
            .map_err(|e| format!("Write error: {e}"))?;
        downloaded += chunk.len() as u64;
        on_progress(downloaded, total);
    }

    file.flush().await.map_err(|e| format!("Flush error: {e}"))?;
    drop(file);

    // Rename .part to final filename
    tokio::fs::rename(&temp, &dest)
        .await
        .map_err(|e| format!("Rename error: {e}"))?;

    eprintln!("Download complete: {}", dest.display());
    Ok(dest)
}

/// Delete a model file by filename.
pub fn delete_model(filename: &str) -> Result<(), String> {
    let path = models_dir().join(filename);
    if path.exists() {
        std::fs::remove_file(&path).map_err(|e| format!("Delete failed: {e}"))?;
    }
    Ok(())
}
