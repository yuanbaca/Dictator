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

/// Get the models directory path, creating it if needed.
pub fn models_dir() -> PathBuf {
    // Try exe-relative first (for deployed builds)
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let candidates = [
                exe_dir.join("../../../../models"),
                exe_dir.join("../../models"),
                exe_dir.join("models"),
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

    // Default: create models/ relative to CWD
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
