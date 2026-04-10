use crate::templates::{ChatFormat, FormatType};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;
use std::num::NonZero;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LlmError {
    #[error("Model not loaded")]
    NotLoaded,
    #[error("Model file not found: {0}")]
    ModelNotFound(PathBuf),
    #[error("Failed to load model: {0}")]
    LoadFailed(String),
    #[error("Inference failed: {0}")]
    InferenceFailed(String),
}

pub struct Formatter {
    backend: LlamaBackend,
    model: LlamaModel,
    chat_format: ChatFormat,
}

impl Formatter {
    /// Load a GGUF model from the given path.
    pub fn new(model_path: &Path) -> Result<Self, LlmError> {
        if !model_path.exists() {
            return Err(LlmError::ModelNotFound(model_path.to_path_buf()));
        }

        eprintln!("Loading LLM from: {}", model_path.display());

        let backend =
            LlamaBackend::init().map_err(|e| LlmError::LoadFailed(format!("{e}")))?;

        let model_params = LlamaModelParams::default();

        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .map_err(|e| LlmError::LoadFailed(format!("{e}")))?;

        let chat_format = ChatFormat::detect(&model_path.to_string_lossy());
        eprintln!("LLM loaded successfully (chat format: {:?})", chat_format);
        Ok(Self { backend, model, chat_format })
    }

    /// Format raw dictated text using the given format type.
    /// Returns the formatted text.
    pub fn format_text(&self, raw_text: &str, format: FormatType) -> Result<String, LlmError> {
        if format == FormatType::None {
            return Ok(raw_text.to_string());
        }

        let prompt = format.build_prompt(raw_text, self.chat_format);

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZero::new(2048));

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| LlmError::InferenceFailed(format!("Context creation failed: {e}")))?;

        let tokens = self
            .model
            .str_to_token(&prompt, llama_cpp_2::model::AddBos::Always)
            .map_err(|e| LlmError::InferenceFailed(format!("Tokenization failed: {e}")))?;

        let mut batch = LlamaBatch::new(2048, 1);

        for (i, token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch
                .add(*token, i as i32, &[0], is_last)
                .map_err(|e| LlmError::InferenceFailed(format!("Batch add failed: {e}")))?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| LlmError::InferenceFailed(format!("Prompt decode failed: {e}")))?;

        // Sample output tokens
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(0.3), // low temp for formatting — we want consistency
            LlamaSampler::dist(42),
        ]);

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output = String::new();
        let mut n_cur = tokens.len() as i32;
        let max_tokens = 1024;

        for _ in 0..max_tokens {
            let token = sampler.sample(&ctx, -1);

            if self.model.is_eog_token(token) {
                break;
            }

            let piece = self
                .model
                .token_to_piece(token, &mut decoder, false, None)
                .map_err(|e| LlmError::InferenceFailed(format!("Token decode failed: {e}")))?;
            output.push_str(&piece);

            batch.clear();
            batch
                .add(token, n_cur, &[0], true)
                .map_err(|e| LlmError::InferenceFailed(format!("Batch add failed: {e}")))?;
            ctx.decode(&mut batch)
                .map_err(|e| LlmError::InferenceFailed(format!("Decode failed: {e}")))?;
            n_cur += 1;
        }

        // Strip any trailing parenthetical commentary the model may add
        // e.g. "(Note: I rephrased...)" or "(The above has been...)"
        let trimmed = output.trim();
        let result = if let Some(pos) = trimmed.rfind("\n(") {
            let after = &trimmed[pos + 1..];
            if after.ends_with(')') || after.ends_with(").") {
                trimmed[..pos].trim_end().to_string()
            } else {
                trimmed.to_string()
            }
        } else if trimmed.starts_with('(') && trimmed.contains(")\n") {
            // Leading parenthetical note before the actual content
            if let Some(pos) = trimmed.find(")\n") {
                trimmed[pos + 2..].trim_start().to_string()
            } else {
                trimmed.to_string()
            }
        } else {
            trimmed.to_string()
        };

        eprintln!(
            "LLM formatted {} chars -> {} chars ({:?})",
            raw_text.len(),
            result.len(),
            format
        );

        Ok(result)
    }
}

/// Search for an LLM model file, similar to whisper model search.
/// Looks for any .gguf file in the models/ directory.
pub fn find_llm_model_path() -> Option<PathBuf> {
    let candidates = build_search_paths();

    for dir in &candidates {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "gguf") {
                    eprintln!("Found LLM model at: {}", path.display());
                    return Some(path);
                }
            }
        }
    }

    None
}

fn build_search_paths() -> Vec<PathBuf> {
    let mut paths = vec![
        PathBuf::from("../../models"),
        PathBuf::from("models"),
        PathBuf::from("../models"),
    ];

    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            paths.push(exe_dir.join("../../../../models"));
            paths.push(exe_dir.join("../../models"));
            paths.push(exe_dir.join("../models"));
            paths.push(exe_dir.join("models"));
        }
    }

    paths
}
