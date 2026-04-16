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
    using_gpu: bool,
}

impl Formatter {
    /// Load a GGUF model from the given path.
    ///
    /// Tries GPU acceleration (Vulkan) by offloading all layers to GPU.
    /// Falls back to CPU-only if GPU fails. Pass `force_cpu = true` to skip GPU.
    pub fn new(model_path: &Path, force_cpu: bool) -> Result<Self, LlmError> {
        if !model_path.exists() {
            return Err(LlmError::ModelNotFound(model_path.to_path_buf()));
        }

        eprintln!("Loading LLM from: {}", model_path.display());

        let backend =
            LlamaBackend::init().map_err(|e| LlmError::LoadFailed(format!("{e}")))?;

        // Skip GPU if the user forced CPU, the guard detected a crash from a
        // previous session, OR pre-flight detection says no LLM-capable GPU
        // is present. The third check is critical: llama.cpp otherwise
        // happily accepts software renderers and underpowered integrated
        // GPUs (observed on Intel UHD — "loads" via memory overcommit then
        // hangs on first inference). `is_suitable_for_llm` is stricter than
        // the Whisper check — it requires a discrete GPU or an integrated
        // one with ≥4 GB device-local memory.
        let crash_skip = crate::gpu_guard::session_disabled();
        #[cfg(feature = "gpu")]
        let detect_result = crate::gpu_detect::detect();
        #[cfg(feature = "gpu")]
        let detect_skip = !detect_result.is_suitable_for_llm();
        #[cfg(not(feature = "gpu"))]
        let detect_skip = false;
        let skip_gpu = force_cpu || crash_skip || detect_skip;

        let (model, using_gpu) = if !skip_gpu {
            #[cfg(feature = "gpu")]
            eprintln!(
                "Attempting Vulkan GPU acceleration for LLM ({})...",
                detect_result.summary()
            );
            #[cfg(not(feature = "gpu"))]
            eprintln!("Attempting Vulkan GPU acceleration for LLM...");
            crate::gpu_guard::arm();
            let gpu_params = LlamaModelParams::default(); // n_gpu_layers defaults to -1 (all layers)

            match LlamaModel::load_from_file(&backend, model_path, &gpu_params) {
                Ok(m) => {
                    // GPU load succeeded — disarm the crash marker. The
                    // dangerous window is over. Sleep/resume invalidation
                    // is handled proactively by power_events.
                    crate::gpu_guard::disarm();
                    eprintln!("LLM loaded with GPU layer offloading");
                    (m, true)
                }
                Err(e) => {
                    // Soft failure: Vulkan is reachable but couldn't fit the
                    // model. Disarm so next session tries GPU again.
                    crate::gpu_guard::disarm();
                    eprintln!("LLM GPU loading failed: {e}");
                    eprintln!("Falling back to CPU-only for LLM...");
                    let cpu_params = LlamaModelParams::default().with_n_gpu_layers(0);
                    let m = LlamaModel::load_from_file(&backend, model_path, &cpu_params)
                        .map_err(|e| LlmError::LoadFailed(format!("{e}")))?;
                    (m, false)
                }
            }
        } else {
            if force_cpu {
                eprintln!("Force CPU mode — skipping GPU for LLM");
            } else if crash_skip {
                eprintln!("LLM: skipping GPU — previous session crashed (marker detected)");
            } else {
                // detect_skip must be true — either no GPU at all, or a GPU
                // that passes Whisper's filter but not LLM's stricter one.
                #[cfg(feature = "gpu")]
                eprintln!(
                    "LLM: skipping GPU — not suitable for LLM ({})",
                    detect_result.summary()
                );
            }
            let cpu_params = LlamaModelParams::default().with_n_gpu_layers(0);
            let m = LlamaModel::load_from_file(&backend, model_path, &cpu_params)
                .map_err(|e| LlmError::LoadFailed(format!("{e}")))?;
            (m, false)
        };

        let mut chat_format = ChatFormat::detect(&model_path.to_string_lossy());
        // Check for user override
        if let Some(filename) = model_path.file_name() {
            let filename = filename.to_string_lossy();
            if let Some(override_str) = crate::model_manager::get_chat_format_override(&filename) {
                if let Ok(fmt) = serde_json::from_value::<ChatFormat>(
                    serde_json::Value::String(override_str),
                ) {
                    chat_format = fmt;
                    eprintln!("Using user-overridden chat format: {:?}", chat_format);
                }
            }
        }
        eprintln!("LLM loaded successfully (chat format: {:?}, {})", chat_format, if using_gpu { "GPU" } else { "CPU" });
        Ok(Self { backend, model, chat_format, using_gpu })
    }

    /// The chat format this model is using (auto-detected or overridden).
    pub fn chat_format(&self) -> ChatFormat {
        self.chat_format
    }

    /// Whether the LLM is running with GPU layer offloading.
    pub fn is_using_gpu(&self) -> bool {
        self.using_gpu
    }

    /// Format raw dictated text using the given format type.
    /// If `custom_instruction` is provided, it overrides the default template.
    pub fn format_text(
        &self,
        raw_text: &str,
        format: FormatType,
    ) -> Result<String, LlmError> {
        self.format_text_custom(raw_text, format, None)
    }

    /// Format with an optional custom instruction override.
    pub fn format_text_custom(
        &self,
        raw_text: &str,
        format: FormatType,
        custom_instruction: Option<&str>,
    ) -> Result<String, LlmError> {
        if format == FormatType::None {
            return Ok(raw_text.to_string());
        }

        let prompt = format.build_prompt_custom(raw_text, self.chat_format, custom_instruction);

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

        // Strip meta-commentary that small models append despite instructions.
        let result = strip_model_commentary(output.trim(), format);

        eprintln!(
            "LLM formatted {} chars -> {} chars ({:?})",
            raw_text.len(),
            result.len(),
            format
        );

        Ok(result)
    }
}

/// Strip meta-commentary that small LLMs often append to formatted text.
///
/// Handles several patterns:
/// 1. Trailing parenthetical notes: `(Note: I rephrased …)`
/// 2. Leading parenthetical notes: `(The following has been …)\n`
/// 3. Trailing bullet-point changelogs: `\n- Corrected capitalization\n- Removed filler …`
///    — skipped for formats that legitimately produce bullets (BulletSummary, MeetingNotes, Documentation)
/// 4. Trailing "Note:" / "Changes:" blocks
fn strip_model_commentary(text: &str, format: FormatType) -> String {
    let mut result = text.to_string();

    // ── Pass 1: strip trailing parenthetical ─────────────────────────
    // e.g. "(Note: I rephrased...)" at the end
    let trimmed = result.trim_end();
    if let Some(pos) = trimmed.rfind("\n(") {
        let after = &trimmed[pos + 1..];
        if after.ends_with(')') || after.ends_with(").") {
            result = trimmed[..pos].trim_end().to_string();
        }
    }

    // ── Pass 2: strip leading parenthetical ──────────────────────────
    // e.g. "(Here is the corrected text.)\n" at the start
    let trimmed = result.trim_start();
    if trimmed.starts_with('(') {
        if let Some(pos) = trimmed.find(")\n") {
            result = trimmed[pos + 2..].trim_start().to_string();
        }
    }

    // ── Pass 3: strip leading "Here is …:" preamble ─────────────────
    let lower_start = result.trim_start().to_lowercase();
    if lower_start.starts_with("here is")
        || lower_start.starts_with("here's")
        || lower_start.starts_with("below is")
        || lower_start.starts_with("the corrected")
        || lower_start.starts_with("the rewritten")
    {
        if let Some(pos) = result.find('\n') {
            let first_line = result[..pos].to_lowercase();
            if first_line.ends_with(':') || first_line.ends_with("text:") {
                result = result[pos + 1..].trim_start().to_string();
            }
        }
    }

    // ── Pass 4: strip trailing bullet-point changelog ────────────────
    // e.g. "- Corrected capitalization\n- Removed filler words"
    // Skip for formats whose real output IS bullet points.
    let bullets_are_content = matches!(
        format,
        FormatType::BulletSummary | FormatType::MeetingNotes | FormatType::Documentation
    );
    let lines: Vec<&str> = result.lines().collect();
    if lines.len() > 1 && !bullets_are_content {
        // Walk backwards to find consecutive bullet lines
        let mut bullet_start = lines.len();
        for i in (0..lines.len()).rev() {
            let line = lines[i].trim();
            if line.starts_with("- ") || line.starts_with("* ") {
                bullet_start = i;
            } else if line.is_empty() {
                // Allow blank lines within the bullet block
                if bullet_start < lines.len() {
                    continue;
                }
                break;
            } else {
                break;
            }
        }

        if bullet_start < lines.len() {
            let bullet_text: String = lines[bullet_start..].join(" ").to_lowercase();
            const META: &[&str] = &[
                "corrected", "removed", "added", "fixed", "changed", "kept",
                "rephrased", "capitali", "filler", "punctuation", "grammar",
                "tone", "meaning", "restructur", "reword", "original text",
            ];
            if META.iter().any(|kw| bullet_text.contains(kw)) {
                let mut end = bullet_start;
                while end > 0 && lines[end - 1].trim().is_empty() {
                    end -= 1;
                }
                if end > 0 {
                    result = lines[..end].join("\n");
                }
            }
        }
    }

    // ── Pass 5: strip trailing "Note:" / "Changes:" block ────────────
    let lines: Vec<&str> = result.lines().collect();
    if lines.len() > 1 {
        for i in (1..lines.len()).rev() {
            let lower = lines[i].trim().to_lowercase();
            if lower.starts_with("note:")
                || lower.starts_with("notes:")
                || lower.starts_with("changes:")
                || lower.starts_with("changes made:")
            {
                let mut end = i;
                while end > 0 && lines[end - 1].trim().is_empty() {
                    end -= 1;
                }
                if end > 0 {
                    result = lines[..end].join("\n");
                }
                break;
            }
        }
    }

    result.trim().to_string()
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
