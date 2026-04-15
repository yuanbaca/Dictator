//! Local transcription using whisper.cpp via whisper-rs.

use anyhow::{Context, Result};
use std::path::Path;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Result of a transcription job.
#[derive(Debug)]
pub struct TranscriptResult {
    pub text: String,
    pub segments: Vec<TranscriptSegment>,
}

/// A single timed segment from the transcription.
#[derive(Debug)]
pub struct TranscriptSegment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
}

/// Holds a loaded whisper model, ready to transcribe.
pub struct Transcriber {
    ctx: WhisperContext,
    using_gpu: bool,
}

impl Transcriber {
    /// Load a whisper model from a .bin file.
    ///
    /// Tries GPU acceleration (Vulkan) first and falls back to CPU if the GPU
    /// isn't available or fails to initialize. Pass `force_cpu = true` to skip
    /// the GPU attempt entirely.
    pub fn new(model_path: &Path, force_cpu: bool) -> Result<Self> {
        let path_str = model_path.to_str().context("Invalid model path")?;

        // Skip GPU if the user forced CPU, the guard detected a crash from a
        // previous session, OR pre-flight detection says no usable GPU is
        // present. The third check is critical for GPU-less hardware —
        // whisper.cpp otherwise happily accepts software renderers (Microsoft
        // Basic Render Driver etc.) and hangs during actual inference.
        let crash_skip = crate::gpu_guard::session_disabled();
        #[cfg(feature = "gpu")]
        let detect_result = crate::gpu_detect::detect();
        #[cfg(feature = "gpu")]
        let detect_skip = !detect_result.is_usable();
        #[cfg(not(feature = "gpu"))]
        let detect_skip = false;
        let skip_gpu = force_cpu || crash_skip || detect_skip;

        if !skip_gpu {
            #[cfg(feature = "gpu")]
            eprintln!(
                "Attempting Vulkan GPU acceleration for Whisper ({})...",
                detect_result.summary()
            );
            #[cfg(not(feature = "gpu"))]
            eprintln!("Attempting Vulkan GPU acceleration for Whisper...");
            crate::gpu_guard::arm();
            let mut gpu_params = WhisperContextParameters::default();
            gpu_params.use_gpu(true);

            match WhisperContext::new_with_params(path_str, gpu_params) {
                Ok(ctx) => {
                    // Leave the marker armed — only disarm on graceful shutdown.
                    // A crash during inference will leave it in place, which is
                    // what we want: next session will skip GPU and recover.
                    eprintln!("Whisper model loaded with GPU acceleration (Vulkan)");
                    let transcriber = Self {
                        ctx,
                        using_gpu: true,
                    };
                    transcriber.warm_up();
                    return Ok(transcriber);
                }
                Err(e) => {
                    // Soft failure: Rust got an Err back, meaning Vulkan is
                    // reachable but something else went wrong (e.g. model too
                    // big for VRAM). Disarm so the next session tries GPU fresh.
                    crate::gpu_guard::disarm();
                    eprintln!("Whisper GPU initialization failed: {e}");
                    eprintln!("Falling back to CPU...");
                }
            }
        } else if force_cpu {
            eprintln!("Force CPU mode — skipping GPU for Whisper");
        } else if crash_skip {
            eprintln!("Whisper: skipping GPU — previous session crashed (marker detected)");
        } else {
            // detect_skip must be true — log the reason
            #[cfg(feature = "gpu")]
            eprintln!(
                "Whisper: skipping GPU — no usable GPU detected ({})",
                detect_result.summary()
            );
        }

        // CPU fallback (always works)
        let mut cpu_params = WhisperContextParameters::default();
        cpu_params.use_gpu(false);

        let ctx = WhisperContext::new_with_params(path_str, cpu_params)
            .map_err(|e| anyhow::anyhow!("Failed to load whisper model: {e}"))?;

        eprintln!("Whisper model loaded (CPU mode)");
        let transcriber = Self {
            ctx,
            using_gpu: false,
        };
        transcriber.warm_up();
        Ok(transcriber)
    }

    /// Whether the model is running on GPU.
    pub fn is_using_gpu(&self) -> bool {
        self.using_gpu
    }

    /// Run one tiny silent inference to pre-warm kernels, caches, and any
    /// JIT-compiled shaders. The first real call after model load is otherwise
    /// noticeably slow — especially on GPU, where Vulkan has to compile
    /// pipelines on first use. Best-effort: errors are logged but don't
    /// propagate, since a failed warm-up doesn't break normal use.
    fn warm_up(&self) {
        let silence = vec![0.0f32; 16000]; // 1 second of 16kHz silence
        let start = std::time::Instant::now();
        match self.transcribe(&silence) {
            Ok(_) => eprintln!(
                "Whisper warm-up complete ({:.2}s)",
                start.elapsed().as_secs_f32()
            ),
            Err(e) => eprintln!("Whisper warm-up failed (non-fatal): {e}"),
        }
    }

    /// Transcribe audio samples.
    ///
    /// `samples` must be mono f32 audio at 16kHz sample rate.
    pub fn transcribe(&self, samples: &[f32]) -> Result<TranscriptResult> {
        self.transcribe_with_progress(samples, |_| {})
    }

    /// Transcribe audio samples with a progress callback (0-100%).
    pub fn transcribe_with_progress<F>(
        &self,
        samples: &[f32],
        on_progress: F,
    ) -> Result<TranscriptResult>
    where
        F: FnMut(i32) + 'static,
    {
        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| anyhow::anyhow!("Failed to create whisper state: {e}"))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        // English language
        params.set_language(Some("en"));

        // Decode each call independently — don't carry the prior utterance's
        // tokens into this one as context. Equivalent to faster-whisper's
        // `condition_on_previous_text=False`. Without this, Whisper's KV state
        // can bleed between back-to-back calls and cause hallucinated repeats.
        // Important now (correctness for press-to-talk), essential for live
        // mode where we call transcribe once per VAD-gated segment.
        params.set_no_context(true);

        // Enable token-level timestamps
        params.set_token_timestamps(true);

        // Progress callback
        params.set_progress_callback_safe(on_progress);

        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_print_special(false);

        // Use 4 threads on CPU; GPU handles parallelism internally
        params.set_n_threads(if self.using_gpu { 1 } else { 4 });

        state
            .full(params, samples)
            .map_err(|e| anyhow::anyhow!("Transcription failed: {e}"))?;

        let num_segments = state.full_n_segments();

        let mut full_text = String::new();
        let mut segments = Vec::new();

        for i in 0..num_segments {
            let seg = state
                .get_segment(i)
                .ok_or_else(|| anyhow::anyhow!("Failed to get segment {i}"))?;

            let text = seg
                .to_str_lossy()
                .map_err(|e| anyhow::anyhow!("Failed to get segment {i} text: {e}"))?
                .to_string();
            let start = seg.start_timestamp();
            let end = seg.end_timestamp();

            // Skip hallucinated segments (e.g. [BLANK_AUDIO]) so they don't
            // get appended to real speech
            if !is_whisper_hallucination(&text) {
                full_text.push_str(&text);
            }

            segments.push(TranscriptSegment {
                start_ms: start * 10, // whisper timestamps are in 10ms units
                end_ms: end * 10,
                text,
            });
        }

        let text = full_text.trim().to_string();

        Ok(TranscriptResult {
            text: if is_whisper_hallucination(&text) {
                String::new()
            } else {
                text
            },
            segments,
        })
    }
}

/// Whisper outputs these patterns when it hears silence or noise instead of
/// speech.  We treat them as "no speech" rather than real transcription.
fn is_whisper_hallucination(text: &str) -> bool {
    let lower = text.to_lowercase();
    let trimmed = lower.trim().trim_matches(|c: char| c == '.' || c == '!' || c == ' ');

    // Bracketed/parenthesized tags Whisper emits for non-speech
    if trimmed.starts_with('[') && trimmed.ends_with(']') {
        return true; // [BLANK_AUDIO], [silence], [music], etc.
    }
    if trimmed.starts_with('(') && trimmed.ends_with(')') {
        return true; // (blank audio), (silence), (music), etc.
    }

    // Common hallucinated phrases from Whisper's YouTube subtitle training data
    let hallucinations = [
        "thank you",
        "thanks for watching",
        "thank you for watching",
        "subscribe",
        "like and subscribe",
        "subtitles by",
        "translated by",
        "amara.org",
        "www.mooji.org",
        "you",
    ];

    hallucinations.iter().any(|h| trimmed == *h)
}

/// Load a WAV file and convert to mono f32 samples at 16kHz.
///
/// If the WAV is not 16kHz, this will return an error for now.
/// A proper implementation would resample.
pub fn load_wav_as_samples(path: &Path) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("Failed to open WAV file: {}", path.display()))?;

    let spec = reader.spec();
    println!(
        "WAV: {} channels, {} Hz, {:?} format, {} bits",
        spec.channels, spec.sample_rate, spec.sample_format, spec.bits_per_sample
    );

    if spec.sample_rate != 16000 {
        anyhow::bail!(
            "WAV sample rate is {} Hz, but whisper requires 16000 Hz. \
             Please convert your audio to 16kHz mono WAV.",
            spec.sample_rate
        );
    }

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1u32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_val))
                .collect::<Result<Vec<_>, _>>()
                .context("Failed to read WAV samples")?
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .context("Failed to read WAV samples")?,
    };

    // Convert stereo to mono by averaging channels
    if spec.channels == 2 {
        let mono: Vec<f32> = samples
            .chunks(2)
            .map(|pair| (pair[0] + pair[1]) / 2.0)
            .collect();
        Ok(mono)
    } else {
        Ok(samples)
    }
}
