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
}

impl Transcriber {
    /// Load a whisper model from a .bin file.
    pub fn new(model_path: &Path) -> Result<Self> {
        let ctx = WhisperContext::new_with_params(
            model_path.to_str().context("Invalid model path")?,
            WhisperContextParameters::default(),
        )
        .map_err(|e| anyhow::anyhow!("Failed to load whisper model: {e}"))?;

        Ok(Self { ctx })
    }

    /// Transcribe audio samples.
    ///
    /// `samples` must be mono f32 audio at 16kHz sample rate.
    pub fn transcribe(&self, samples: &[f32]) -> Result<TranscriptResult> {
        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| anyhow::anyhow!("Failed to create whisper state: {e}"))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        // English language, auto-detect if needed
        params.set_language(Some("en"));

        // Enable token-level timestamps
        params.set_token_timestamps(true);

        // Print progress to stderr for debugging
        params.set_print_progress(true);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_print_special(false);

        // Single-threaded for simplicity in Phase 0
        params.set_n_threads(4);

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

            let text = seg.to_str_lossy()
                .map_err(|e| anyhow::anyhow!("Failed to get segment {i} text: {e}"))?
                .to_string();
            let start = seg.start_timestamp();
            let end = seg.end_timestamp();

            full_text.push_str(&text);

            segments.push(TranscriptSegment {
                start_ms: start * 10, // whisper timestamps are in 10ms units
                end_ms: end * 10,
                text,
            });
        }

        Ok(TranscriptResult {
            text: full_text.trim().to_string(),
            segments,
        })
    }
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
        let mono: Vec<f32> = samples.chunks(2).map(|pair| (pair[0] + pair[1]) / 2.0).collect();
        Ok(mono)
    } else {
        Ok(samples)
    }
}
