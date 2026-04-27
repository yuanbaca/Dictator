//! Wake-word detection via OpenWakeWord (David Scripka,
//! https://github.com/dscripka/openWakeWord) — the official pre-trained
//! ONNX pipeline.
//!
//! # Why this engine
//!
//! We tried two earlier approaches:
//! 1. **Whisper-as-wake-word** — transcribe every utterance, substring-
//!    match the result. Worked, but burned ~50× the CPU/GPU it needed
//!    to and lagged a full second behind the user finishing the phrase.
//! 2. **Rustpotter** — pure-Rust MFCC + DTW against user-recorded
//!    samples. No external runtime, but a custom MFCC reference built
//!    from a handful of samples in a real room has too high a noise
//!    floor: detections fired on quiet ambient and on the post-recording
//!    paste/keyboard artifacts in tight loops. We could not tune past it.
//!
//! OpenWakeWord ships pre-trained ONNX models for a fixed set of phrases
//! (`alexa`, `hey_jarvis`, `hey_mycroft`, `hey_rhasspy`) trained on much
//! larger speech corpora than any user could provide. Trade-off: only
//! those four phrases — no custom wake words for now.
//!
//! # Pipeline
//!
//! Per 80 ms tick (1280 samples at 16 kHz):
//!
//! 1. Take the most recent **1760** samples (1280 new + 480 lookback for
//!    hop overlap) and run them through `melspectrogram.onnx`. Output:
//!    8 new mel frames of 32 bins each.
//! 2. Apply the post-transform `x = x/10 + 2` to every mel value
//!    (the upstream Python applies it after the ONNX call; the
//!    embedding model expects it).
//! 3. Append to a rolling buffer of mel frames (init filled with ones).
//! 4. Take the last **76** mel frames and run `embedding_model.onnx`,
//!    producing one 96-dim embedding.
//! 5. Append to a rolling embedding buffer.
//! 6. Take the last **N** embeddings (N read from the keyword model's
//!    input shape — typically 16) and run the keyword classifier.
//!    Output is a sigmoided score in `[0, 1]`.
//! 7. The first 5 predictions per session are forcibly zeroed
//!    (suppresses initialization artifacts).
//!
//! # Threshold
//!
//! No internal smoothing — each call is independent. 0.5 is the
//! upstream default; community tuning targets 0.6–0.7 for fewer false
//! positives.

use anyhow::{anyhow, Context, Result};
use ndarray::{Array2, Array3, Array4};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use std::collections::VecDeque;

/// One bundled keyword model. Names map to the official OpenWakeWord
/// release filenames (with `_v0.1.onnx` stripped for the user-visible
/// id and label). 4 official keyword models from v0.5.1 + 1 community-
/// trained one that's pipeline-compatible.
///
/// Note: tried bundling `okay_nabu` from the rhasspy ecosystem but the
/// only readily available distribution there is TFLite, not ONNX —
/// the URL we tried at github.com/rhasspy/models served a 404 HTML
/// page. Re-add when a verified ONNX export surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WakePhrase {
    /// Official: "alexa"
    Alexa,
    /// Official: "hey jarvis"
    HeyJarvis,
    /// Official: "hey mycroft"
    HeyMycroft,
    /// Official: "hey rhasspy"
    HeyRhasspy,
    /// Community: GLaDOS-themed "hey glados" (huggingface).
    HeyGlados,
}

impl WakePhrase {
    /// Stable string id used in localStorage, settings, and on the wire.
    pub fn id(&self) -> &'static str {
        match self {
            Self::Alexa => "alexa",
            Self::HeyJarvis => "hey_jarvis",
            Self::HeyMycroft => "hey_mycroft",
            Self::HeyRhasspy => "hey_rhasspy",
            Self::HeyGlados => "hey_glados",
        }
    }

    /// Human-readable label shown in the UI.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Alexa => "Alexa",
            Self::HeyJarvis => "Hey Jarvis",
            Self::HeyMycroft => "Hey Mycroft",
            Self::HeyRhasspy => "Hey Rhasspy",
            Self::HeyGlados => "Hey GLaDOS (community)",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        Some(match id {
            "alexa" => Self::Alexa,
            "hey_jarvis" => Self::HeyJarvis,
            "hey_mycroft" => Self::HeyMycroft,
            "hey_rhasspy" => Self::HeyRhasspy,
            "hey_glados" => Self::HeyGlados,
            _ => return None,
        })
    }

    pub fn all() -> [Self; 5] {
        [
            Self::Alexa,
            Self::HeyJarvis,
            Self::HeyMycroft,
            Self::HeyRhasspy,
            Self::HeyGlados,
        ]
    }

    fn model_bytes(&self) -> &'static [u8] {
        match self {
            Self::Alexa => ALEXA_MODEL,
            Self::HeyJarvis => HEY_JARVIS_MODEL,
            Self::HeyMycroft => HEY_MYCROFT_MODEL,
            Self::HeyRhasspy => HEY_RHASSPY_MODEL,
            Self::HeyGlados => HEY_GLADOS_MODEL,
        }
    }
}

// ── Bundled ONNX models ────────────────────────────────────────────────
//
// Total ~5.6 MB added to the binary. ort can build a Session from an
// in-memory byte slice via `commit_from_memory`, so no disk extraction
// is needed — models live in the exe and load straight to ort.

const MELSPEC_MODEL: &[u8] =
    include_bytes!("../resources/openwakeword/melspectrogram.onnx");
const EMBEDDING_MODEL: &[u8] =
    include_bytes!("../resources/openwakeword/embedding_model.onnx");
const ALEXA_MODEL: &[u8] =
    include_bytes!("../resources/openwakeword/alexa_v0.1.onnx");
const HEY_JARVIS_MODEL: &[u8] =
    include_bytes!("../resources/openwakeword/hey_jarvis_v0.1.onnx");
const HEY_MYCROFT_MODEL: &[u8] =
    include_bytes!("../resources/openwakeword/hey_mycroft_v0.1.onnx");
const HEY_RHASSPY_MODEL: &[u8] =
    include_bytes!("../resources/openwakeword/hey_rhasspy_v0.1.onnx");
const HEY_GLADOS_MODEL: &[u8] =
    include_bytes!("../resources/openwakeword/hey_glados.onnx");

// ── Pipeline constants ─────────────────────────────────────────────────

/// New audio per tick — 80 ms at 16 kHz.
const STEP_SAMPLES: usize = 1280;
/// Total samples fed to the mel model per tick. 1280 + 480-sample
/// lookback preserves hop-window overlap at the buffer boundary.
const MEL_INPUT_SAMPLES: usize = 1760;
/// Mel frames consumed per embedding inference.
const MEL_WINDOW: usize = 76;
/// Mel bins per frame.
const MEL_BINS: usize = 32;
/// Cap on the rolling mel buffer (~10 s at 97 frames/s).
const MEL_BUFFER_MAX: usize = 970;
/// Embedding dimensionality.
const EMBEDDING_DIM: usize = 96;
/// Cap on the rolling embedding buffer (~10 s).
const EMBEDDING_BUFFER_MAX: usize = 120;
/// Suppress this many predictions at session start — OpenWakeWord
/// zeros them upstream because freshly-init'd buffers produce noisy
/// scores for the first few inferences.
const INIT_SUPPRESS_PREDICTIONS: u32 = 5;

/// Streaming detector. Push audio in any chunk size; pull scores out.
pub struct WakeWordDetector {
    melspec_session: Session,
    embedding_session: Session,
    keyword_session: Session,

    /// Number of embeddings the loaded keyword model expects per
    /// inference (read from its input shape — usually 16, but the
    /// official `hey_rhasspy` model uses a different value).
    keyword_n: usize,

    /// Raw audio waiting to be fed to the mel spectrogram model. We
    /// drain `STEP_SAMPLES` per tick but keep the trailing 480 samples
    /// around for the next tick's lookback.
    audio_buffer: VecDeque<f32>,

    /// Rolling mel-frame buffer. Init filled with ones — zero-init
    /// produces weird first inferences in the embedding model.
    mel_buffer: VecDeque<[f32; MEL_BINS]>,

    /// Rolling embedding buffer. Pre-primed with zeros so the first
    /// real keyword inference doesn't see undefined memory.
    embedding_buffer: VecDeque<[f32; EMBEDDING_DIM]>,

    /// How many keyword predictions we've produced. Used to suppress
    /// the first few (per OpenWakeWord's first-5-zero guard).
    predictions_made: u32,
}

impl WakeWordDetector {
    /// Build a detector for the requested phrase.
    pub fn new(phrase: WakePhrase) -> Result<Self> {
        let melspec_session = Session::builder()
            .context("ort: SessionBuilder::new for melspec")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .context("ort: optimization level for melspec")?
            .commit_from_memory(MELSPEC_MODEL)
            .context("ort: commit melspec model")?;

        let embedding_session = Session::builder()
            .context("ort: SessionBuilder::new for embedding")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .context("ort: optimization level for embedding")?
            .commit_from_memory(EMBEDDING_MODEL)
            .context("ort: commit embedding model")?;

        let keyword_session = Session::builder()
            .context("ort: SessionBuilder::new for keyword")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .context("ort: optimization level for keyword")?
            .commit_from_memory(phrase.model_bytes())
            .context("ort: commit keyword model")?;

        // Read the keyword model's expected number of embeddings from its
        // input shape. The Python upstream uses
        //     model.get_inputs()[0].shape[1]
        // which is what we replicate here. Falls back to 16 if for some
        // reason the dim is dynamic / unreadable.
        let keyword_n = keyword_session
            .inputs
            .first()
            .and_then(|i| {
                use ort::value::ValueType;
                if let ValueType::Tensor { shape, .. } = &i.input_type {
                    shape.get(1).copied()
                } else {
                    None
                }
            })
            .filter(|&n| n > 0)
            .map(|n| n as usize)
            .unwrap_or(16);

        let mut mel_buffer: VecDeque<[f32; MEL_BINS]> =
            VecDeque::with_capacity(MEL_BUFFER_MAX);
        for _ in 0..MEL_WINDOW {
            mel_buffer.push_back([1.0; MEL_BINS]);
        }

        let mut embedding_buffer: VecDeque<[f32; EMBEDDING_DIM]> =
            VecDeque::with_capacity(EMBEDDING_BUFFER_MAX);
        for _ in 0..keyword_n {
            embedding_buffer.push_back([0.0; EMBEDDING_DIM]);
        }

        Ok(Self {
            melspec_session,
            embedding_session,
            keyword_session,
            keyword_n,
            audio_buffer: VecDeque::with_capacity(MEL_INPUT_SAMPLES * 2),
            mel_buffer,
            embedding_buffer,
            predictions_made: 0,
        })
    }

    /// Push f32 audio at 16 kHz mono. Returns the most recent keyword
    /// score that was produced (or `None` if no full tick happened yet).
    /// Scores during the suppression window are reported as 0.0.
    pub fn push_samples(&mut self, samples: &[f32]) -> Result<Option<f32>> {
        self.audio_buffer.extend(samples.iter().copied());

        let mut last_score: Option<f32> = None;
        // Keep ticking as long as we have a full step plus the lookback
        // window worth of samples.
        while self.audio_buffer.len() >= MEL_INPUT_SAMPLES {
            let chunk: Vec<f32> = self
                .audio_buffer
                .iter()
                .take(MEL_INPUT_SAMPLES)
                .copied()
                .collect();
            // Drop only `STEP_SAMPLES` — the trailing 480 stay for next tick.
            self.audio_buffer.drain(..STEP_SAMPLES);

            let mel_frames = self.run_melspec(&chunk)?;
            for frame in mel_frames {
                self.mel_buffer.push_back(frame);
                if self.mel_buffer.len() > MEL_BUFFER_MAX {
                    self.mel_buffer.pop_front();
                }
            }

            if self.mel_buffer.len() < MEL_WINDOW {
                continue;
            }
            let embedding = self.run_embedding()?;
            self.embedding_buffer.push_back(embedding);
            if self.embedding_buffer.len() > EMBEDDING_BUFFER_MAX {
                self.embedding_buffer.pop_front();
            }

            if self.embedding_buffer.len() < self.keyword_n {
                continue;
            }
            let raw_score = self.run_keyword()?;
            self.predictions_made = self.predictions_made.saturating_add(1);
            let score = if self.predictions_made <= INIT_SUPPRESS_PREDICTIONS {
                0.0
            } else {
                raw_score
            };
            last_score = Some(score);
        }

        Ok(last_score)
    }

    fn run_melspec(&mut self, samples: &[f32]) -> Result<Vec<[f32; MEL_BINS]>> {
        let input = Array2::from_shape_vec((1, samples.len()), samples.to_vec())
            .context("melspec: build input array")?;
        let tensor = Tensor::from_array(input).context("melspec: build tensor")?;
        let input_name = self.melspec_session.inputs[0].name.clone();
        // Capture the output name *before* taking the &mut for `run` —
        // ort::Session has both `inputs` and `outputs` fields and run
        // mutably borrows through `&mut self`, so any later read of
        // `.outputs` collides with the borrow checker.
        let output_name = self.melspec_session.outputs[0].name.clone();
        let outputs = self
            .melspec_session
            .run(ort::inputs![input_name => tensor])
            .context("melspec: run inference")?;
        let (shape, data) = outputs[output_name.as_str()]
            .try_extract_tensor::<f32>()
            .context("melspec: extract output")?;
        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        if dims.len() != 4 || dims[3] != MEL_BINS {
            return Err(anyhow!(
                "melspec: unexpected output shape {:?} (want [1,1,n,{MEL_BINS}])",
                dims
            ));
        }
        let n_frames = dims[2];
        // Apply the upstream-mandatory mel transform `x/10 + 2` while
        // copying out, so downstream stages see the form the embedding
        // model was trained against.
        let mut frames: Vec<[f32; MEL_BINS]> = Vec::with_capacity(n_frames);
        for f in 0..n_frames {
            let mut frame = [0.0f32; MEL_BINS];
            for b in 0..MEL_BINS {
                let idx = f * MEL_BINS + b;
                frame[b] = data[idx] / 10.0 + 2.0;
            }
            frames.push(frame);
        }
        Ok(frames)
    }

    fn run_embedding(&mut self) -> Result<[f32; EMBEDDING_DIM]> {
        // Last 76 mel frames flattened into a [1, 76, 32, 1] tensor.
        let mut buf: Vec<f32> = Vec::with_capacity(MEL_WINDOW * MEL_BINS);
        let len = self.mel_buffer.len();
        for frame in self.mel_buffer.iter().skip(len - MEL_WINDOW) {
            buf.extend_from_slice(frame);
        }
        let input = Array4::from_shape_vec((1, MEL_WINDOW, MEL_BINS, 1), buf)
            .context("embedding: build input array")?;
        let tensor = Tensor::from_array(input).context("embedding: build tensor")?;
        let input_name = self.embedding_session.inputs[0].name.clone();
        let output_name = self.embedding_session.outputs[0].name.clone();
        let outputs = self
            .embedding_session
            .run(ort::inputs![input_name => tensor])
            .context("embedding: run inference")?;
        let (_shape, data) = outputs[output_name.as_str()]
            .try_extract_tensor::<f32>()
            .context("embedding: extract output")?;
        // Output is [1, 1, 1, 96] (or any shape that flattens to 96).
        if data.len() != EMBEDDING_DIM {
            return Err(anyhow!(
                "embedding: expected {EMBEDDING_DIM} values, got {}",
                data.len()
            ));
        }
        let mut out = [0.0f32; EMBEDDING_DIM];
        out.copy_from_slice(data);
        Ok(out)
    }

    fn run_keyword(&mut self) -> Result<f32> {
        let mut buf: Vec<f32> =
            Vec::with_capacity(self.keyword_n * EMBEDDING_DIM);
        let len = self.embedding_buffer.len();
        for emb in self.embedding_buffer.iter().skip(len - self.keyword_n) {
            buf.extend_from_slice(emb);
        }
        let input = Array3::from_shape_vec((1, self.keyword_n, EMBEDDING_DIM), buf)
            .context("keyword: build input array")?;
        let tensor = Tensor::from_array(input).context("keyword: build tensor")?;
        let input_name = self.keyword_session.inputs[0].name.clone();
        let output_name = self.keyword_session.outputs[0].name.clone();
        let outputs = self
            .keyword_session
            .run(ort::inputs![input_name => tensor])
            .context("keyword: run inference")?;
        let (_shape, data) = outputs[output_name.as_str()]
            .try_extract_tensor::<f32>()
            .context("keyword: extract output")?;
        data.first()
            .copied()
            .ok_or_else(|| anyhow!("keyword: empty output tensor"))
    }
}
