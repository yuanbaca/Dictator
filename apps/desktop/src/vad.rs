//! Voice activity detection for live-mode endpointing.
//!
//! Wraps Silero VAD (via the `voice_activity_detector` crate) behind a
//! push-based API: feed mono f32 samples at 16 kHz in any chunk size, drain
//! `VadEvent`s describing speech boundaries. One utterance = one `Start`
//! followed by one `End` carrying the buffered audio.
//!
//! The detector runs on 512-sample windows (32 ms) internally — that's what
//! Silero V5 was trained on. We buffer whatever the caller pushes, split into
//! 512-sample chunks, and decide per-chunk whether we're in speech.
//!
//! Endpointing rule: once speech starts, we wait for `MIN_SILENCE_CHUNKS`
//! consecutive non-speech chunks before emitting `End`. That's the knob you
//! trade off comfort-of-pausing against responsiveness.
//!
//! A small ring buffer (`LOOKBACK_CHUNKS`) of pre-Start audio is prepended to
//! each utterance so the first phoneme isn't clipped — Silero has ~100 ms of
//! latency before it confidently flags speech, and the caller would hear the
//! user cut off without this.

use anyhow::{Context, Result};
use voice_activity_detector::VoiceActivityDetector;

/// Sample rate we operate at. Silero V5 supports 8 kHz and 16 kHz; our whole
/// pipeline is 16 kHz so we hardcode that.
pub const SAMPLE_RATE: usize = 16_000;

/// Silero V5 window size at 16 kHz. Changing this means retraining — don't.
const CHUNK_SAMPLES: usize = 512;

/// Speech probability above which a chunk is considered speech.
const SPEECH_THRESHOLD: f32 = 0.5;

/// Number of consecutive non-speech chunks needed to close an utterance.
/// 20 chunks * 32 ms ≈ 640 ms of silence. Compromise between the original
/// 320 ms (10 chunks, too aggressive — premature endpoints mid-thought)
/// and 900 ms (28 chunks, accurate but felt sluggish). 640 ms is long
/// enough that natural thinking pauses don't split utterances, but short
/// enough that the system still feels responsive.
const MIN_SILENCE_CHUNKS: u32 = 20;

/// Pre-Start audio kept in a ring buffer and prepended to each utterance.
/// 8 chunks * 32 ms = 256 ms of lookback. With the longer silence
/// threshold, detection latency matters more — extra lookback protects
/// the first syllable from being clipped.
const LOOKBACK_CHUNKS: usize = 8;

/// Events emitted by the VAD as it processes streaming audio.
#[derive(Debug)]
pub enum VadEvent {
    /// Speech just started. Use to update UI ("Listening...").
    Start,
    /// Speech just ended. `samples` holds the full utterance audio, ready to
    /// hand to Whisper. Includes the `LOOKBACK_CHUNKS` of pre-Start audio.
    End { samples: Vec<f32> },
}

/// Streaming VAD state machine.
pub struct Vad {
    inner: VoiceActivityDetector,
    /// Accumulator for samples pushed by the caller that haven't yet filled a
    /// 512-sample window. Drained into the detector one chunk at a time.
    pending: Vec<f32>,
    /// Ring buffer of the most recent N chunks' worth of audio, kept so the
    /// first bit of the utterance survives Silero's detection delay.
    lookback: std::collections::VecDeque<Vec<f32>>,
    /// Audio accumulated for the currently-open utterance.
    utterance: Vec<f32>,
    in_speech: bool,
    silence_chunks: u32,
    /// Events queued for the caller to drain.
    events: Vec<VadEvent>,
}

impl Vad {
    /// Build a new detector. Fails if Silero VAD can't be loaded — usually a
    /// missing `onnxruntime.dll` next to the exe.
    pub fn new() -> Result<Self> {
        let inner = VoiceActivityDetector::builder()
            .sample_rate(SAMPLE_RATE as i64)
            .chunk_size(CHUNK_SAMPLES)
            .build()
            .context("Failed to initialize Silero VAD")?;

        Ok(Self {
            inner,
            pending: Vec::with_capacity(CHUNK_SAMPLES * 2),
            lookback: std::collections::VecDeque::with_capacity(LOOKBACK_CHUNKS),
            utterance: Vec::new(),
            in_speech: false,
            silence_chunks: 0,
            events: Vec::new(),
        })
    }

    /// Push any number of samples. Events are produced internally — call
    /// `drain_events()` to collect them.
    pub fn push_samples(&mut self, samples: &[f32]) {
        self.pending.extend_from_slice(samples);

        while self.pending.len() >= CHUNK_SAMPLES {
            // Split off one 512-sample chunk. Clone rather than drain so we
            // can keep it in the lookback ring buffer without extra copies.
            let chunk: Vec<f32> = self.pending.drain(..CHUNK_SAMPLES).collect();
            self.process_chunk(chunk);
        }
    }

    fn process_chunk(&mut self, chunk: Vec<f32>) {
        let prob = self.inner.predict(chunk.iter().copied());
        let is_speech = prob >= SPEECH_THRESHOLD;

        if is_speech {
            if !self.in_speech {
                // Speech just started. Drain lookback into the utterance so
                // the attack of the first word isn't cut off, then append
                // this chunk.
                self.utterance.clear();
                for past in self.lookback.drain(..) {
                    self.utterance.extend_from_slice(&past);
                }
                self.utterance.extend_from_slice(&chunk);
                self.in_speech = true;
                self.silence_chunks = 0;
                self.events.push(VadEvent::Start);
            } else {
                self.utterance.extend_from_slice(&chunk);
                self.silence_chunks = 0;
            }
        } else if self.in_speech {
            // Still inside an open utterance, just heard silence. Keep the
            // audio (so trailing sibilants / breath don't cut off) and tick
            // the silence counter.
            self.utterance.extend_from_slice(&chunk);
            self.silence_chunks += 1;
            if self.silence_chunks >= MIN_SILENCE_CHUNKS {
                // Seed the lookback ring with the tail of this utterance
                // (which is trailing silence) so the NEXT Start has
                // pre-speech audio immediately. Without this, a quick
                // follow-up word would have an empty lookback and its
                // first syllable would be clipped.
                self.lookback.clear();
                let tail_samples = LOOKBACK_CHUNKS * CHUNK_SAMPLES;
                let start = self.utterance.len().saturating_sub(tail_samples);
                for c in self.utterance[start..].chunks(CHUNK_SAMPLES) {
                    self.lookback.push_back(c.to_vec());
                }

                let samples = std::mem::take(&mut self.utterance);
                self.in_speech = false;
                self.silence_chunks = 0;
                self.events.push(VadEvent::End { samples });
            }
        } else {
            // Not in speech and still not — feed the ring buffer so we have
            // lookback ready when speech does start.
            if self.lookback.len() == LOOKBACK_CHUNKS {
                self.lookback.pop_front();
            }
            self.lookback.push_back(chunk);
        }
    }

    /// Take all events the VAD has queued. The internal queue is cleared.
    pub fn drain_events(&mut self) -> Vec<VadEvent> {
        std::mem::take(&mut self.events)
    }

    /// Force-close any in-progress utterance (e.g. user pressed Stop).
    /// Emits an `End` event with whatever audio is currently buffered, then
    /// resets. Any pending samples below 512 are discarded.
    pub fn flush(&mut self) -> Vec<VadEvent> {
        self.pending.clear();
        if self.in_speech {
            let samples = std::mem::take(&mut self.utterance);
            self.in_speech = false;
            self.silence_chunks = 0;
            self.events.push(VadEvent::End { samples });
        }
        self.lookback.clear();
        std::mem::take(&mut self.events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Silence only → no events, just fills the lookback ring.
    #[test]
    fn silence_produces_no_events() {
        let Ok(mut vad) = Vad::new() else {
            // onnxruntime.dll not available in test env — skip
            eprintln!("Skipping test: VAD init failed (DLL missing in test env)");
            return;
        };
        let silence = vec![0.0f32; SAMPLE_RATE]; // 1 second
        vad.push_samples(&silence);
        let events = vad.drain_events();
        assert!(
            events.is_empty(),
            "silence should not produce events, got {} events",
            events.len()
        );
    }

    /// Push nothing, drain nothing.
    #[test]
    fn no_samples_no_events() {
        let Ok(mut vad) = Vad::new() else {
            return;
        };
        assert!(vad.drain_events().is_empty());
    }

    /// Flush while idle is a no-op.
    #[test]
    fn flush_when_idle_is_noop() {
        let Ok(mut vad) = Vad::new() else {
            return;
        };
        let events = vad.flush();
        assert!(events.is_empty());
    }
}
