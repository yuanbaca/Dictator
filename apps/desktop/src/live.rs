//! Live-mode transcription session.
//!
//! One session ties together VAD endpointing and per-segment Whisper
//! transcription. The producer (phone WebSocket handler or local cpal loop)
//! pushes raw f32 samples at 16 kHz into `frame_tx`; the session spins up a
//! background task that runs samples through Silero VAD, and whenever the
//! VAD reports an endpoint, hands that utterance to Whisper on a blocking
//! thread.
//!
//! Transcription results come back via `event_rx` so the producer can inject
//! them, forward them to the phone, emit a Tauri event to the desktop UI,
//! and write to history — all orthogonal to what the session does.
//!
//! Drop `frame_tx` to end the session cleanly — the task flushes any
//! in-progress utterance, drains the transcription queue, emits `Ended`,
//! and shuts down.

use crate::transcription::Transcriber;
use crate::vad::{Vad, VadEvent, SAMPLE_RATE};
use anyhow::Result;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

/// Minimum utterance length (in samples) to bother transcribing.
/// Shorter bursts are almost always coughs, lip smacks, or breaths that
/// produce garbage from Whisper ("...", "you", etc.). With the current VAD
/// settings (~640 ms trailing silence + 256 ms lookback), roughly 0.9 s of
/// each utterance is padding. Requiring 1.1 s total means the speaker must
/// produce at least ~200 ms of actual speech — enough to filter lip smacks
/// while keeping single follow-up words like "thing" or "wrong".
const MIN_UTTERANCE_SAMPLES: usize = SAMPLE_RATE * 11 / 10; // 1.1 s → 17 600 samples

/// Events emitted by a running live session.
#[derive(Debug, Clone)]
pub enum LiveEvent {
    /// Speech just began on a new utterance. Useful for UI status.
    SegmentStart,
    /// A segment finished transcribing.
    Segment {
        /// Raw Whisper output for this utterance.
        text: String,
        /// Wall-clock milliseconds spent in `transcribe`.
        duration_ms: u64,
    },
    /// Transcription failed for one segment. The session keeps running.
    SegmentError(String),
    /// Session has cleanly shut down; no more events will follow.
    Ended,
}

/// Producer-facing handle to a running session.
pub struct LiveSessionHandle {
    /// Send mono f32 samples at 16 kHz in any chunk size. Drop (or call
    /// `end_input`) to signal the end of the audio stream.
    pub frame_tx: mpsc::UnboundedSender<Vec<f32>>,
    /// Pull transcription events.
    pub event_rx: mpsc::UnboundedReceiver<LiveEvent>,
}

impl LiveSessionHandle {
    /// Signal the session to flush and wind down. Replaces the audio sender
    /// with one whose receiver has already been dropped, so the session's
    /// recv() sees no more senders, completes, flushes any in-progress
    /// utterance through Whisper, and emits `LiveEvent::Ended`.
    ///
    /// Keep the handle around after calling this so you can still drain
    /// events from `event_rx`.
    pub fn end_input(&mut self) {
        let (dead, _) = mpsc::unbounded_channel::<Vec<f32>>();
        self.frame_tx = dead;
    }
}

/// Spawn a live session. Runs until the caller drops `frame_tx`.
///
/// `transcriber` is the shared app transcriber; the session locks it inside
/// `spawn_blocking` per segment, the same pattern press-to-talk uses.
///
/// When `cross_segment_context` is true, the tail of each segment's
/// transcription is fed as `initial_prompt` to the next segment, giving
/// Whisper linguistic continuity across VAD-gated boundaries.
pub fn spawn(
    transcriber: Arc<Mutex<Option<Transcriber>>>,
    cross_segment_context: bool,
) -> Result<LiveSessionHandle> {
    let (frame_tx, mut frame_rx) = mpsc::unbounded_channel::<Vec<f32>>();
    let (event_tx, event_rx) = mpsc::unbounded_channel::<LiveEvent>();

    // Initialize VAD up front so producers can fail fast if onnxruntime.dll
    // is missing — returning the error synchronously is friendlier than a
    // silent task crash.
    let mut vad = Vad::new()?;

    tokio::spawn(async move {
        // Inner channel: VAD reader pushes completed utterance audio here;
        // a dedicated transcriber task drains it and runs Whisper. Keeping
        // the two loops separate means VAD keeps reading new audio even
        // while Whisper is busy on the previous segment.
        let (seg_tx, mut seg_rx) = mpsc::unbounded_channel::<Vec<f32>>();

        let transcriber_inner = transcriber.clone();
        let event_tx_inner = event_tx.clone();
        let transcriber_task = tokio::spawn(async move {
            // When cross-segment context is on, we carry the tail of the
            // previous segment's text forward as Whisper's initial_prompt.
            // Cap at ~200 chars — Whisper's prompt token budget is limited
            // and we only need the trailing clause for continuity.
            let mut prev_context: Option<String> = None;

            while let Some(samples) = seg_rx.recv().await {
                // Skip micro-utterances that produce Whisper garbage
                // ("...", "you", "the", single-word hallucinations).
                if samples.len() < MIN_UTTERANCE_SAMPLES {
                    eprintln!(
                        "live: skipping micro-utterance ({} ms, need {} ms)",
                        samples.len() * 1000 / SAMPLE_RATE,
                        MIN_UTTERANCE_SAMPLES * 1000 / SAMPLE_RATE,
                    );
                    continue;
                }
                let start = std::time::Instant::now();
                let t = transcriber_inner.clone();
                let ctx = if cross_segment_context {
                    prev_context.clone()
                } else {
                    None
                };
                let join = tokio::task::spawn_blocking(move || {
                    let guard = t.lock().unwrap();
                    match guard.as_ref() {
                        Some(tx) => tx
                            .transcribe_with_context(&samples, ctx.as_deref())
                            .map(|r| r.text)
                            .map_err(|e| e.to_string()),
                        None => Err("Model not loaded".to_string()),
                    }
                })
                .await;

                let duration_ms = start.elapsed().as_millis() as u64;
                let event = match join {
                    Ok(Ok(ref text)) if cross_segment_context => {
                        // Keep the tail of this segment as context for the next.
                        let tail: String = text.chars().rev().take(200).collect::<Vec<_>>().into_iter().rev().collect();
                        prev_context = Some(tail);
                        LiveEvent::Segment {
                            text: text.clone(),
                            duration_ms,
                        }
                    }
                    Ok(Ok(text)) => LiveEvent::Segment { text, duration_ms },
                    Ok(Err(e)) => LiveEvent::SegmentError(e),
                    Err(e) => LiveEvent::SegmentError(format!("task panicked: {e}")),
                };
                if event_tx_inner.send(event).is_err() {
                    // Receiver dropped — nobody is listening, no point continuing.
                    break;
                }
            }
        });

        // VAD pump: read audio frames from the producer, feed the VAD, and
        // forward its events. Start events go straight to the caller; End
        // events route the utterance samples to the transcriber task.
        while let Some(samples) = frame_rx.recv().await {
            vad.push_samples(&samples);
            for ev in vad.drain_events() {
                match ev {
                    VadEvent::Start => {
                        if event_tx.send(LiveEvent::SegmentStart).is_err() {
                            break;
                        }
                    }
                    VadEvent::End { samples } => {
                        if seg_tx.send(samples).is_err() {
                            break;
                        }
                    }
                }
            }
        }

        // Producer is done — flush whatever the VAD was holding.
        for ev in vad.flush() {
            if let VadEvent::End { samples } = ev {
                let _ = seg_tx.send(samples);
            }
        }

        // Closing seg_tx lets the transcriber task drain the queue and exit.
        drop(seg_tx);
        let _ = transcriber_task.await;
        let _ = event_tx.send(LiveEvent::Ended);
    });

    Ok(LiveSessionHandle { frame_tx, event_rx })
}
