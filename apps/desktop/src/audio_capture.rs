//! Microphone capture utilities.
//!
//! Two capture modes:
//! - **Press-to-talk** (`record_until_stopped`): accumulates all audio in
//!   memory and returns the complete buffer once the stop flag is set.
//! - **Live streaming** (`stream_to_live`): pipes each cpal callback
//!   chunk — mono-converted and resampled to 16 kHz — directly into a
//!   `tokio::sync::mpsc::UnboundedSender<Vec<f32>>` so the live session's
//!   VAD can process audio in real time.

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

/// Return a list of available input device names.
pub fn list_input_devices() -> Vec<String> {
    let host = cpal::default_host();
    let mut names = Vec::new();
    if let Ok(devices) = host.input_devices() {
        for dev in devices {
            if let Ok(name) = dev.name() {
                names.push(name);
            }
        }
    }
    names
}

/// Records from the microphone until `stop` is set to true.
/// If `device_name` is Some, uses that device; otherwise uses the system default.
/// `peak_level` is atomically updated with the maximum absolute sample value
/// since it was last read (enables real-time silence detection from another thread).
///
/// When `preview_tx` is Some, each callback also mono-converts + resamples the
/// audio to 16 kHz and sends it to the channel, enabling a live VAD preview
/// to run alongside the recording without a second microphone stream.
///
/// When `silence_tx` is Some, each callback sends the same 16 kHz mono audio
/// to a separate channel intended for the silence-auto-stop watchdog. Both
/// channels are independent so the preview pipeline and the watchdog can
/// coexist without competing for the mic.
///
/// Returns mono f32 samples at 16kHz.
pub fn record_until_stopped(
    stop: Arc<AtomicBool>,
    device_name: Option<String>,
    peak_level: Arc<AtomicU32>,
    preview_tx: Option<mpsc::UnboundedSender<Vec<f32>>>,
    silence_tx: Option<mpsc::UnboundedSender<Vec<f32>>>,
) -> Result<Vec<f32>> {
    let host = cpal::default_host();
    let device = if let Some(ref name) = device_name {
        host.input_devices()
            .context("Failed to enumerate audio devices")?
            .find(|d| d.name().map(|n| n == *name).unwrap_or(false))
            .with_context(|| format!("Audio device '{}' not found", name))?
    } else {
        host.default_input_device()
            .context("No microphone found! Check your audio settings.")?
    };

    let config = device
        .default_input_config()
        .context("Failed to get microphone config")?;

    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;

    let raw_samples: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let samples_clone = raw_samples.clone();

    // Clone peak_level for each callback — non-negative f32 bit patterns sort
    // identically to unsigned integers, so fetch_max on to_bits() is correct.
    let peak_f32 = peak_level.clone();
    let peak_i16 = peak_level.clone();
    let peak_u16 = peak_level.clone();

    // Clone preview sender for each sample-format branch.
    let ptx_f32 = preview_tx.clone();
    let ptx_i16 = preview_tx.clone();
    let ptx_u16 = preview_tx;

    // Same fan-out for the silence-auto-stop watchdog channel.
    let stx_f32 = silence_tx.clone();
    let stx_i16 = silence_tx.clone();
    let stx_u16 = silence_tx;

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let chunk_peak = data.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
                peak_f32.fetch_max(chunk_peak.to_bits(), Ordering::Relaxed);
                samples_clone.lock().unwrap().extend_from_slice(data);
                // Fork to preview pipeline (mono + resample to 16 kHz). Reuse
                // the same conversion for the silence watchdog when both are
                // active so we don't pay for the resampling twice.
                let needs_16k = ptx_f32.is_some() || stx_f32.is_some();
                if needs_16k {
                    let mono = to_mono_f32(data, channels);
                    let chunk = resample_chunk(&mono, sample_rate, 16000);
                    if let Some(ref tx) = ptx_f32 {
                        let _ = tx.send(chunk.clone());
                    }
                    if let Some(ref tx) = stx_f32 {
                        let _ = tx.send(chunk);
                    }
                }
            },
            |err| eprintln!("Audio error: {err}"),
            None,
        )?,
        cpal::SampleFormat::I16 => {
            let sc = raw_samples.clone();
            device.build_input_stream(
                &config.into(),
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let floats: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();
                    let chunk_peak = floats.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
                    peak_i16.fetch_max(chunk_peak.to_bits(), Ordering::Relaxed);
                    sc.lock().unwrap().extend_from_slice(&floats);
                    let needs_16k = ptx_i16.is_some() || stx_i16.is_some();
                    if needs_16k {
                        let mono = to_mono_f32(&floats, channels);
                        let chunk = resample_chunk(&mono, sample_rate, 16000);
                        if let Some(ref tx) = ptx_i16 {
                            let _ = tx.send(chunk.clone());
                        }
                        if let Some(ref tx) = stx_i16 {
                            let _ = tx.send(chunk);
                        }
                    }
                },
                |err| eprintln!("Audio error: {err}"),
                None,
            )?
        }
        cpal::SampleFormat::U16 => {
            let sc = raw_samples.clone();
            device.build_input_stream(
                &config.into(),
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    let floats: Vec<f32> = data
                        .iter()
                        .map(|&s| (s as f32 - 32768.0) / 32768.0)
                        .collect();
                    let chunk_peak = floats.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
                    peak_u16.fetch_max(chunk_peak.to_bits(), Ordering::Relaxed);
                    sc.lock().unwrap().extend_from_slice(&floats);
                    let needs_16k = ptx_u16.is_some() || stx_u16.is_some();
                    if needs_16k {
                        let mono = to_mono_f32(&floats, channels);
                        let chunk = resample_chunk(&mono, sample_rate, 16000);
                        if let Some(ref tx) = ptx_u16 {
                            let _ = tx.send(chunk.clone());
                        }
                        if let Some(ref tx) = stx_u16 {
                            let _ = tx.send(chunk);
                        }
                    }
                },
                |err| eprintln!("Audio error: {err}"),
                None,
            )?
        }
        fmt => anyhow::bail!("Unsupported audio format: {fmt:?}"),
    };

    stream.play().context("Failed to start microphone")?;

    // Wait for stop signal
    while !stop.load(Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    drop(stream);

    let raw = raw_samples.lock().unwrap();

    // Convert to mono
    let mono: Vec<f32> = if channels > 1 {
        raw.chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        raw.clone()
    };

    // Resample to 16kHz if needed
    if sample_rate == 16000 {
        Ok(mono)
    } else {
        Ok(resample_chunk(&mono, sample_rate, 16000))
    }
}

// ─── Live-mode streaming ───────────────────────────────────────────────

/// A running microphone capture that feeds audio into a live session.
/// Dropping the handle (or calling [`stop`](Self::stop)) ends the capture
/// **synchronously** — i.e. by the time `drop` returns, the underlying
/// cpal stream has been released back to the OS and the mic device can
/// be opened again. This matters because Windows audio drivers routinely
/// refuse two concurrent input streams on the same device, and our own
/// recording / wake-word flows hand the mic back and forth in quick
/// succession.
pub struct LiveCaptureHandle {
    stop: Arc<AtomicBool>,
    /// Join handle for the cpal keepalive thread. Taken on drop so we
    /// can wait for it to finish releasing the cpal stream.
    join: Option<std::thread::JoinHandle<()>>,
}

impl LiveCaptureHandle {
    /// Signal the capture to shut down. Returns immediately — use `Drop`
    /// (i.e. let the handle go out of scope) if you need a synchronous
    /// stop that guarantees mic release.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

impl Drop for LiveCaptureHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.join.take() {
            // Block until the cpal thread exits and drops `cpal::Stream`,
            // which is what actually releases the WASAPI handle. Bounded
            // in practice by the keepalive sleep interval below.
            let _ = handle.join();
        }
    }
}

/// Stream microphone audio at 16 kHz mono into a live session's frame
/// channel.
///
/// Each cpal callback (~10 ms at 48 kHz) is mono-converted, linearly
/// resampled to 16 kHz, and pushed into `frame_tx`. The VAD internally
/// re-buffers into its own 512-sample (32 ms) windows so any incoming
/// chunk size is fine.
///
/// If `device_name` is `Some`, uses that device; otherwise uses the
/// system default input.
///
/// `cpal::Stream` is `!Send`, so device opening, stream building, and
/// the keepalive spin-loop all happen on a single dedicated thread.
/// The calling thread blocks briefly (sub-millisecond) on a
/// synchronous channel until the thread reports init success or
/// failure, then returns the handle.
pub fn stream_to_live(
    frame_tx: mpsc::UnboundedSender<Vec<f32>>,
    device_name: Option<String>,
) -> Result<LiveCaptureHandle> {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = stop.clone();

    // The spawned thread reports back whether init succeeded before
    // entering its keepalive loop. We block on this channel so callers
    // get a synchronous Result.
    let (init_tx, init_rx) = std::sync::mpsc::sync_channel::<Result<()>>(0);

    let join = std::thread::Builder::new()
        .name("live-mic".into())
        .spawn(move || {
            let result = (|| -> Result<cpal::Stream> {
                let host = cpal::default_host();
                let device = if let Some(ref name) = device_name {
                    host.input_devices()
                        .context("Failed to enumerate audio devices")?
                        .find(|d| d.name().map(|n| n == *name).unwrap_or(false))
                        .with_context(|| format!("Audio device '{name}' not found"))?
                } else {
                    host.default_input_device()
                        .context("No microphone found! Check your audio settings.")?
                };

                let config = device
                    .default_input_config()
                    .context("Failed to get microphone config")?;

                let sample_rate = config.sample_rate().0;
                let channels = config.channels() as usize;

                let stream = match config.sample_format() {
                    cpal::SampleFormat::F32 => {
                        let tx = frame_tx;
                        device.build_input_stream(
                            &config.into(),
                            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                                let mono = to_mono_f32(data, channels);
                                let chunk = resample_chunk(&mono, sample_rate, 16000);
                                let _ = tx.send(chunk);
                            },
                            |err| eprintln!("Live audio error: {err}"),
                            None,
                        )?
                    }
                    cpal::SampleFormat::I16 => {
                        let tx = frame_tx;
                        device.build_input_stream(
                            &config.into(),
                            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                                let floats: Vec<f32> =
                                    data.iter().map(|&s| s as f32 / 32768.0).collect();
                                let mono = to_mono_f32(&floats, channels);
                                let chunk = resample_chunk(&mono, sample_rate, 16000);
                                let _ = tx.send(chunk);
                            },
                            |err| eprintln!("Live audio error: {err}"),
                            None,
                        )?
                    }
                    cpal::SampleFormat::U16 => {
                        let tx = frame_tx;
                        device.build_input_stream(
                            &config.into(),
                            move |data: &[u16], _: &cpal::InputCallbackInfo| {
                                let floats: Vec<f32> = data
                                    .iter()
                                    .map(|&s| (s as f32 - 32768.0) / 32768.0)
                                    .collect();
                                let mono = to_mono_f32(&floats, channels);
                                let chunk = resample_chunk(&mono, sample_rate, 16000);
                                let _ = tx.send(chunk);
                            },
                            |err| eprintln!("Live audio error: {err}"),
                            None,
                        )?
                    }
                    fmt => anyhow::bail!("Unsupported audio format: {fmt:?}"),
                };

                stream.play().context("Failed to start microphone")?;
                Ok(stream)
            })();

            match result {
                Ok(stream) => {
                    // Signal init success, then keep the stream alive.
                    let _ = init_tx.send(Ok(()));
                    let _stream = stream;
                    // Short keepalive interval so `LiveCaptureHandle::drop`'s
                    // join doesn't stall recordings/listeners that need to
                    // reopen the mic. 20 ms is fast enough that the user
                    // doesn't notice and slow enough not to peg a CPU core.
                    while !stop_clone.load(Ordering::Relaxed) {
                        std::thread::sleep(std::time::Duration::from_millis(20));
                    }
                    // _stream drops here, releasing the WASAPI handle.
                }
                Err(e) => {
                    let _ = init_tx.send(Err(e));
                }
            }
        })
        .context("Failed to spawn live-mic thread")?;

    // Block until the mic thread finishes init.
    init_rx
        .recv()
        .map_err(|_| anyhow::anyhow!("live-mic thread exited before init"))??;

    Ok(LiveCaptureHandle { stop, join: Some(join) })
}

// ─── Shared helpers ────────────────────────────────────────────────────

/// Mix multi-channel f32 audio down to mono by averaging channels.
fn to_mono_f32(data: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        data.to_vec()
    } else {
        data.chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    }
}

/// Linear-interpolation resample from `from_rate` Hz to `to_rate` Hz.
/// Returns the input unchanged when rates match.
fn resample_chunk(mono: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || mono.is_empty() {
        return mono.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let new_len = (mono.len() as f64 * ratio) as usize;
    let mut out = Vec::with_capacity(new_len);
    for i in 0..new_len {
        let src = i as f64 / ratio;
        let idx = src as usize;
        let frac = (src - idx as f64) as f32;
        let sample = if idx + 1 < mono.len() {
            mono[idx] * (1.0 - frac) + mono[idx + 1] * frac
        } else if idx < mono.len() {
            mono[idx]
        } else {
            0.0
        };
        out.push(sample);
    }
    out
}
