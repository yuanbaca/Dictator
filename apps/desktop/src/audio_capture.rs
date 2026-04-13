//! Microphone capture utilities.

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

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
/// Returns mono f32 samples at 16kHz.
pub fn record_until_stopped(
    stop: Arc<AtomicBool>,
    device_name: Option<String>,
    peak_level: Arc<AtomicU32>,
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

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let chunk_peak = data.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
                peak_f32.fetch_max(chunk_peak.to_bits(), Ordering::Relaxed);
                samples_clone.lock().unwrap().extend_from_slice(data);
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
        let ratio = 16000.0 / sample_rate as f64;
        let new_len = (mono.len() as f64 * ratio) as usize;
        let mut resampled = Vec::with_capacity(new_len);
        for i in 0..new_len {
            let src_pos = i as f64 / ratio;
            let idx = src_pos as usize;
            let frac = src_pos - idx as f64;
            let sample = if idx + 1 < mono.len() {
                mono[idx] * (1.0 - frac as f32) + mono[idx + 1] * frac as f32
            } else if idx < mono.len() {
                mono[idx]
            } else {
                0.0
            };
            resampled.push(sample);
        }
        Ok(resampled)
    }
}
