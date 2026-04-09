//! Live dictation from PC microphone.
//!
//! 1. Press Enter to start recording
//! 2. Speak into your microphone
//! 3. Press Enter to stop
//! 4. Click into a text field within 3 seconds
//! 5. Transcribed text appears!

#[path = "../injection.rs"]
mod injection;
#[path = "../transcription.rs"]
mod transcription;

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

fn main() -> Result<()> {
    println!("=========================================");
    println!("   DeskMic Dictation - Microphone Test");
    println!("=========================================");
    println!();

    // Find the model
    let model_path = find_model()?;
    println!("Loading whisper model...");
    let transcriber = transcription::Transcriber::new(&model_path)?;
    println!("Model loaded!");
    println!();

    loop {
        println!("-----------------------------------------");
        println!("Press ENTER to start recording (or type 'quit' to exit)");
        print!("> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("q") {
            println!("Goodbye!");
            break;
        }

        // Record audio
        println!();
        println!("  ** RECORDING ** Speak now!");
        println!("  Press ENTER when done speaking...");
        println!();

        let samples = record_until_enter()?;
        let duration = samples.len() as f64 / 16000.0;

        if duration < 0.5 {
            println!("  Recording too short ({:.1}s). Try again.", duration);
            continue;
        }

        println!("  Recorded {:.1}s of audio.", duration);
        println!();

        // Transcribe
        println!("  Transcribing...");
        let start = std::time::Instant::now();
        let result = transcriber.transcribe(&samples)?;
        let elapsed = start.elapsed();

        if result.text.is_empty() {
            println!("  No speech detected. Try again.");
            continue;
        }

        println!("  Transcribed in {:.1}s", elapsed.as_secs_f64());
        println!();
        println!("  Text: \"{}\"", result.text);
        println!();

        // Ask what to do
        println!("  What would you like to do?");
        println!("    [1] Paste into focused window (you'll have 3 seconds to click a text field)");
        println!("    [2] Type into focused window (slower but works in more apps)");
        println!("    [3] Skip (don't insert)");
        print!("  > ");
        io::stdout().flush()?;

        let mut choice = String::new();
        io::stdin().read_line(&mut choice)?;

        match choice.trim() {
            "1" | "" => {
                println!();
                println!("  Click into a text field now!");
                countdown(3);
                println!("  Pasting...");
                injection::inject_text(&result.text, injection::InjectionMode::Paste)?;
                println!("  Done!");
            }
            "2" => {
                println!();
                println!("  Click into a text field now!");
                countdown(3);
                println!("  Typing...");
                injection::inject_text(&result.text, injection::InjectionMode::Type)?;
                println!("  Done!");
            }
            _ => {
                println!("  Skipped.");
            }
        }

        println!();
    }

    Ok(())
}

/// Record from the default microphone until the user presses Enter.
/// Returns mono f32 samples at 16kHz.
fn record_until_enter() -> Result<Vec<f32>> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("No microphone found! Check your audio settings.")?;

    let config = device
        .default_input_config()
        .context("Failed to get microphone config")?;

    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;

    let raw_samples: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let samples_clone = raw_samples.clone();

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                samples_clone.lock().unwrap().extend_from_slice(data);
            },
            |err| eprintln!("  Audio error: {err}"),
            None,
        )?,
        cpal::SampleFormat::I16 => {
            let samples_clone = raw_samples.clone();
            device.build_input_stream(
                &config.into(),
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let floats: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();
                    samples_clone.lock().unwrap().extend_from_slice(&floats);
                },
                |err| eprintln!("  Audio error: {err}"),
                None,
            )?
        }
        cpal::SampleFormat::U16 => {
            let samples_clone = raw_samples.clone();
            device.build_input_stream(
                &config.into(),
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    let floats: Vec<f32> = data
                        .iter()
                        .map(|&s| (s as f32 - 32768.0) / 32768.0)
                        .collect();
                    samples_clone.lock().unwrap().extend_from_slice(&floats);
                },
                |err| eprintln!("  Audio error: {err}"),
                None,
            )?
        }
        fmt => anyhow::bail!("Unsupported audio format: {fmt:?}"),
    };

    stream.play().context("Failed to start microphone")?;

    // Wait for Enter
    let mut buf = String::new();
    io::stdin().read_line(&mut buf)?;

    drop(stream);

    let raw = raw_samples.lock().unwrap();

    // Convert to mono if stereo
    let mono: Vec<f32> = if channels > 1 {
        raw.chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        raw.clone()
    };

    // Resample to 16kHz if needed (simple linear interpolation)
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

fn countdown(seconds: u64) {
    for i in (1..=seconds).rev() {
        print!("  {}...", i);
        io::stdout().flush().unwrap();
        thread::sleep(Duration::from_secs(1));
    }
    println!();
}

/// Search for the whisper model in common locations relative to the binary.
fn find_model() -> Result<PathBuf> {
    let candidates = [
        PathBuf::from("../../models/ggml-base.en.bin"),
        PathBuf::from("models/ggml-base.en.bin"),
        PathBuf::from("../models/ggml-base.en.bin"),
        PathBuf::from("../../models/ggml-base.en.bin"),
    ];

    for path in &candidates {
        if path.exists() {
            println!("Found model: {}", path.display());
            return Ok(path.clone());
        }
    }

    // Try relative to executable location
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let model = exe_dir.join("../../models/ggml-base.en.bin");
            if model.exists() {
                println!("Found model: {}", model.display());
                return Ok(model);
            }
        }
    }

    anyhow::bail!(
        "Could not find whisper model!\n\
         Expected at: models/ggml-base.en.bin\n\
         \n\
         The model should have been downloaded during setup.\n\
         If missing, it can be downloaded from:\n\
         https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin\n\
         \n\
         Save it to the 'models' folder in the project root."
    )
}
