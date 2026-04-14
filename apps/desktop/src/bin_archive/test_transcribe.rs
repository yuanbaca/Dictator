//! Test binary for whisper transcription.
//!
//! Usage:
//!   cargo run --bin test-transcribe -- --model models/ggml-base.en.bin audio.wav
//!
//! The WAV file must be 16kHz mono (or stereo, which will be downmixed).

#[path = "../transcription.rs"]
mod transcription;

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "test-transcribe", about = "Test DeskMic whisper transcription")]
struct Args {
    /// Path to whisper model file (.bin)
    #[arg(short = 'M', long, default_value = "models/ggml-base.en.bin")]
    model: PathBuf,

    /// Path to WAV file to transcribe
    wav: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("=== DeskMic Transcription Test ===");
    println!("Model: {}", args.model.display());
    println!("Audio: {}", args.wav.display());
    println!();

    // Load audio
    println!("Loading WAV file...");
    let samples = transcription::load_wav_as_samples(&args.wav)?;
    let duration_secs = samples.len() as f64 / 16000.0;
    println!("Loaded {:.1}s of audio ({} samples)", duration_secs, samples.len());
    println!();

    // Load model and transcribe
    println!("Loading whisper model...");
    let start = std::time::Instant::now();
    let transcriber = transcription::Transcriber::new(&args.model, false)?;
    println!("Model loaded in {:?}", start.elapsed());
    println!();

    println!("Transcribing...");
    let start = std::time::Instant::now();
    let result = transcriber.transcribe(&samples)?;
    let elapsed = start.elapsed();

    println!();
    println!("=== Result ===");
    println!("Text: {}", result.text);
    println!();
    println!("Segments:");
    for seg in &result.segments {
        println!(
            "  [{:>6}ms - {:>6}ms] {}",
            seg.start_ms, seg.end_ms, seg.text
        );
    }
    println!();
    println!("Transcription took {elapsed:?} for {duration_secs:.1}s of audio");
    println!(
        "Speed: {:.1}x realtime",
        duration_secs / elapsed.as_secs_f64()
    );

    Ok(())
}
