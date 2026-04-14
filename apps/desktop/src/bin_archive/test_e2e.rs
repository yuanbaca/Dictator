//! End-to-end test: load WAV -> transcribe -> inject into focused window.
//!
//! Usage:
//!   cargo run --bin test-e2e -- --model models/ggml-base.en.bin --mode paste audio.wav
//!
//! After transcription completes, you have a few seconds to click into a text field.

#[path = "../injection.rs"]
mod injection;
#[path = "../transcription.rs"]
mod transcription;

use clap::Parser;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

#[derive(Parser)]
#[command(name = "test-e2e", about = "End-to-end DeskMic test: WAV -> transcribe -> inject")]
struct Args {
    /// Path to whisper model file (.bin)
    #[arg(short = 'M', long, default_value = "models/ggml-base.en.bin")]
    model: PathBuf,

    /// Injection mode: "type" or "paste"
    #[arg(short, long, default_value = "paste")]
    mode: String,

    /// Seconds to wait before injection
    #[arg(short, long, default_value = "3")]
    delay: u64,

    /// Path to WAV file
    wav: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let total_start = std::time::Instant::now();

    let inject_mode = match args.mode.as_str() {
        "type" => injection::InjectionMode::Type,
        "paste" => injection::InjectionMode::Paste,
        other => anyhow::bail!("Unknown mode: {other}. Use 'type' or 'paste'."),
    };

    println!("=== DeskMic End-to-End Test ===");
    println!("Model:     {}", args.model.display());
    println!("Audio:     {}", args.wav.display());
    println!("Inject:    {:?}", inject_mode);
    println!();

    // Step 1: Load audio
    println!("[1/3] Loading WAV file...");
    let start = std::time::Instant::now();
    let samples = transcription::load_wav_as_samples(&args.wav)?;
    let audio_duration = samples.len() as f64 / 16000.0;
    println!("  Loaded {:.1}s of audio in {:?}", audio_duration, start.elapsed());

    // Step 2: Transcribe
    println!("[2/3] Transcribing...");
    let start = std::time::Instant::now();
    let transcriber = transcription::Transcriber::new(&args.model, false)?;
    let model_load = start.elapsed();

    let start = std::time::Instant::now();
    let result = transcriber.transcribe(&samples)?;
    let transcribe_time = start.elapsed();

    println!("  Model load: {model_load:?}");
    println!("  Transcribe: {transcribe_time:?}");
    println!("  Text: \"{}\"", result.text);
    println!();

    // Step 3: Inject
    println!("[3/3] Preparing to inject text...");
    println!("  Click into a text field now! You have {} seconds...", args.delay);

    for i in (1..=args.delay).rev() {
        println!("  {i}...");
        thread::sleep(Duration::from_secs(1));
    }

    println!("  Injecting!");
    let start = std::time::Instant::now();
    injection::inject_text(&result.text, inject_mode)?;
    let inject_time = start.elapsed();

    println!("  Injection took {inject_time:?}");
    println!();
    println!("=== Summary ===");
    println!("  Audio duration:    {audio_duration:.1}s");
    println!("  Model load:        {model_load:?}");
    println!("  Transcription:     {transcribe_time:?}");
    println!("  Injection:         {inject_time:?}");
    println!("  Total wall time:   {:?}", total_start.elapsed());

    Ok(())
}
