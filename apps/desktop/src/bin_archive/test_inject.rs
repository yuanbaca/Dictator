//! Test binary for text injection.
//!
//! Usage:
//!   cargo run --bin test-inject -- --mode type "Hello from DeskMic!"
//!   cargo run --bin test-inject -- --mode paste "Hello from DeskMic!"
//!
//! After running, you have 3 seconds to click into a text field (e.g. Notepad).
//! The text will be injected into whatever window is focused.

#[path = "../injection.rs"]
mod injection;

use clap::Parser;
use std::thread;
use std::time::Duration;

#[derive(Parser)]
#[command(name = "test-inject", about = "Test DeskMic text injection")]
struct Args {
    /// Injection mode: "type" or "paste"
    #[arg(short, long, default_value = "paste")]
    mode: String,

    /// Text to inject
    #[arg(default_value = "Hello from DeskMic Dictation! This is a test.")]
    text: String,

    /// Seconds to wait before injecting (time to focus target window)
    #[arg(short, long, default_value = "3")]
    delay: u64,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let mode = match args.mode.as_str() {
        "type" => injection::InjectionMode::Type,
        "paste" => injection::InjectionMode::Paste,
        other => anyhow::bail!("Unknown mode: {other}. Use 'type' or 'paste'."),
    };

    println!("=== DeskMic Injection Test ===");
    println!("Mode:  {:?}", mode);
    println!("Text:  {:?}", args.text);
    println!();
    println!("You have {} seconds to click into a text field...", args.delay);

    for i in (1..=args.delay).rev() {
        println!("  {i}...");
        thread::sleep(Duration::from_secs(1));
    }

    println!("Injecting now!");

    let start = std::time::Instant::now();
    injection::inject_text(&args.text, mode)?;
    let elapsed = start.elapsed();

    println!("Done! Injection took {elapsed:?}");
    Ok(())
}
