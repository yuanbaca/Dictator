//! Generate a short silent WAV file for testing the pipeline.
//! Usage: cargo run --bin gen-test-wav

fn main() -> anyhow::Result<()> {
    let path = "test_silence.wav";
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)?;

    // 2 seconds of silence
    for _ in 0..(16000 * 2) {
        writer.write_sample(0i16)?;
    }

    writer.finalize()?;
    println!("Created {path} (2s silence, 16kHz mono)");
    println!();
    println!("For a real test, record yourself saying something and save as");
    println!("a 16kHz mono WAV file, then run:");
    println!("  cargo run --bin test-e2e -- your_recording.wav");

    Ok(())
}
