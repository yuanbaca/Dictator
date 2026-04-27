//! Tiny persistence layer for user preferences that *must* be known before
//! the Tauri webview has booted.
//!
//! Most UI state lives in `localStorage` and gets pushed into the Rust
//! backend after the webview loads. That's fine for anything the backend
//! doesn't need immediately — hotkey strings, TTS voice IDs, etc. — because
//! the UI sync runs within a second of startup.
//!
//! `force_cpu` is different: the startup thread in `main.rs` has to pick
//! a value *before* the webview exists so it knows whether to load Whisper
//! on GPU or CPU. If we default to `false` and rely on the UI to flip us
//! later, we hit a nasty race:
//!
//!   1. Backend spawns loader thread with `force_cpu = false` → loads GPU.
//!   2. Webview boots, reads `dictator-force-cpu = 1` from `localStorage`,
//!      calls `set_force_cpu(true)`.
//!   3. `set_force_cpu` clears the just-loaded transcriber (preference
//!      genuinely changed), but nothing re-triggers the load.
//!   4. UI sits on "Loading whisper model…" forever.
//!
//! Persisting `force_cpu` to disk and reading it at startup eliminates the
//! race: the loader thread reads the correct value the first time, and the
//! UI's startup sync becomes a no-op (idempotent because values match).
//!
//! Stored as JSON in the models directory alongside `word_bank.json` for
//! consistency with the existing persistence pattern. Room left to grow
//! into a general prefs file if we need to pre-load other settings later.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
struct UserPrefs {
    #[serde(default)]
    force_cpu: bool,
}

fn prefs_path() -> PathBuf {
    crate::model_manager::models_dir().join("user_prefs.json")
}

fn load_all() -> UserPrefs {
    let path = prefs_path();
    match std::fs::read_to_string(&path) {
        Ok(s) => serde_json::from_str(&s).unwrap_or_else(|e| {
            eprintln!(
                "user_prefs: malformed {} ({e}) — using defaults",
                path.display()
            );
            UserPrefs::default()
        }),
        Err(_) => UserPrefs::default(),
    }
}

fn save_all(prefs: &UserPrefs) -> Result<(), String> {
    let path = prefs_path();
    let json = serde_json::to_string_pretty(prefs)
        .map_err(|e| format!("user_prefs serialize failed: {e}"))?;
    std::fs::write(&path, json)
        .map_err(|e| format!("user_prefs write failed: {e}"))
}

/// Return the persisted `force_cpu` preference, or `false` if no prefs
/// file exists yet. Never fails — a broken prefs file falls back to the
/// default rather than blocking startup.
pub fn load_force_cpu() -> bool {
    load_all().force_cpu
}

/// Persist a new `force_cpu` value. Errors are logged and swallowed: disk
/// issues shouldn't break the in-memory toggle.
pub fn save_force_cpu(force: bool) {
    let mut prefs = load_all();
    if prefs.force_cpu == force {
        return;
    }
    prefs.force_cpu = force;
    if let Err(e) = save_all(&prefs) {
        eprintln!("user_prefs: failed to save force_cpu ({e})");
    }
}
