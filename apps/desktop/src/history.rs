//! Persistent transcription history.
//!
//! Each completed dictation leaves an `Entry` here: the raw Whisper output
//! (pre-word-bank, for troubleshooting / spotting word-bank candidates),
//! the final text the user ended up with (post-word-bank and optionally
//! post-LLM-format), plus metadata. The UI shows final text by default and
//! reveals raw on click.
//!
//! **Storage:** JSON file in the models directory, same pattern as
//! `word_bank.json` and `chat_format_overrides.json`. Capped at
//! `MAX_ENTRIES` — newest first; oldest dropped on overflow.
//!
//! **Privacy note:** this is transcript data sitting on disk. Users who
//! dictate sensitive content can clear it from the panel (individual
//! entries or all). The file lives alongside the rest of the app's user
//! data, not in a cloud-synced directory.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Max entries we keep on disk. 50 is enough to cover a work session's
/// worth of dictation and a day's worth of casual use without the file
/// growing unbounded. Bigger would be fine for performance; the cap is
/// mostly about avoiding a forever-growing personal-data file.
pub const MAX_ENTRIES: usize = 50;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// Unique identifier, also doubles as creation time (ms since epoch).
    /// The UI uses this for delete operations and for stable keying when
    /// re-rendering.
    pub id: u64,
    /// Same as `id` — kept as a separate field so older JSON files that
    /// someone hand-edits don't need `id` to make sense as a timestamp.
    pub timestamp: u64,
    /// What Whisper actually produced, BEFORE word-bank substitution.
    /// This is what users need to see to discover new word-bank entries
    /// ("oh, Whisper keeps hearing 'clod' instead of 'Claude'").
    pub raw: String,
    /// What the user ended up with after the full pipeline — word bank
    /// applied, then LLM format if one was used. If no format was applied,
    /// this is the same as the post-word-bank text.
    pub final_text: String,
    /// Format type used, or "none". Matches the strings used by the
    /// format enum (clean_up, email, etc.) so the UI can show a label.
    pub format_type: String,
}

fn history_path() -> PathBuf {
    crate::model_manager::models_dir().join("history.json")
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        // Clock going backwards is exotic enough to not try to recover —
        // use 0 so new entries are still distinguishable by insertion
        // order even if timestamps are weird.
        .unwrap_or(0)
}

/// Load the full history from disk. Missing / malformed files return an
/// empty Vec rather than erroring — a broken history file should never
/// block the app.
pub fn load() -> Vec<Entry> {
    let path = history_path();
    match std::fs::read_to_string(&path) {
        Ok(s) => serde_json::from_str(&s).unwrap_or_else(|e| {
            eprintln!("history: malformed {} ({e}) — ignoring", path.display());
            Vec::new()
        }),
        Err(_) => Vec::new(),
    }
}

/// Persist the given list. Caller is responsible for ordering (newest
/// first) and capping to `MAX_ENTRIES`.
pub fn save(entries: &[Entry]) -> Result<(), String> {
    let path = history_path();
    let json = serde_json::to_string_pretty(entries)
        .map_err(|e| format!("History serialize failed: {e}"))?;
    std::fs::write(&path, json).map_err(|e| format!("History write failed: {e}"))
}

/// Add a new entry at the front (newest first), trimming the tail if
/// we've exceeded `MAX_ENTRIES`. Returns the newly-created entry's id
/// so the caller can reference it if needed.
///
/// Empty `raw` is rejected — we skip logging no-speech / canceled runs.
/// The frontend can still call this for post-format updates; those
/// callers shouldn't be passing empty raw either but defensive is cheap.
pub fn append(raw: &str, final_text: &str, format_type: &str) -> Result<u64, String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err("History: refusing to log empty transcript".to_string());
    }

    let mut entries = load();
    let id = now_millis();
    // If the clock resolution gave us a duplicate id (possible on very
    // fast back-to-back calls), bump by 1 ms until unique. Keeps the
    // UI's "use id as a key" assumption safe.
    let id = if entries.iter().any(|e| e.id == id) {
        let mut next = id + 1;
        while entries.iter().any(|e| e.id == next) {
            next += 1;
        }
        next
    } else {
        id
    };

    let entry = Entry {
        id,
        timestamp: id,
        raw: raw.to_string(),
        final_text: final_text.to_string(),
        format_type: format_type.to_string(),
    };
    entries.insert(0, entry);
    if entries.len() > MAX_ENTRIES {
        entries.truncate(MAX_ENTRIES);
    }
    save(&entries)?;
    Ok(id)
}

/// Remove a single entry by id. Missing id is not an error (delete of
/// something already gone is benign — maybe the user clicked delete
/// twice).
pub fn delete(id: u64) -> Result<(), String> {
    let mut entries = load();
    let before = entries.len();
    entries.retain(|e| e.id != id);
    if entries.len() == before {
        // Nothing removed; don't bother writing.
        return Ok(());
    }
    save(&entries)
}

/// Wipe all history.
pub fn clear() -> Result<(), String> {
    save(&[])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: u64, raw: &str, final_text: &str) -> Entry {
        Entry {
            id,
            timestamp: id,
            raw: raw.to_string(),
            final_text: final_text.to_string(),
            format_type: "none".to_string(),
        }
    }

    // These tests operate on pure functions and in-memory state — no disk
    // round-trip — so they're safe to run in parallel with each other
    // and with any real app instance.

    #[test]
    fn entry_serde_roundtrip() {
        let e = entry(1234, "raw text", "Final text.");
        let json = serde_json::to_string(&e).unwrap();
        let back: Entry = serde_json::from_str(&json).unwrap();
        assert_eq!(e, back);
    }

    #[test]
    fn entries_serde_roundtrip() {
        let list = vec![
            entry(3, "c raw", "c"),
            entry(2, "b raw", "b"),
            entry(1, "a raw", "a"),
        ];
        let json = serde_json::to_string(&list).unwrap();
        let back: Vec<Entry> = serde_json::from_str(&json).unwrap();
        assert_eq!(list, back);
    }

    #[test]
    fn malformed_json_does_not_panic() {
        // serde_json will happily refuse; make sure we don't
        let broken = "{not json";
        let result: Result<Vec<Entry>, _> = serde_json::from_str(broken);
        assert!(result.is_err());
    }
}
