//! User-editable word replacement bank for Whisper output.
//!
//! Whisper has consistent blind spots — proper nouns, domain terms, brand
//! names. "Claude" becomes "clod"; "GitHub" becomes "get hub"; names of
//! coworkers get garbled the same way every time. Rather than swap models
//! or fine-tune, the word bank lets the user keep a personal list of
//! `{from -> to}` corrections that get applied post-transcription.
//!
//! **Pipeline position:** applied inside `stop_and_transcribe`, right after
//! Whisper produces text and before returning to the frontend. Every
//! downstream consumer (LLM formatter, direct inject, auto-format) sees
//! corrected text. The reformat-selection hotkey operates on already-typed
//! text and deliberately doesn't re-run the word bank.
//!
//! **Matching:** case-insensitive, Unicode-aware whole-word boundaries
//! via the `regex` crate's `\b`. Replacement text is used verbatim (the
//! user picked the casing they wanted).
//!
//! **Storage:** JSON file in the models directory alongside
//! `chat_format_overrides.json`, following the same pattern.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// One replacement rule. `from` is what Whisper produced (or is likely to);
/// `to` is what the user wants to see in the final text.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub from: String,
    pub to: String,
}

fn word_bank_path() -> PathBuf {
    crate::model_manager::models_dir().join("word_bank.json")
}

/// Load the word bank from disk. Returns an empty Vec if the file is
/// missing, unreadable, or malformed — a broken word bank should never
/// block transcription.
pub fn load() -> Vec<Entry> {
    let path = word_bank_path();
    match std::fs::read_to_string(&path) {
        Ok(s) => serde_json::from_str(&s).unwrap_or_else(|e| {
            eprintln!("word_bank: malformed {} ({e}) — ignoring", path.display());
            Vec::new()
        }),
        Err(_) => Vec::new(),
    }
}

/// Save the word bank to disk, replacing the existing file. The UI calls
/// this after each edit — simpler than tracking individual add/remove/edit
/// operations, and the list is tiny.
pub fn save(entries: &[Entry]) -> Result<(), String> {
    let path = word_bank_path();
    let json = serde_json::to_string_pretty(entries)
        .map_err(|e| format!("Word bank serialize failed: {e}"))?;
    std::fs::write(&path, json)
        .map_err(|e| format!("Word bank write failed: {e}"))
}

/// Apply all replacement rules to `text` in order. Rules with empty `from`
/// (or whose regex can't compile — shouldn't happen since we escape the
/// pattern, but defensive) are silently skipped so a single bad entry
/// doesn't poison the whole bank.
pub fn apply(text: &str, entries: &[Entry]) -> String {
    if entries.is_empty() || text.is_empty() {
        return text.to_string();
    }

    let mut result = text.to_string();
    for entry in entries {
        let from = entry.from.trim();
        let to = entry.to.as_str();
        if from.is_empty() {
            continue;
        }

        // Whole-word matching via `\b` — but only on edges that are
        // actually word characters. A phrase ending in punctuation like
        // "u.s." can't satisfy a trailing `\b` (the final `.` isn't a
        // word char, so there's no word↔non-word transition), which
        // would make the rule never match. Inspect the edge chars and
        // add `\b` only where it's meaningful.
        let first_is_word = from.chars().next().is_some_and(is_word_char);
        let last_is_word = from.chars().next_back().is_some_and(is_word_char);
        let prefix = if first_is_word { r"\b" } else { "" };
        let suffix = if last_is_word { r"\b" } else { "" };
        let pattern = format!("{prefix}{}{suffix}", regex::escape(from));

        let re = match regex::RegexBuilder::new(&pattern)
            .case_insensitive(true)
            .build()
        {
            Ok(r) => r,
            Err(e) => {
                eprintln!("word_bank: skipping entry '{from}': {e}");
                continue;
            }
        };

        // NoExpand: the replacement is a literal string — don't let $1-style
        // references in the user's replacement text get interpreted.
        result = re
            .replace_all(&result, regex::NoExpand(to))
            .into_owned();
    }
    result
}

/// Matches Rust regex `\w`: alphanumerics (Unicode) + ASCII underscore.
/// Used to decide whether `\b` can fire on a given edge character.
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entries(pairs: &[(&str, &str)]) -> Vec<Entry> {
        pairs
            .iter()
            .map(|(f, t)| Entry {
                from: f.to_string(),
                to: t.to_string(),
            })
            .collect()
    }

    #[test]
    fn empty_bank_is_noop() {
        assert_eq!(apply("hello world", &[]), "hello world");
    }

    #[test]
    fn basic_replacement() {
        let bank = entries(&[("clod", "Claude")]);
        assert_eq!(apply("ask clod a question", &bank), "ask Claude a question");
    }

    #[test]
    fn case_insensitive_match() {
        let bank = entries(&[("clod", "Claude")]);
        assert_eq!(apply("Clod answered", &bank), "Claude answered");
        assert_eq!(apply("CLOD answered", &bank), "Claude answered");
    }

    #[test]
    fn whole_word_only() {
        let bank = entries(&[("clod", "Claude")]);
        // "clods" should NOT become "Claudes" — different word
        assert_eq!(apply("the clods gathered", &bank), "the clods gathered");
        // "unclod" should NOT be touched
        assert_eq!(apply("he is unclod", &bank), "he is unclod");
    }

    #[test]
    fn boundary_at_punctuation() {
        let bank = entries(&[("clod", "Claude")]);
        assert_eq!(apply("clod.", &bank), "Claude.");
        assert_eq!(apply("(clod)", &bank), "(Claude)");
        assert_eq!(apply("\"clod\"", &bank), "\"Claude\"");
    }

    #[test]
    fn multiple_rules_applied_in_order() {
        let bank = entries(&[("clod", "Claude"), ("get hub", "GitHub")]);
        assert_eq!(
            apply("ask clod about get hub", &bank),
            "ask Claude about GitHub"
        );
    }

    #[test]
    fn multi_word_from() {
        let bank = entries(&[("get hub", "GitHub")]);
        assert_eq!(apply("push to get hub", &bank), "push to GitHub");
    }

    #[test]
    fn empty_from_skipped() {
        let bank = entries(&[("", "replacement"), ("clod", "Claude")]);
        // Empty from shouldn't match anything or crash; clod should still work.
        assert_eq!(apply("clod", &bank), "Claude");
    }

    #[test]
    fn regex_metachars_in_from_are_literal() {
        // "u.s." should match literally — the dots shouldn't be regex wildcards
        let bank = entries(&[("u.s.", "US")]);
        assert_eq!(apply("the u.s. economy", &bank), "the US economy");
        // And it shouldn't match "uXsX"
        assert_eq!(apply("uxsx", &bank), "uxsx");
    }

    #[test]
    fn dollar_sign_in_replacement_is_literal() {
        // If user types "$1" as replacement, it should stay "$1" — not be
        // interpreted as a regex capture group reference.
        let bank = entries(&[("dollar", "$1")]);
        assert_eq!(apply("one dollar", &bank), "one $1");
    }
}
