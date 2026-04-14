//! GPU crash detection via marker file.
//!
//! Vulkan context can be invalidated by GPU driver crashes, sleep/resume
//! cycles, or driver updates. When that happens, loading or running a model
//! on GPU can cause a hard abort from the C++ side (llama.cpp or whisper.cpp)
//! that Rust's `Result` won't catch — the process just dies.
//!
//! We guard against a crash loop with a marker file:
//!   1. Before attempting any GPU load, we `arm()` — write the marker.
//!   2. The marker stays on disk for the entire session.
//!   3. On graceful quit (tray Quit menu), we `disarm()` — delete it.
//!   4. On next startup, `init()` checks for the marker. If present, the
//!      previous session crashed or was force-killed while GPU was active,
//!      so we skip GPU for this session. The UI surfaces a "Try GPU Again"
//!      affordance so the user can re-enable it on demand.
//!
//! False positives (user force-kills via Task Manager) cost one session of
//! CPU-only operation and one click to recover. That's an acceptable price
//! for protection against the real crash loop.
//!
//! Lives next to the exe for portable-install compatibility.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

const MARKER_FILENAME: &str = ".gpu_crash.lock";

/// Whether the marker was present at startup (i.e. previous session didn't
/// disarm cleanly). Sticky for the life of the process — used by the UI.
static CRASH_RECOVERED: AtomicBool = AtomicBool::new(false);

/// Whether GPU should be skipped for this session. Starts equal to
/// `CRASH_RECOVERED` and can be cleared by the user via `allow_gpu_retry()`.
static SESSION_DISABLED: AtomicBool = AtomicBool::new(false);

/// Path to the crash-marker file (next to the exe, like everything else).
pub fn marker_path() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            return exe_dir.join(MARKER_FILENAME);
        }
    }
    PathBuf::from(MARKER_FILENAME)
}

/// Call ONCE at startup, before any GPU work. Reads the marker state, stashes
/// it, and clears the marker so a clean session can arm/disarm fresh.
pub fn init() {
    let detected = marker_path().exists();
    CRASH_RECOVERED.store(detected, Ordering::SeqCst);
    SESSION_DISABLED.store(detected, Ordering::SeqCst);
    if detected {
        eprintln!("gpu_guard: crash marker detected from previous session — GPU disabled");
        let _ = std::fs::remove_file(marker_path());
    }
}

/// True if we detected a leftover marker at startup. Sticky for the session,
/// even after the user clicks "Try GPU Again" — the UI can use this to show
/// a persistent note until the app is next restarted.
pub fn crash_recovered() -> bool {
    CRASH_RECOVERED.load(Ordering::SeqCst)
}

/// True if GPU should be skipped for this session (because of a detected crash).
/// Cleared by `allow_gpu_retry()`.
pub fn session_disabled() -> bool {
    SESSION_DISABLED.load(Ordering::SeqCst)
}

/// User-initiated: clear the session-disabled flag so GPU will be attempted
/// again on the next model load. Called from the `retry_gpu` Tauri command.
pub fn allow_gpu_retry() {
    SESSION_DISABLED.store(false, Ordering::SeqCst);
    // Also delete the marker proactively in case something already re-armed it.
    let path = marker_path();
    if path.exists() {
        let _ = std::fs::remove_file(&path);
    }
    eprintln!("gpu_guard: GPU retry allowed — next model load will attempt GPU");
}

/// Write the marker before attempting a GPU model load. Safe to call repeatedly.
pub fn arm() {
    let path = marker_path();
    if let Err(e) = std::fs::write(&path, b"gpu-in-use") {
        eprintln!("gpu_guard: failed to write marker at {}: {e}", path.display());
    }
}

/// Remove the marker. Called on graceful shutdown and after soft GPU failures
/// (Rust-level `Err`, which means Vulkan itself is fine).
pub fn disarm() {
    let path = marker_path();
    if path.exists() {
        if let Err(e) = std::fs::remove_file(&path) {
            eprintln!("gpu_guard: failed to remove marker at {}: {e}", path.display());
        }
    }
}
