//! Windows power event handling — proactively unload GPU models on system resume.
//!
//! When the PC sleeps or hibernates, the Vulkan context held by loaded models
//! can be invalidated. Using the model after resume often crashes from the C++
//! side (whisper.cpp or llama.cpp) in a way Rust's `Result` can't catch.
//!
//! This module subscribes to Windows' suspend/resume notifications via
//! `PowerRegisterSuspendResumeNotification`. On resume, we drop the loaded
//! models so they re-initialize with a fresh Vulkan context on next use.
//!
//! Layer 2 of the GPU resilience story — Layer 1 (`gpu_guard`) is the safety
//! net that catches crashes that slipped through. Layer 2 tries to prevent
//! those crashes in the first place.

#![cfg(windows)]

use crate::llm;
use crate::transcription;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Foundation::HANDLE;
use windows::Win32::System::Power::{
    PowerRegisterSuspendResumeNotification, DEVICE_NOTIFY_SUBSCRIBE_PARAMETERS,
};
use windows::Win32::UI::WindowsAndMessaging::{
    DEVICE_NOTIFY_CALLBACK, PBT_APMRESUMEAUTOMATIC, PBT_APMRESUMESUSPEND, PBT_APMSUSPEND,
};

/// Handles to the pieces of AppState we need to clear on resume.
/// Populated once at registration time, read inside the OS callback.
struct ResumeHandles {
    transcriber: Arc<Mutex<Option<transcription::Transcriber>>>,
    formatter: Arc<Mutex<Option<llm::Formatter>>>,
    using_gpu: Arc<Mutex<Option<bool>>>,
    llm_using_gpu: Arc<Mutex<Option<bool>>>,
    llm_status: Arc<Mutex<Option<bool>>>,
}

static HANDLES: OnceLock<ResumeHandles> = OnceLock::new();

/// Reset a `Mutex<Option<T>>` to `None`, tolerating a poisoned mutex.
/// Called from the OS power callback — must not panic.
fn clear_option<T>(m: &Arc<Mutex<Option<T>>>) {
    match m.lock() {
        Ok(mut g) => *g = None,
        Err(poisoned) => *poisoned.into_inner() = None,
    }
}

/// Callback invoked by the OS on power events. Runs on a system-provided thread.
///
/// Must not panic (UB in `extern "system"` fn across the FFI boundary), and
/// should return quickly — the OS is waiting on us to acknowledge the event.
unsafe extern "system" fn on_power_event(
    _context: *const c_void,
    event_type: u32,
    _setting: *const c_void,
) -> u32 {
    // Wrap everything in catch_unwind — we cannot let a panic propagate into
    // the OS callback path.
    let _ = std::panic::catch_unwind(|| match event_type {
        PBT_APMSUSPEND => {
            eprintln!("power_events: system entering sleep");
            // The models are still "valid" at this moment; no action needed.
            // We unload on resume instead, when the Vulkan context is known
            // to be potentially stale.
        }
        PBT_APMRESUMESUSPEND | PBT_APMRESUMEAUTOMATIC => {
            if let Some(h) = HANDLES.get() {
                // Only unload if models were actually on GPU — CPU-loaded
                // models don't have a Vulkan context to go stale, so forcing
                // a reload there just slows down the next dictation.
                let whisper_on_gpu = matches!(h.using_gpu.lock().ok().as_deref(), Some(&Some(true)));
                let llm_on_gpu = matches!(h.llm_using_gpu.lock().ok().as_deref(), Some(&Some(true)));

                if whisper_on_gpu || llm_on_gpu {
                    eprintln!(
                        "power_events: system resumed (event 0x{event_type:x}) — unloading GPU models"
                    );
                    clear_option(&h.transcriber);
                    clear_option(&h.formatter);
                    clear_option(&h.using_gpu);
                    clear_option(&h.llm_using_gpu);
                    clear_option(&h.llm_status);
                } else {
                    eprintln!(
                        "power_events: system resumed (event 0x{event_type:x}) — models on CPU, no unload needed"
                    );
                }
            }
        }
        _ => {
            // Other power events (battery state, power scheme changes) —
            // not relevant to GPU context freshness.
        }
    });
    0 // S_OK / NO_ERROR
}

/// Register for Windows suspend/resume notifications.
///
/// The OS will invoke our callback on sleep and resume events for as long as
/// the process lives (we leak the registration handle intentionally).
///
/// Returns `Err` if registration fails — callers should log and continue. The
/// crash-marker guard (`gpu_guard`) still protects against crash loops even
/// without proactive notifications.
pub fn register(
    transcriber: Arc<Mutex<Option<transcription::Transcriber>>>,
    formatter: Arc<Mutex<Option<llm::Formatter>>>,
    using_gpu: Arc<Mutex<Option<bool>>>,
    llm_using_gpu: Arc<Mutex<Option<bool>>>,
    llm_status: Arc<Mutex<Option<bool>>>,
) -> Result<(), String> {
    let handles = ResumeHandles {
        transcriber,
        formatter,
        using_gpu,
        llm_using_gpu,
        llm_status,
    };
    HANDLES
        .set(handles)
        .map_err(|_| "power_events: already registered".to_string())?;

    // When `flags = DEVICE_NOTIFY_CALLBACK`, the `recipient` HANDLE is actually
    // a pointer to a DEVICE_NOTIFY_SUBSCRIBE_PARAMETERS struct holding our
    // callback function pointer. Leak the struct so it outlives registration —
    // the OS keeps the callback pointer alive beyond the call itself.
    let params = Box::leak(Box::new(DEVICE_NOTIFY_SUBSCRIBE_PARAMETERS {
        Callback: Some(on_power_event),
        Context: std::ptr::null_mut(),
    }));

    let recipient = HANDLE(params as *mut _ as *mut c_void);
    let mut reg_handle: *mut c_void = std::ptr::null_mut();

    let result = unsafe {
        PowerRegisterSuspendResumeNotification(
            DEVICE_NOTIFY_CALLBACK,
            recipient,
            &mut reg_handle,
        )
    };

    // `WIN32_ERROR(0)` is ERROR_SUCCESS.
    if result.0 != 0 {
        return Err(format!(
            "PowerRegisterSuspendResumeNotification failed: 0x{:x}",
            result.0
        ));
    }

    // `reg_handle` is a raw pointer (Copy) — the OS already holds the real
    // registration internally. We don't need to retain the handle: we never
    // unregister, and Windows cleans up automatically on process exit.
    let _ = reg_handle;
    eprintln!("power_events: registered for suspend/resume notifications");
    Ok(())
}
