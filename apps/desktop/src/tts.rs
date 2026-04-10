use std::sync::Mutex;
use tts::Tts;

/// Thread-safe wrapper around the TTS engine.
/// Lazy-loaded on first speak request, then kept alive for the session.
pub struct Speaker {
    engine: Mutex<Tts>,
}

impl Speaker {
    pub fn new() -> Result<Self, String> {
        let engine = Tts::default().map_err(|e| format!("Failed to init TTS: {e}"))?;
        Ok(Self {
            engine: Mutex::new(engine),
        })
    }

    /// Speak text aloud. If already speaking, interrupts the current speech.
    pub fn speak(&self, text: &str) -> Result<(), String> {
        let mut engine = self.engine.lock().unwrap();
        engine
            .speak(text, true)
            .map_err(|e| format!("TTS speak failed: {e}"))?;
        Ok(())
    }

    /// Stop any ongoing speech.
    pub fn stop(&self) -> Result<(), String> {
        let mut engine = self.engine.lock().unwrap();
        engine
            .stop()
            .map_err(|e| format!("TTS stop failed: {e}"))?;
        Ok(())
    }

    /// Check if currently speaking.
    pub fn is_speaking(&self) -> bool {
        let engine = self.engine.lock().unwrap();
        engine.is_speaking().unwrap_or(false)
    }

    /// Set speech rate (words per minute). Default varies by platform.
    pub fn set_rate(&self, rate: f32) -> Result<(), String> {
        let mut engine = self.engine.lock().unwrap();
        engine
            .set_rate(rate)
            .map_err(|e| format!("TTS set rate failed: {e}"))?;
        Ok(())
    }

    /// Get current speech rate.
    pub fn get_rate(&self) -> f32 {
        let engine = self.engine.lock().unwrap();
        engine.get_rate().unwrap_or(1.0)
    }

    /// List available voice names.
    pub fn list_voices(&self) -> Vec<String> {
        let engine = self.engine.lock().unwrap();
        engine
            .voices()
            .unwrap_or_default()
            .into_iter()
            .map(|v| v.name().to_string())
            .collect()
    }

    /// Set voice by name.
    pub fn set_voice(&self, name: &str) -> Result<(), String> {
        let mut engine = self.engine.lock().unwrap();
        let voices = engine.voices().map_err(|e| format!("TTS voices failed: {e}"))?;
        let voice = voices
            .into_iter()
            .find(|v| v.name() == name)
            .ok_or_else(|| format!("Voice not found: {name}"))?;
        engine
            .set_voice(&voice)
            .map_err(|e| format!("TTS set voice failed: {e}"))?;
        Ok(())
    }
}

/// Capture currently highlighted/selected text from any application.
/// Simulates Ctrl+C, waits briefly, then reads the clipboard.
pub fn capture_selected_text() -> Result<String, String> {
    use arboard::Clipboard;
    use windows::Win32::UI::Input::KeyboardAndMouse::{
        SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYBD_EVENT_FLAGS,
        KEYEVENTF_KEYUP, VIRTUAL_KEY,
    };

    let vk_control = VIRTUAL_KEY(0x11); // VK_CONTROL
    let vk_shift = VIRTUAL_KEY(0x10); // VK_SHIFT
    let vk_alt = VIRTUAL_KEY(0x12); // VK_MENU (Alt)
    let vk_c = VIRTUAL_KEY(0x43); // VK_C

    // Save current clipboard contents, then DROP the handle before simulating Ctrl+C.
    // Windows only allows one process to hold the clipboard open at a time.
    let old_text = {
        let mut cb = Clipboard::new().map_err(|e| format!("Clipboard error: {e}"))?;
        cb.get_text().unwrap_or_default()
    }; // cb dropped here — clipboard is free for Ctrl+C

    fn key_up(vk: VIRTUAL_KEY) -> INPUT {
        INPUT {
            r#type: INPUT_KEYBOARD,
            Anonymous: INPUT_0 {
                ki: KEYBDINPUT {
                    wVk: vk,
                    wScan: 0,
                    dwFlags: KEYEVENTF_KEYUP,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        }
    }

    fn key_down(vk: VIRTUAL_KEY) -> INPUT {
        INPUT {
            r#type: INPUT_KEYBOARD,
            Anonymous: INPUT_0 {
                ki: KEYBDINPUT {
                    wVk: vk,
                    wScan: 0,
                    dwFlags: KEYBD_EVENT_FLAGS(0),
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        }
    }

    // Release any modifier keys that might still be held from the hotkey combo,
    // then send a clean Ctrl+C, then release Ctrl.
    let inputs = [
        key_up(vk_shift),
        key_up(vk_alt),
        key_up(vk_control),
        key_down(vk_control),
        key_down(vk_c),
        key_up(vk_c),
        key_up(vk_control),
    ];

    unsafe {
        SendInput(&inputs, std::mem::size_of::<INPUT>() as i32);
    }

    // Wait for clipboard to update — 300ms is reliable across apps
    std::thread::sleep(std::time::Duration::from_millis(300));

    // Read clipboard, then restore the old contents
    let new_text = {
        let mut cb = Clipboard::new().map_err(|e| format!("Clipboard error: {e}"))?;
        let text = cb.get_text().unwrap_or_default();
        // Restore old clipboard so we don't clobber the user's copy buffer
        if text != old_text && !old_text.is_empty() {
            let _ = cb.set_text(&old_text);
        }
        text
    };

    if new_text.is_empty() {
        return Err("No text selected".to_string());
    }

    Ok(new_text)
}
