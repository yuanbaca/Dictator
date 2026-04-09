//! Text injection module for Windows.
//!
//! Supports two strategies:
//! - **Typing**: Simulates keystrokes via SendInput (broad compat, slower)
//! - **Paste**: Copies to clipboard + Ctrl+V (fast, touches clipboard)

use anyhow::{Context, Result};
use std::thread;
use std::time::Duration;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYBD_EVENT_FLAGS,
    KEYEVENTF_KEYUP, KEYEVENTF_UNICODE, VIRTUAL_KEY, VK_CONTROL, VK_V,
};

/// Which strategy to use for text insertion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InjectionMode {
    /// Simulate keystrokes one character at a time via SendInput.
    Type,
    /// Copy text to clipboard, paste with Ctrl+V, restore previous clipboard.
    Paste,
}

/// Insert text into the currently focused window.
pub fn inject_text(text: &str, mode: InjectionMode) -> Result<()> {
    match mode {
        InjectionMode::Type => inject_via_typing(text),
        InjectionMode::Paste => inject_via_paste(text),
    }
}

/// Simulate typing each character using SendInput with KEYEVENTF_UNICODE.
fn inject_via_typing(text: &str) -> Result<()> {
    for ch in text.chars() {
        // Newlines: send Enter key instead of unicode newline
        if ch == '\n' {
            send_vk_key(VIRTUAL_KEY(0x0D))?; // VK_RETURN
            continue;
        }
        if ch == '\r' {
            continue; // skip carriage returns, handle \n above
        }

        let code = ch as u16;

        // Key down
        let down = INPUT {
            r#type: INPUT_KEYBOARD,
            Anonymous: INPUT_0 {
                ki: KEYBDINPUT {
                    wVk: VIRTUAL_KEY(0),
                    wScan: code,
                    dwFlags: KEYEVENTF_UNICODE,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        };

        // Key up
        let up = INPUT {
            r#type: INPUT_KEYBOARD,
            Anonymous: INPUT_0 {
                ki: KEYBDINPUT {
                    wVk: VIRTUAL_KEY(0),
                    wScan: code,
                    dwFlags: KEYEVENTF_UNICODE | KEYEVENTF_KEYUP,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        };

        let inputs = [down, up];
        let sent = unsafe { SendInput(&inputs, size_of::<INPUT>() as i32) };
        if sent != 2 {
            anyhow::bail!("SendInput failed for char '{ch}' (U+{code:04X})");
        }

        // Small delay between keystrokes to avoid overwhelming target apps
        thread::sleep(Duration::from_millis(5));
    }

    Ok(())
}

/// Insert text by copying to clipboard, pasting with Ctrl+V, then restoring
/// the previous clipboard contents.
fn inject_via_paste(text: &str) -> Result<()> {
    let mut clipboard = arboard::Clipboard::new()
        .context("Failed to open clipboard")?;

    // Save current clipboard text (best-effort)
    let previous = clipboard.get_text().ok();

    // Set our text
    clipboard
        .set_text(text.to_string())
        .context("Failed to set clipboard text")?;

    // Small delay to let clipboard settle
    thread::sleep(Duration::from_millis(50));

    // Send Ctrl+V
    send_ctrl_v()?;

    // Wait for paste to complete
    thread::sleep(Duration::from_millis(100));

    // Restore previous clipboard contents (best-effort)
    if let Some(prev) = previous {
        // Small delay before restoring
        thread::sleep(Duration::from_millis(50));
        let _ = clipboard.set_text(prev);
    }

    Ok(())
}

/// Send Ctrl+V keystroke via SendInput.
fn send_ctrl_v() -> Result<()> {
    let ctrl_down = make_key_input(VK_CONTROL, false);
    let v_down = make_key_input(VK_V, false);
    let v_up = make_key_input(VK_V, true);
    let ctrl_up = make_key_input(VK_CONTROL, true);

    let inputs = [ctrl_down, v_down, v_up, ctrl_up];
    let sent = unsafe { SendInput(&inputs, size_of::<INPUT>() as i32) };
    if sent != 4 {
        anyhow::bail!("SendInput failed for Ctrl+V");
    }
    Ok(())
}

/// Send a single virtual key press and release.
fn send_vk_key(vk: VIRTUAL_KEY) -> Result<()> {
    let down = make_key_input(vk, false);
    let up = make_key_input(vk, true);
    let inputs = [down, up];
    let sent = unsafe { SendInput(&inputs, size_of::<INPUT>() as i32) };
    if sent != 2 {
        anyhow::bail!("SendInput failed for VK {:?}", vk);
    }
    Ok(())
}

/// Build an INPUT struct for a virtual key event.
fn make_key_input(vk: VIRTUAL_KEY, key_up: bool) -> INPUT {
    let flags = if key_up {
        KEYEVENTF_KEYUP
    } else {
        KEYBD_EVENT_FLAGS(0)
    };
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: vk,
                wScan: 0,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: 0,
            },
        },
    }
}
