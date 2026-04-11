Dictator - What you say goes... (to your cursor)
==================================================

Voice-to-text for Windows. Dictate from your computer with a hotkey, or
from your phone as a wireless mic. Text appears wherever your cursor is.
All processing happens locally -- no cloud, no accounts, no data leaves
your network.


TWO WAYS TO DICTATE
-------------------

From your computer (hotkey):
  Press Ctrl+Shift+Space, speak into your PC mic, press again.
  Text appears at your cursor. Works in any app.

From your phone (wireless mic):
  Open the URL shown in the app on your phone's browser.
  Tap the mic button, speak, and text appears on your PC.


GETTING STARTED
---------------

1. Run dictator.exe
2. On first launch, pick a speech model to download (~50-500 MB)
3. Wait for the status to turn green ("Ready")
4. Press Ctrl+Shift+Space to dictate, or open the URL on your phone


FEATURES
--------

- Speech-to-text via Whisper (runs locally on your PC)
- Global hotkeys -- record, inject text, and read-aloud from any app
- AI text formatting (optional local LLM, downloads ~2 GB model)
- Neural text-to-speech with Piper voices (optional, 10+ voices)
- Windows SAPI text-to-speech (built-in, no download needed)
- System tray auto-format toggle (left-click to toggle, right-click
  to pick format: clean up, email, meeting notes, documentation, message)
- Escape key cancels recording or stops TTS (works globally)
- Phone companion with recording, key buttons, and read-aloud
- Auto-space after each dictation (toggleable)
- Multiple Whisper model sizes (tiny to large)
- GPU acceleration support (Vulkan, optional)


CONNECTION OPTIONS (phone)
--------------------------

LAN (same WiFi):
  Opens on port 3456. Your phone may show a security warning the first
  time -- accept it once and it won't ask again.

Tailscale (any network):
  Enable Tailscale in Dictator's Settings. Install Tailscale on both PC
  and phone. No browser warnings, works from anywhere. Port 3457.


SYSTEM REQUIREMENTS
-------------------

Minimum:  Windows 10/11, 8 GB RAM, any modern CPU (2018+)
Recommended:  16 GB RAM for all features (AI formatting + neural TTS)


TROUBLESHOOTING
---------------

"Loading whisper model..." stays yellow:
  Wait 10-20 seconds on first launch. The model takes time to load.

Phone can't connect:
  Make sure both devices are on the same WiFi network.
  Check that Windows Firewall allows dictator.exe.

No text appears after speaking:
  Click where you want text to go before recording.
  Make sure the status shows "Ready" (green dot).

Hotkey doesn't work:
  Check Settings to see/change the recording hotkey.
  Make sure another app isn't using the same key combination.


LICENSE
-------

Private software.
