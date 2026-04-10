# Dictator

Speak into your phone, text appears at your cursor on Windows.

Your phone acts as a wireless microphone. Audio is sent to your PC over your local network (or Tailscale), transcribed locally using Whisper, and injected into whatever app has focus.

**All processing happens on your machine. No cloud services, no accounts, no data leaves your network.**

## How It Works

1. Run Dictator on your Windows PC
2. Open the URL shown in the app on your phone's browser
3. Tap the mic button on your phone and speak
4. Text appears at your cursor on your PC

## Features

- Local speech-to-text via [whisper.cpp](https://github.com/ggerganov/whisper.cpp) (no internet required)
- **AI text formatting** — optional local LLM (llama.cpp) cleans up, rephrases, or reformats your dictation before insertion. Multiple presets: clean up, professional, casual, bullet points, and more. Downloads a ~2 GB model on first use.
- **Neural text-to-speech** — highlight text anywhere and hear it read aloud using [Piper](https://github.com/rhasspy/piper) neural voices or Windows SAPI. Pause/resume with the same hotkey. 10+ downloadable voice models.
- Text injection via clipboard paste or simulated keystrokes
- Auto-space after each dictation (toggleable)
- Escape key cancels recording or stops TTS playback
- Configurable hotkeys for recording, text injection, and read-aloud
- Two connection options:
  - **LAN** (port 3456) -- self-signed cert, same WiFi, works immediately
  - **Tailscale** (port 3457) -- trusted cert, works from anywhere, no browser warnings
- Optional GPU acceleration via Vulkan
- Phone companion web app with recording, key buttons, and read-aloud trigger

## Requirements

### Build Requirements

- [Rust](https://rustup.rs/) (stable toolchain)
- [Node.js](https://nodejs.org/) (LTS, for Tauri CLI)
- [CMake](https://cmake.org/download/)
- [LLVM/Clang](https://releases.llvm.org/) (for bindgen -- install to `C:\Program Files\LLVM`)
- [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with "Desktop development with C++"
- Whisper model file: `models/ggml-base.en.bin` (download from the [whisper.cpp models page](https://huggingface.co/ggerganov/whisper.cpp/tree/main))

### Runtime Requirements

- Windows 10/11
- The whisper model file (`models/ggml-base.en.bin`) must be present relative to the executable
- Phone and PC on the same WiFi network (for LAN mode)
- Windows Firewall must allow inbound TCP on port 3456 (and 3457 if using Tailscale)

### Optional

- [Tailscale](https://tailscale.com/) on both PC and phone for trusted HTTPS and remote access
- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) for GPU-accelerated transcription

## Building

### CPU build (default)

```
build.bat
```

### GPU build (Vulkan)

```
build-gpu.bat
```

Requires the Vulkan SDK to be installed. The app will try GPU first and fall back to CPU automatically.

### Running

```
run.bat
```

Or directly: `apps\desktop\target\release\dictator.exe`

## Phone Setup

When you open the URL on your phone, the browser needs HTTPS to access the microphone. Dictator handles this automatically:

- **LAN mode**: Your phone will show a "not private" warning. Accept it once (iPhone: Show Details > visit this website; Android: Advanced > Proceed).
- **Tailscale mode**: No warnings at all. Install Tailscale on both devices, enable HTTPS certs in the Tailscale admin panel, and restart Dictator.

Full setup instructions are served at `/setup` on the companion server.

## Project Structure

```
apps/desktop/          Tauri desktop app (Rust backend + HTML frontend)
  src/main.rs          App entry point, Tauri commands
  src/server.rs        HTTPS server for phone companion
  src/transcription.rs Whisper integration
  src/tts.rs           Text-to-speech (SAPI + Piper neural voices)
  src/llm.rs           LLM text formatting via llama.cpp
  src/templates.rs     LLM format presets (clean up, professional, etc.)
  src/model_manager.rs Whisper model discovery and download
  src/tailscale.rs     Tailscale detection and cert generation
  ui/index.html        Desktop UI
  companion/           Phone companion web app
    index.html         Mic capture + WebSocket client
    setup.html         HTTPS setup guide
models/                Whisper model files (not committed)
```

## License

Private repository.
