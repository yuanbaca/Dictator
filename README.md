# Dictator

**What you say goes... (to your cursor)**

Voice-to-text for Windows — dictate from your computer with a hotkey, or from your phone as a wireless mic. Text appears wherever your cursor is. All processing happens locally on your machine. No cloud, no accounts, no data leaves your network.

## Two Ways to Dictate

### From Your Computer (hotkey)
Press a hotkey (default: `Ctrl+Shift+Space`), speak into your PC's microphone, press again — text appears at your cursor. Works in any app: Notepad, Word, browser, chat, IDE, anywhere.

### From Your Phone (wireless mic)
Open the URL shown in the app on your phone's browser, tap the mic button and speak. Your phone acts as a wireless microphone over your local network, and text appears at your cursor on your PC.

## Features

- **Local speech-to-text** via [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — no internet required, multiple model sizes (tiny to large)
- **AI text formatting** — optional local LLM via [llama.cpp](https://github.com/ggerganov/llama.cpp) reformats dictated speech before insertion. 10 format presets across 3 cleanup tiers, 2 email styles, formal letter, bullet summary, meeting notes, documentation, and message. Downloads Phi-3 Mini (~2 GB) on first use. Supports custom GGUF models — drop them in `models/` and they auto-appear. Three chat template formats supported: Phi-3, Llama 3, and ChatML (Mistral/Qwen/Gemma). Format is auto-detected from filename or manually selectable in Settings. GGUF files are validated before loading.
- **Neural text-to-speech** — highlight text anywhere and hear it read aloud using [Piper](https://github.com/rhasspy/piper) neural voices or Windows SAPI. Pause/resume with the same hotkey. 10+ downloadable voice models.
- **Configurable hotkeys** — record, inject text, and read-aloud each have their own global hotkey. Works from any app without switching windows.
- **AI formatting from the system tray** — left-click the tray icon to toggle auto-format, right-click to pick a format preset (clean up, email, meeting notes, documentation, message)
- **Text injection** via clipboard paste or simulated keystrokes
- **Auto-space** after each dictation (toggleable)
- **Escape key** cancels recording or stops TTS playback — works globally, even when Dictator isn't focused
- **Phone companion** web app with recording, key buttons, and read-aloud trigger
- **Two connection options** for phone:
  - **LAN** (port 3456) — self-signed cert, same WiFi, works immediately
  - **Tailscale** (port 3457) — trusted cert, works from anywhere, no browser warnings
- **Optional GPU acceleration** via Vulkan
- **First-run model download** — the app downloads what it needs on first launch; no manual setup

## How It Works

1. Run Dictator on your Windows PC
2. **Desktop dictation**: Press `Ctrl+Shift+Space` (or your configured hotkey), speak, press again — text is inserted at your cursor
3. **Phone dictation**: Open the URL shown in the app on your phone's browser, tap the mic button, speak — text appears on your PC
4. **Read aloud**: Highlight text in any app, press the read-aloud hotkey — hear it spoken with neural TTS
5. **AI formatting**: Enable auto-format in Settings or via the tray icon — dictation is cleaned up by a local LLM before insertion

## Requirements

### Runtime
- Windows 10/11
- Phone and PC on the same WiFi network (for phone dictation via LAN)

### Build
- [Rust](https://rustup.rs/) (stable toolchain)
- [Node.js](https://nodejs.org/) (LTS, for Tauri CLI)
- [CMake](https://cmake.org/download/)
- [LLVM/Clang](https://releases.llvm.org/) (for bindgen — install to `C:\Program Files\LLVM`)
- [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with "Desktop development with C++"

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
- **Tailscale mode**: No warnings at all. Enable Tailscale in Dictator's Settings, and install Tailscale on both devices.

Full setup instructions are served at `/setup` on the companion server.

## Project Structure

```
apps/desktop/          Tauri desktop app (Rust backend + HTML frontend)
  src/main.rs          App entry point, Tauri commands, system tray
  src/server.rs        HTTPS server for phone companion
  src/transcription.rs Whisper integration
  src/tts.rs           Text-to-speech (SAPI + Piper neural voices)
  src/llm.rs           LLM text formatting via llama.cpp
  src/templates.rs     LLM format presets and chat templates (Phi-3, Llama 3, ChatML)
  src/model_manager.rs Whisper/LLM/Piper model discovery, download, and GGUF validation
  src/tailscale.rs     Tailscale detection and cert generation
  ui/index.html        Desktop UI
  companion/           Phone companion web app
    index.html         Mic capture + WebSocket client
    setup.html         HTTPS setup guide
models/                Model files (downloaded on first run, not committed)
```

## AI Models

Dictator uses two types of local AI models (no cloud, no API keys):

| Model | Purpose | Default | Size |
|-------|---------|---------|------|
| Whisper (whisper.cpp) | Speech-to-text | ggml-base.en.bin | ~142 MB |
| LLM (llama.cpp) | Text formatting | Phi-3 Mini Q4_K_M | ~2.4 GB |

The LLM is **optional** — it only downloads when you enable auto-format. Without it, raw transcription is inserted directly.

### Using your own models

- **Whisper:** Place any ggml Whisper `.bin` file in the `models/` folder. The app picks the first one it finds.
- **LLM:** Place any GGUF file in the `models/` folder. The app auto-detects Phi-3 and Llama 3 chat formats from the filename; other models use Llama 3 format by default. Q4_K_M quantizations work best.

## Uninstall

Open **Settings** > scroll to bottom > click **Remove all app data**. This deletes all models, certificates, settings, and the autostart registry entry.

After cleanup, close the app and delete the app folder. See [docs/uninstall.md](docs/uninstall.md) for manual cleanup steps.

## License

Private repository.
