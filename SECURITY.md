# Security Policy

## Reporting a Vulnerability

Open a [GitHub issue](https://github.com/yuanbaca/Dictator/issues/new) describing what you found. Include:

- A clear description of the issue and the impact
- Steps to reproduce
- Affected version(s) (e.g. `0.5.0`)
- Any logs, panics, or screenshots that would help

This is a single-maintainer hobby project, so I'm not running a private security inbox — public issues are fine and let other users see what's been reported. I'll get to it as I'm able and aim to ship fixes in the next release.

## Supported Versions

Dictator is a single-developer hobby project. Only the **latest published release** receives fixes. Older versions are not maintained.

## Security Model — what's actually shipping

This section describes the model in the current codebase, not aspirational design. Be skeptical of any older docs you find that contradict it.

### Threat model in scope

- A program on the same Windows account injecting unwanted text via Dictator's WebSocket — out of scope, since anything on the same account can already inject text directly.
- A device on the **same LAN** as the desktop trying to drive Dictator without permission — partially in scope (see below).
- A device on the **public internet** trying to reach Dictator — in scope; the desktop binds only to LAN/Tailscale interfaces and never to a public address.

### Network surface

By default, **the LAN phone server is off**. With it off, Dictator runs fully local: no inbound listener, no exposed ports, no surface for a network attacker.

When the LAN server is enabled in Settings, the desktop opens **port 3456** with a self-signed TLS cert and binds it to the local network interface only. The phone companion connects over HTTPS. The cert is regenerated on each install; there is no certificate pinning today.

When Tailscale is enabled, the desktop additionally opens **port 3457** with a Tailscale-issued cert. Trust comes from Tailscale itself: only devices on your tailnet can reach it, authenticated by Tailscale's identity layer.

### Mic-to-text path

- Audio capture, transcription (Whisper.cpp), wake-word detection (OpenWakeWord), and text formatting (llama.cpp) all run **locally on the user's machine**.
- No audio, no transcripts, and no telemetry are sent to any server.
- The only outbound network traffic is:
  - Initial model downloads from Hugging Face / GitHub on first use
  - The version-check Gist (a single GET on launch)
  - Optional cloud TTS APIs (Minimax) **only if the user explicitly enters an API key**

### What is NOT in place

A few things you might assume are present but aren't:

- **No formal pairing flow.** Anything that can reach port 3456 over LAN can talk to the WebSocket. Trust is "you control your LAN" / "you trust your tailnet." Earlier design docs described a 6-digit pairing code with session tokens; that was never implemented.
- **No per-device authentication tokens.** Connections are not authenticated beyond TLS.
- **No replay protection on control messages.**
- **No certificate pinning.**

If you need stronger guarantees, **don't enable the LAN server** — leave Dictator in its default fully-local mode and use the desktop hotkey or wake word instead. Tailscale's identity layer is the recommended path if you do need phone-to-PC over a network.

### Local data

Settings, the word-replacement bank, history, recorded wake-word state, and saved API keys are stored on disk under the install directory and the per-user app-data folder. None of this is encrypted at rest beyond Windows ACLs.

## Third-Party Components

Dictator bundles or links to several open-source projects, each with its own security posture and reporting channel:

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [OpenWakeWord](https://github.com/dscripka/openWakeWord)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Piper](https://github.com/rhasspy/piper)
- [Tauri](https://tauri.app/)

Vulnerabilities in those projects should be reported upstream.
