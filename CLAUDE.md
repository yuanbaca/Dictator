# Dictator - Development Guide

## What This Project Is

A desktop dictation app: speak into your phone, text appears at the cursor on your Windows PC. Phone connects over LAN or Tailscale. Desktop does transcription locally and injects text into the focused application.

## Stack

- **Desktop**: Tauri v2 + TypeScript frontend + Rust backend
- **Mobile**: Web app (served by desktop or standalone) -- React or vanilla TS
- **Transcription**: Faster-Whisper (Python sidecar) or whisper.cpp (Rust/C++)
- **Transport**: WebSocket (JSON control messages, binary audio frames)
- **Text injection**: Windows SendInput (typing) and clipboard paste via Win32 APIs

## Architecture Rules

- **Desktop is the authority.** It owns pairing, transcription, insertion, and security.
- **Mobile is a capture/control client only.** It captures audio and sends commands.
- **Transcription is abstracted behind a provider interface.** Never hardcode to one engine.
- **Text injection is abstracted behind an interface.** Support typing mode, paste mode, and auto mode with platform-specific backends.
- **Transport protocol lives in code** (`apps/desktop/src/server.rs`). The earlier `docs/protocol.md` design doc was removed — it described un-shipped pairing/token machinery and was misleading.

## Build Order (Critical)

When implementing, follow this priority order:

1. Prove text insertion works (typing + paste into Notepad/browser)
2. Prove phone-to-desktop audio transfer works
3. Prove local transcription works
4. Then polish UI and pairing

Text insertion is the hardest constraint. Prove it first.

## Implementation Rules

- Keep desktop authority strict -- mobile never decides where/how text is inserted
- Do not tightly couple transcription to one engine
- Keep transport protocol simple and documented
- Abstract text injection behind an interface
- Treat Windows as first-class; other platforms are future adapters
- Log every step of: audio receive -> transcribe -> inject
- Optimize for reliability before fancy UI

## Coding Standards

- Rust code: use `thiserror` for error types, `tokio` for async, `serde` for serialization
- TypeScript: strict mode, no `any` types
- All WebSocket message types defined in `packages/shared-types`

## Key Documentation

- `README.md` -- user-facing summary
- `SECURITY.md` -- shipping security model + reporting contact
- The `docs/` folder of internal planning / phase docs was removed before
  the repo went public — anything still relevant lives in README, this
  file, or in code comments.
