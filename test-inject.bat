@echo off
set "PATH=C:\Program Files\CMake\bin;%PATH%"
set "LIBCLANG_PATH=C:\Program Files\LLVM\bin"

cd /d "%~dp0apps\desktop"

echo === DeskMic Text Injection Test ===
echo.
echo This will inject test text into whatever window you click into.
echo You'll have 5 seconds to click into a text field (Notepad, browser, etc).
echo.
pause

cargo run --release --bin test-inject -- --mode paste --delay 5 "Hello from DeskMic Dictation! If you can read this, text injection is working."
pause
