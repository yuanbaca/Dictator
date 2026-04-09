@echo off
echo === Building DeskMic Dictation ===
echo.

set "PATH=C:\Program Files\CMake\bin;%PATH%"
set "LIBCLANG_PATH=C:\Program Files\LLVM\bin"

cd /d "%~dp0apps\desktop"
npx tauri build
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo BUILD FAILED. See errors above.
    pause
    exit /b 1
)

echo.
echo === Build complete! ===
echo Binaries are in: apps\desktop\target\release\
echo.
pause
