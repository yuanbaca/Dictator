@echo off
set "PATH=C:\Program Files\CMake\bin;%PATH%"
set "LIBCLANG_PATH=C:\Program Files\LLVM\bin"

cd /d "%~dp0apps\desktop"
cargo run --release --bin dictate-mic 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Something went wrong. See errors above.
    pause
)
