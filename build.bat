@echo off
echo ============================================
echo  Dictator - Release Build (GPU enabled)
echo ============================================
echo.

:: ── Prerequisites ────────────────────────────
:: 1. Vulkan SDK installed (vulkan.lunarg.com) — needed at build time only
:: 2. CMake on PATH
:: 3. LLVM/Clang for bindgen
::
:: GPU (Vulkan) is the DEFAULT feature. The exe will auto-detect GPU at
:: runtime and fall back to CPU if no Vulkan-capable GPU is found.
::
:: IMPORTANT: We use "NMake Makefiles" as the CMake generator because
:: VS2025 Build Tools (MSBuild v18) has a race condition with the
:: ExternalProject pattern used by llama.cpp's Vulkan shader compiler.
:: NMake builds serially and avoids this. Without this, the build fails
:: with "vulkan-shaders-gen" errors. See docs/build-notes.md for details.

:: Check for Vulkan SDK
if "%VULKAN_SDK%"=="" (
    echo ERROR: Vulkan SDK not found!
    echo.
    echo Install the Vulkan SDK from: https://vulkan.lunarg.com/sdk/home
    echo Then restart your terminal.
    pause
    exit /b 1
)

echo Vulkan SDK: %VULKAN_SDK%

:: Set build environment
set "PATH=C:\Program Files\CMake\bin;%PATH%"
set "LIBCLANG_PATH=C:\Program Files\LLVM\bin"

:: Use NMake to avoid MSBuild race condition (see comment above)
set "CMAKE_GENERATOR=NMake Makefiles"

:: Use a short target dir to avoid Windows MAX_PATH issues with deeply
:: nested Vulkan shader build paths
set "CARGO_TARGET_DIR=C:\b"

cd /d "%~dp0apps\desktop"
echo Building release with GPU acceleration (Vulkan)...
echo.
cargo build --release
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo BUILD FAILED. See errors above.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Build complete!
echo  Exe: C:\b\release\dictator.exe
echo ============================================
echo.
pause
