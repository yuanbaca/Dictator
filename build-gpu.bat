@echo off
echo ============================================
echo  DeskMic Dictation - GPU Build (Vulkan)
echo ============================================
echo.

:: Check for Vulkan SDK
if "%VULKAN_SDK%"=="" (
    echo ERROR: Vulkan SDK not found!
    echo.
    echo To enable GPU acceleration:
    echo   1. Download the Vulkan SDK from: https://vulkan.lunarg.com/sdk/home
    echo   2. Run the installer and accept defaults
    echo   3. Restart your terminal ^(the installer sets VULKAN_SDK automatically^)
    echo   4. Run this script again
    echo.
    echo Or use build.bat to build without GPU ^(CPU only^).
    pause
    exit /b 1
)

echo Vulkan SDK found at: %VULKAN_SDK%
echo.

:: Set build environment
set PATH=C:\Program Files\CMake\bin;%PATH%
set LIBCLANG_PATH=C:\Program Files\LLVM\bin

cd apps\desktop
echo Building with GPU acceleration (Vulkan)...
echo.
npx tauri build -- --features gpu

echo.
echo ============================================
echo Build complete! Run with: run.bat
echo ============================================
pause
