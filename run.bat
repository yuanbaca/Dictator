@echo off
echo Starting DeskMic Dictation...
cd /d "%~dp0apps\desktop"
start "" target\release\deskmic.exe
