@echo off
REM =============================================================================
REM  setup_and_run.bat  —  Windows one-shot setup and pipeline launcher
REM
REM  1. Creates a Python venv at .venv\ (if absent)
REM  2. Installs all dependencies from src\requirements.txt
REM  3. Runs the full training pipeline
REM  4. Launches the Streamlit UI (logs -> logs\streamlit.log)
REM
REM  Usage:
REM      setup_and_run.bat              rem full pipeline + UI
REM      setup_and_run.bat --skip-train rem skip training
REM =============================================================================
setlocal enabledelayedexpansion

set "ROOT=%~dp0"
set "VENV=%ROOT%.venv"
set "PYTHON=%VENV%\Scripts\python.exe"
set "PIP=%VENV%\Scripts\pip.exe"
set "STREAMLIT=%VENV%\Scripts\streamlit.exe"

set SKIP_TRAIN=0
if "%~1"=="--skip-train" set SKIP_TRAIN=1

REM ── 1. Virtual environment ────────────────────────────────────────────────
if not exist "%VENV%\Scripts\activate.bat" (
    echo [setup] Creating virtual environment at .venv\ ...
    python -m venv "%VENV%"
)
call "%VENV%\Scripts\activate.bat"

REM ── 2. Dependencies ──────────────────────────────────────────────────────
echo [setup] Installing requirements ...
"%PIP%" install --upgrade pip --quiet
"%PIP%" install -r "%ROOT%src\requirements.txt" --quiet
echo [setup] Dependencies OK

REM ── 3. Pipeline ──────────────────────────────────────────────────────────
cd /d "%ROOT%"

if %SKIP_TRAIN%==0 (
    echo.
    echo [pipeline] Step 1/3 - Data loader ...
    "%PYTHON%" src\data_loader.py

    echo [pipeline] Step 2/3 - Training surrogate model (200 epochs^) ...
    "%PYTHON%" src\train.py --epochs 200

    echo [pipeline] Step 3/3 - Calibration smoke-test ...
    "%PYTHON%" src\calibrator.py
) else (
    echo [pipeline] Training skipped (--skip-train^)
)

REM ── 4. Streamlit UI ──────────────────────────────────────────────────────
if not exist "%ROOT%logs\" mkdir "%ROOT%logs\"
echo.
echo [ui] Starting Streamlit app at http://localhost:8501
echo [ui] Logs -^> logs\streamlit.log
start /B "" "%STREAMLIT%" run src\app.py --server.headless true --browser.gatherUsageStats false ^> "%ROOT%logs\streamlit.log" 2^>^&1 < NUL
echo [ui] Streamlit launched in background.
