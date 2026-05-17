@echo off
REM =============================================================================
REM  setup_and_run.bat  —  Windows one-shot setup and pipeline launcher
REM
REM  1. Creates a Python venv at .venv\ (if absent)
REM  2. Installs all dependencies from src\requirements.txt
REM  3. Runs the MLP surrogate pipeline (data loader -> train -> calibration test)
REM  4. Generates the LSTM sequence dataset (OU trajectories -> W surfaces)
REM  5. Trains the LSTM temporal dynamics model
REM  6. Launches the Streamlit UI (logs -> logs\streamlit.log)
REM
REM  Usage:
REM      setup_and_run.bat              rem full pipeline + UI
REM      setup_and_run.bat --skip-train rem skip all training
REM      setup_and_run.bat --skip-lstm  rem skip LSTM steps only (4-5)
REM =============================================================================
setlocal enabledelayedexpansion

set "ROOT=%~dp0"
set "VENV=%ROOT%.venv"
set "PYTHON=%VENV%\Scripts\python.exe"
set "PIP=%VENV%\Scripts\pip.exe"
set "STREAMLIT=%VENV%\Scripts\streamlit.exe"

set SKIP_TRAIN=0
set SKIP_LSTM=0
if "%~1"=="--skip-train" set SKIP_TRAIN=1
if "%~1"=="--skip-lstm"  set SKIP_LSTM=1
if "%~2"=="--skip-train" set SKIP_TRAIN=1
if "%~2"=="--skip-lstm"  set SKIP_LSTM=1

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

REM ── 3. MLP surrogate pipeline ─────────────────────────────────────────────
cd /d "%ROOT%"

if %SKIP_TRAIN%==0 (
    echo.
    echo [pipeline] Step 1/5 - Data loader (Total Variance transformation^) ...
    "%PYTHON%" src\data_loader.py

    echo [pipeline] Step 2/5 - Training MLP surrogate (200 epochs^) ...
    "%PYTHON%" src\train.py --epochs 200

    echo [pipeline] Step 3/5 - Calibration smoke-test (L-BFGS-B^) ...
    "%PYTHON%" src\calibrator.py
) else (
    echo [pipeline] Training skipped (--skip-train^)
)

REM ── 4-5. LSTM temporal dynamics pipeline ──────────────────────────────────
if %SKIP_TRAIN%==0 (
    if %SKIP_LSTM%==0 (
        echo.
        echo [lstm] Step 4/5 - Generating LSTM sequence dataset (102,000 windows^) ...
        "%PYTHON%" scripts\generate_seq_data.py

        echo [lstm] Step 5/5 - Training LSTM dynamics model (up to 200 epochs^) ...
        "%PYTHON%" src\train_seq.py --epochs 200
    ) else (
        echo [lstm] LSTM training skipped (--skip-lstm^)
    )
)

REM ── 6. Streamlit UI ──────────────────────────────────────────────────────
if not exist "%ROOT%logs\" mkdir "%ROOT%logs\"
echo.
echo [ui] Starting Streamlit app at http://localhost:8501
echo [ui] Logs -^> logs\streamlit.log
start /B "" "%STREAMLIT%" run src\app.py --server.headless true --browser.gatherUsageStats false ^> "%ROOT%logs\streamlit.log" 2^>^&1 ^< NUL
echo [ui] Streamlit launched in background.
