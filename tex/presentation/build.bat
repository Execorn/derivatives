@echo off
REM =============================================================================
REM  tex/presentation/build.bat
REM  Compile the defense presentation (presentation.tex) to PDF.
REM
REM  Strategy: run from script directory with -output-directory.
REM  The final PDF is MOVED here so it's not left in the cache.
REM
REM  Requirements: pdflatex + beamer (MiKTeX: https://miktex.org)
REM  Usage:  tex\presentation\build.bat
REM =============================================================================
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
REM Remove trailing backslash
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
for %%I in ("%SCRIPT_DIR%\..") do set "TEX_ROOT=%%~fI"
set "CACHE_DIR=%TEX_ROOT%\.latex_cache"
set "DOC=presentation"

if not exist "%CACHE_DIR%" mkdir "%CACHE_DIR%"
cd /d "%SCRIPT_DIR%"

echo [build] Compiling %DOC%.tex (defense slides) ...
pdflatex -interaction=nonstopmode -output-directory="%CACHE_DIR%" "%DOC%.tex"
pdflatex -interaction=nonstopmode -output-directory="%CACHE_DIR%" "%DOC%.tex"

move /Y "%CACHE_DIR%\%DOC%.pdf" "%SCRIPT_DIR%\%DOC%.pdf"
echo [build] Done  -^>  tex\presentation\%DOC%.pdf
