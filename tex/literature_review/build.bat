@echo off
REM =============================================================================
REM  tex/literature_review/build.bat
REM  Compile the literature review (main.tex) with pdflatex + biber.
REM
REM  Strategy: run from script directory with -output-directory.
REM  The final PDF is MOVED here so it's not left in the cache.
REM
REM  Usage:  tex\literature_review\build.bat
REM =============================================================================
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
for %%I in ("%SCRIPT_DIR%\..") do set "TEX_ROOT=%%~fI"
set "CACHE_DIR=%TEX_ROOT%\.latex_cache"
set "DOC=main"

if not exist "%CACHE_DIR%" mkdir "%CACHE_DIR%"
cd /d "%SCRIPT_DIR%"

echo [build] Compiling %DOC%.tex (literature review) ...
pdflatex -interaction=nonstopmode -output-directory="%CACHE_DIR%" "%DOC%.tex"
biber --output-directory="%CACHE_DIR%" "%CACHE_DIR%\%DOC%"
pdflatex -interaction=nonstopmode -output-directory="%CACHE_DIR%" "%DOC%.tex"
pdflatex -interaction=nonstopmode -output-directory="%CACHE_DIR%" "%DOC%.tex"

move /Y "%CACHE_DIR%\%DOC%.pdf" "%SCRIPT_DIR%\%DOC%.pdf"
echo [build] Done  -^>  tex\literature_review\%DOC%.pdf
