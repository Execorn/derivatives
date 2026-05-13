@echo off
REM =============================================================================
REM  build_all_tex.bat  —  Compile all LaTeX documents in the project (Windows)
REM
REM  Usage:  build_all_tex.bat
REM =============================================================================
setlocal

set "ROOT=%~dp0"

echo =================================================================
echo   Building all LaTeX documents
echo =================================================================

echo.
echo --- Literature Review ---
call "%ROOT%tex\literature_review\build.bat"

echo.
echo --- Defense Presentation ---
call "%ROOT%tex\presentation\build.bat"

echo.
echo =================================================================
echo   All documents compiled successfully.
echo   literature_review -^> tex\literature_review\main.pdf
echo   presentation      -^> tex\presentation\presentation.pdf
echo =================================================================
