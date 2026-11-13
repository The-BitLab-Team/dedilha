@echo off
rem Abre o Dedilha no navegador (interface web local).
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo O Dedilha ainda nao foi instalado. De dois cliques em instalar.bat primeiro.
  echo.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -m dedilha.cli serve
echo.
pause
