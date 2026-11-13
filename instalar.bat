@echo off
rem Instala o Dedilha (cria a pasta .venv e baixa as dependencias).
chcp 65001 >nul
cd /d "%~dp0"

set "PY="
for %%V in (3.12 3.13 3.11) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
  )
)
if not defined PY set "PY=python"

echo Usando: %PY%
%PY% install.py %*
echo.
pause
