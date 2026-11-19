@echo off
REM Inicia a plataforma "Humano ou IA?" (abre o navegador em http://127.0.0.1:7860)
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Criando ambiente virtual e instalando dependencias - somente na primeira vez...
  python -m venv .venv
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
  .venv\Scripts\python -m pip install -r requirements.txt
)
echo Carregando os modelos (pode levar alguns segundos)...
.venv\Scripts\python app.py
pause
