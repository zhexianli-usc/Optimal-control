@echo off
REM Run Burgers data generation (and optional plot) using conda env "optimal-control".
REM From repo root, open "Anaconda Prompt" or ensure conda is in PATH, then:
REM   cd burgers_bc_data
REM   run_with_conda.bat
REM Or from repo root:  burgers_bc_data\run_with_conda.bat

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

call conda activate optimal-control 2>nul
if errorlevel 1 (
  echo Conda env "optimal-control" not found. Create it from repo root with:
  echo   conda env create -f environment.yml
  echo Then run this script again.
  exit /b 1
)

python generate_burgers_bc_data.py --samples 50 --out burgers_bc_data.npz --plot
exit /b 0
