
@echo off
setlocal

cd /d "%~dp0"

if "%~1"=="" (
    echo ERROR: Dataset path not provided.
    echo Usage: RUN_STAGE3_WINDOWS.bat "PATH_TO_DATASET"
    exit /b 1
)

if not exist "%~1\" (
    echo ERROR: Dataset folder not found.
    exit /b 1
)

python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

python run_stage3.py --data-dir "%~1" --output-dir results
if errorlevel 1 exit /b 1

echo.
echo Stage 3 reproduction completed successfully.
endlocal
