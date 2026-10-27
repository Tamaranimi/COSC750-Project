@echo off
setlocal
python -m pip install -r requirements.txt
python run_stage3.py --data-dir "%USERPROFILE%\Downloads\archive" --output-dir results
if errorlevel 1 (
  echo.
  echo Stage 3 run failed. Read the error above before using any results.
  exit /b 1
)
echo.
echo Done. Send results\baseline_summary.csv and results\baseline_table_rows.tex back to ChatGPT to insert the actual numbers into the paper and slides.
endlocal
