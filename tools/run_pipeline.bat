@echo off
setlocal
set "PYTHON=D:\software\miniconda3\envs\rail\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
"%PYTHON%" "%~dp0run_pipeline.py" %*
if "%~1"=="" pause
