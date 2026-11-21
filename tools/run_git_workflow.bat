@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_git_workflow.ps1"
set "rail_workflow_exit_code=%errorlevel%"
if not "%rail_workflow_exit_code%"=="0" pause
exit /b %rail_workflow_exit_code%
