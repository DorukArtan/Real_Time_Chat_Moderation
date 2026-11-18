@echo off
setlocal
set "MODERATION_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%MODERATION_PYTHON%" (
    echo Project Python was not found: "%MODERATION_PYTHON%"
    echo See README.md for environment setup.
    exit /b 1
)
pushd "%~dp0"
if errorlevel 1 exit /b 1
"%MODERATION_PYTHON%" "%~dp0app.py" %*
set "MODERATION_EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %MODERATION_EXIT_CODE%
