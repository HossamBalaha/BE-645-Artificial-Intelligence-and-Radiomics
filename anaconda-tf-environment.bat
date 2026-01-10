@echo off
REM anaconda-tf-environment.bat
REM Improved helper to create a Conda environment and install Python deps on Windows.
REM Features:
REM  - Usage: anaconda-tf-environment.bat [env_name] [python_version] [--no-gpu] [--force] [--no-pause] [--silent] [--help] [--clean-log]
REM  - Defaults: env_name=be645, python_version=3.10
REM  - Detects NVIDIA GPU via nvidia-smi and (by default) installs cudatoolkit/cuDNN from conda-forge
REM  - Can skip GPU install with --no-gpu
REM  - --force removes an existing environment before creating it
REM  - Writes a log file anaconda-tf-environment.log next to this script
REM  - --silent suppresses console output but still writes to the log file
REM  - --clean-log generates a cleaned version of the log without encoding artifacts

:: ----------------------
:: Default settings (change these if you like)
set "ENV_NAME=be645"
set "PY_VER=3.10"
set "DO_GPU=1"
set "FORCE_REMOVE=0"
set "PAUSE_ON_EXIT=1"
set "LOGFILE=%~dp0anaconda-tf-environment.log"
set "SILENT=0"
set "CLEAN_LOG=0"

:: Startup diagnostic (logs command-line args) and ensure logfile exists
echo ======= anaconda-tf-environment log ======= > "%LOGFILE%"
echo Script invoked: %~nx0 %* >> "%LOGFILE%"
echo Starting script: %~nx0 %* at %DATE% %TIME% >> "%LOGFILE%"

:: Call the main entrypoint with all arguments to avoid accidental fall-through into labels
call :main %*
exit /b %ERRORLEVEL%

:: ----------------------
:: Helper functions (labels)

:usage
echo.
echo Usage: %~nx0 [env_name] [python_version] [--no-gpu] [--force] [--no-pause] [--silent] [--help] [--clean-log]
echo.
echo Examples:
echo   %~nx0                     ^> create default env 'be645' with Python 3.10 (interactive)
echo   %~nx0 myenv 3.9           ^> create 'myenv' with Python 3.9
echo   %~nx0 myenv 3.10 --no-gpu  ^> create without attempting GPU installs
echo   %~nx0 myenv 3.10 --force    ^> force-remove existing 'myenv' then create
echo.
echo Flags:
echo   --no-gpu     Skip installing CUDA/cuDNN packages
echo   --force      Remove any existing environment with the same name before creating
echo   --no-pause   Do not pause at the end (useful for automation)
echo   --silent     Suppress console output, log only
echo   --clean-log  Generate a cleaned log file without garbled characters
echo   --help       Show this help message
echo.
echo Log file: %LOGFILE%
echo.
goto :eof

:: ----------------------
:: Simple logging helper (appends with timestamp)
:log
echo [%DATE% %TIME%] %*  >> "%LOGFILE%"
if "%SILENT%"=="1" (
  REM silent mode: do not print to console
) else (
  echo %*
)
goto :eof

:: ----------------------
:: Main entry point (argument parser follows)
:: ----------------------
:main
setlocal EnableDelayedExpansion
set "ARG_IDX=0"
set "SKIPPED_CREATE=0"

:parse_args
if "%~1"=="" goto args_parsed
set /a ARG_IDX+=1
if "%~1"=="--help" (call :usage & exit /b 0)
if "%~1"=="--no-gpu" (set "DO_GPU=0" & shift & goto parse_args)
if "%~1"=="--force" (set "FORCE_REMOVE=1" & shift & goto parse_args)
if "%~1"=="--no-pause" (set "PAUSE_ON_EXIT=0" & shift & goto parse_args)
if "%~1"=="--silent" (set "SILENT=1" & shift & goto parse_args)
if "%~1"=="--quiet" (set "SILENT=1" & shift & goto parse_args)
if "%~1"=="--clean-log" (set "CLEAN_LOG=1" & shift & goto parse_args)
if "%ARG_IDX%"=="1" (set "ENV_NAME=%~1" & shift & goto parse_args)
if "%ARG_IDX%"=="2" (set "PY_VER=%~1" & shift & goto parse_args)
call :log "WARNING: Ignoring unknown argument: %~1"
shift
goto parse_args
:args_parsed
endlocal & set "ENV_NAME=%ENV_NAME%" & set "PY_VER=%PY_VER%" & set "DO_GPU=%DO_GPU%" ^
         & set "FORCE_REMOVE=%FORCE_REMOVE%" & set "PAUSE_ON_EXIT=%PAUSE_ON_EXIT%" ^
         & set "SILENT=%SILENT%" & set "CLEAN_LOG=%CLEAN_LOG%"

:: Start log
echo ======= anaconda-tf-environment log ======= >> "%LOGFILE%"
echo Script started at %DATE% %TIME% >> "%LOGFILE%"
call :log "Requested env: %ENV_NAME%, Python: %PY_VER%, GPU install: %DO_GPU%, Force remove: %FORCE_REMOVE%, Pause: %PAUSE_ON_EXIT%, Silent: %SILENT%"

:: Check conda availability
where conda >nul 2>&1
if errorlevel 1 (
  call :log "ERROR: 'conda' not found in PATH. Please run this script from an Anaconda Prompt or run 'conda init' and restart your shell."
  if "%PAUSE_ON_EXIT%"=="1" pause
  exit /b 1
)
call :log "Conda found on PATH"

:: Update conda in base environment first (non-fatal; streamed to log)
call :log "Updating conda (base) to latest from defaults..."
if "%SILENT%"=="1" (
  powershell -NoProfile -Command "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; (& conda update -n base -c defaults conda -y --json) 2>&1 | Out-File -FilePath \"%LOGFILE%\" -Append -Encoding utf8; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }"
) else (
  powershell -NoProfile -Command "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; (& conda update -n base -c defaults conda -y --json) 2>&1 | Tee-Object -FilePath \"%LOGFILE%\" -Append; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }"
)
if errorlevel 1 (
  call :log "WARNING: 'conda update' failed or encountered errors. Continuing; you may retry manually with 'conda update -n base -c defaults conda'."
) else (
  call :log "Updated conda successfully."
)

:: Ensure requirements.txt exists next to script
if not exist "%~dp0requirements.txt" (
  call :log "ERROR: requirements.txt not found at %~dp0requirements.txt"
  call :log "Please place requirements.txt next to this script and re-run."
  if "%PAUSE_ON_EXIT%"=="1" pause
  exit /b 1
)
call :log "Found requirements.txt"

:: Check if environment exists
call conda env list | findstr /C:"%ENV_NAME%" >nul 2>&1
set "ENV_EXISTS=%ERRORLEVEL%"
if %ENV_EXISTS% EQU 0 (
  call :log "Environment '%ENV_NAME%' already exists."
  if "%FORCE_REMOVE%"=="1" (
    call :log "--force specified: removing existing environment '%ENV_NAME%'. This may take a while..."
    if "%SILENT%"=="1" (
      powershell -NoProfile -Command "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; (conda env remove -n \"%ENV_NAME%\" -y --json) 2>&1 | Out-File -FilePath \"%LOGFILE%\" -Append -Encoding utf8; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }"
    ) else (
      powershell -NoProfile -Command "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; (conda env remove -n \"%ENV_NAME%\" -y --json) 2>&1 | Tee-Object -FilePath \"%LOGFILE%\" -Append; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }"
    )
    if errorlevel 1 (
      call :log "ERROR: Failed to remove existing environment '%ENV_NAME%'. See log for details."
      if "%PAUSE_ON_EXIT%"=="1" pause
      exit /b 1
    ) else (
      call :log "Removed existing environment '%ENV_NAME%'."
      set "ENV_EXISTS=1"
      call :log "Environment flag updated: ENV_EXISTS=%ENV_EXISTS% (will create new environment)."
      call :log "Creating Conda environment '%ENV_NAME%' with Python %PY_VER%' (immediate after --force removal)..."
      if "%SILENT%"=="1" (
        powershell -NoProfile -Command "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; (conda create -n \"%ENV_NAME%\" python=%PY_VER% -y --json) 2>&1 | Out-File -FilePath \"%LOGFILE%\" -Append -Encoding utf8; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }"
      ) else (
        powershell -NoProfile -Command "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; (conda create -n \"%ENV_NAME%\" python=%PY_VER% -y --json) 2>&1 | Tee-Object -FilePath \"%LOGFILE%\" -Append; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }"
      )
      if errorlevel 1 (
        call :log "ERROR: Failed to create Conda environment '%ENV_NAME%' after removal. See %LOGFILE% for details."
        if "%PAUSE_ON_EXIT%"=="1" pause
        exit /b 1
      ) else (
        call :log "Created environment '%ENV_NAME%' (after removal)."
      )
      set "SKIPPED_CREATE=1"
    )
  ) else (
    call :log "Existing environment will be reused. To recreate it, rerun with --force."
  )
) else (
  call :log "Environment '%ENV_NAME%' does not exist yet. Will create it."
)

:: Create environment if needed
if "%SKIPPED_CREATE%"=="1" (
  call :log "Environment was created during --force removal. Skipping standard creation block."
) else (
  if %ENV_EXISTS% NEQ 0 (
    call :log "Creating Conda environment '%ENV_NAME%' with Python %PY_VER%..."
    if "%SILENT%"=="1" (
      powershell -NoProfile -Command "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; (conda create -n \"%ENV_NAME%\" python=%PY_VER% -y --json) 2>&1 | Out-File -FilePath \"%LOGFILE%\" -Append -Encoding utf8; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }"
    ) else (
      powershell -NoProfile -Command "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; (conda create -n \"%ENV_NAME%\" python=%PY_VER% -y --json) 2>&1 | Tee-Object -FilePath \"%LOGFILE%\" -Append; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }"
    )
    if errorlevel 1 (
      call :log "ERROR: Failed to create Conda environment '%ENV_NAME%'. See %LOGFILE% for details."
      if "%PAUSE_ON_EXIT%"=="1" pause
      exit /b 1
    ) else (
      call :log "Created environment '%ENV_NAME%'."
    )
  ) else (
    call :log "Skipping creation (environment exists and --force not used)."
  )
)

:: Activate environment
call :log "Activating environment '%ENV_NAME%'..."
call :log "Running commands inside '%ENV_NAME%' via 'conda run' (no shell activation required)."

:: GPU detection and optional install
if "%DO_GPU%"=="1" (
  where nvidia-smi >nul 2>&1
  if not errorlevel 1 (
    call :log "NVIDIA GPU detected (nvidia-smi available). Attempting to install cudatoolkit and cudnn from conda-forge..."
    if "%SILENT%"=="1" (
      powershell -NoProfile -Command "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; (conda install -n \"%ENV_NAME%\" -c conda-forge cudatoolkit cudnn -y --json) 2>&1 | Out-File -FilePath \"%LOGFILE%\" -Append -Encoding utf8; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }"
    ) else (
      powershell -NoProfile -Command "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; (conda install -n \"%ENV_NAME%\" -c conda-forge cudatoolkit cudnn -y --json) 2>&1 | Tee-Object -FilePath \"%LOGFILE%\" -Append; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }"
    )
    if errorlevel 1 (
      call :log "WARNING: cudatoolkit/cuDNN install failed or packages not available. See log for details. Proceeding with Python package install."
    ) else (
      call :log "Installed cudatoolkit/cuDNN successfully from 'conda-forge'."
    )
  ) else (
    call :log "No NVIDIA GPU detected (nvidia-smi not found). Skipping GPU package installation."
  )
) else (
  call :log "--no-gpu specified: skipping GPU package installation."
)

:: Upgrade pip and install Python packages from requirements.txt inside the created env
call :log "Preparing to install Python packages from requirements.txt (install all via pip inside the env)..."
call :log "Upgrading pip inside '%ENV_NAME%' (via conda run)..."
if "%SILENT%"=="1" (
  powershell -NoProfile -Command "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; (& conda run -n \"%ENV_NAME%\" python -m pip install --upgrade pip --progress-bar off) 2>&1 | Out-File -FilePath \"%LOGFILE%\" -Append -Encoding utf8; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }"
) else (
  powershell -NoProfile -Command "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; (& conda run -n \"%ENV_NAME%\" python -m pip install --upgrade pip --progress-bar off) 2>&1 | Tee-Object -FilePath \"%LOGFILE%\" -Append; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }"
)
if errorlevel 1 (
  call :log "WARNING: pip upgrade inside env failed. Continuing to install requirements."
)

call :log "Installing Python packages from requirements.txt (via conda run and pip)..."
if "%SILENT%"=="1" (
  powershell -NoProfile -Command "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; (& conda run -n \"%ENV_NAME%\" python -m pip install --progress-bar off -r \"%~dp0requirements.txt\") 2>&1 | Out-File -FilePath \"%LOGFILE%\" -Append -Encoding utf8; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }"
) else (
  powershell -NoProfile -Command "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; (& conda run -n \"%ENV_NAME%\" python -m pip install --progress-bar off -r \"%~dp0requirements.txt\") 2>&1 | Tee-Object -FilePath \"%LOGFILE%\" -Append; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }"
)
if errorlevel 1 (
  call :log "ERROR: pip install -r requirements failed. See %LOGFILE% for details."
  if "%PAUSE_ON_EXIT%"=="1" pause
  exit /b 1
)
call :log "Successfully installed Python packages from requirements.txt"

:: Final messages
call :log "SUCCESS: The Conda environment '%ENV_NAME%' is ready."
if "%SILENT%"=="1" (
  REM silent: do not print final messages to console
) else (
  echo.
  echo To activate it in a new shell run:  conda activate %ENV_NAME%
  echo If you need GPU support make sure your system has a supported NVIDIA driver installed and that nvidia-smi is on PATH.
  echo Log file: %LOGFILE%
  echo.
)

:: Optional: Clean the log to remove garbled output
if "%CLEAN_LOG%"=="1" (
    call :log "Cleaning log file to remove encoding artifacts..."
    REM powershell -NoProfile -Command "try { $clean = Get-Content -Path '%LOGFILE%' -Encoding UTF8 | Where-Object { $_ -notmatch '\uFFFD' -and ( $_ -match '^\[.*\]' -or $_ -match '^{.*}' -or $_ -match '^(Successfully|ERROR|WARNING|Usage|To activate|Log file|All requested packages already installed|By downloading and using|NVIDIA GPU detected|Found requirements\\.txt|Conda found on PATH|Requested env:|Updated conda successfully|Created environment|Removed existing environment|Installed cudatoolkit/cuDNN successfully|Preparing to install|Upgrading pip|Installing Python packages|SUCCESS)' -or $_ -match '^={5,}' -or $_ -notmatch '[^\x09\x0A\x0D\x20-\uFFFF]' ) } ; Set-Content -Path '%~dp0anaconda-tf-environment-clean.log' -Value $clean -Encoding UTF8 ; } catch { Write-Error $_ ; exit 1 }"
    powershell -NoProfile -Command "try { $clean = Get-Content -LiteralPath '%LOGFILE%' -Encoding UTF8 | Where-Object { $_ -notmatch '\uFFFD' -and ( $_ -match '\[.*\]' -or $_ -match '{.*}' -or $_ -match '(Successfully|ERROR|WARNING|Usage|To activate|Log file|All requested packages already installed|By downloading and using|NVIDIA GPU detected|Found requirements\\.txt|Conda found on PATH|Requested env:|Updated conda successfully|Created environment|Removed existing environment|Installed cudatoolkit/cuDNN successfully|Preparing to install|Upgrading pip|Installing Python packages|SUCCESS)' -or $_ -match '={5,}' -or $_ -notmatch '[^\x09\x0A\x0D\x20-\uFFFF]' ) } ; Set-Content -LiteralPath '%~dp0anaconda-tf-environment-clean.log' -Value $clean -Encoding UTF8 ; } catch { Write-Error $_ ; exit 1 }"

    if errorlevel 1 (
        call :log "WARNING: Failed to generate clean log."
    ) else (
        call :log "Clean log written to: %~dp0anaconda-tf-environment-clean.log"
        if "%SILENT%"=="0" (
            echo Clean log: %~dp0anaconda-tf-environment-clean.log
        )
    )
)

if "%PAUSE_ON_EXIT%"=="1" pause
exit /b 0

:: Return from main when called
goto :eof
