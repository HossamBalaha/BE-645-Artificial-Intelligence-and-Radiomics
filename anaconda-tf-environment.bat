@echo off
:: This batch file sets up a TensorFlow environment with CUDA support on Windows.
:: It stops execution if any step fails and provides detailed feedback.

:: Step 1: Create a new Conda environment with Python 3.10
echo Creating a new Conda environment named 'tf' with Python 3.10...
conda create -n tf python=3.10.* -y
if %ERRORLEVEL% neq 0 (
    echo Failed to create the Conda environment. Exiting...
    pause
    exit /b %ERRORLEVEL%
)
echo Environment 'tf' created successfully.

:: Step 2: Activate the 'tf' environment
echo Activating the 'tf' environment...
call conda activate tf
if %ERRORLEVEL% neq 0 (
    echo Failed to activate the 'tf' environment. Exiting...
    pause
    exit /b %ERRORLEVEL%
)
echo Environment 'tf' activated successfully.

:: Step 3: Install CUDA Toolkit from conda-forge
echo Installing CUDA Toolkit from conda-forge...
conda install -c conda-forge cudatoolkit -y
if %ERRORLEVEL% neq 0 (
    echo Failed to install CUDA Toolkit. Exiting...
    pause
    exit /b %ERRORLEVEL%
)
echo CUDA Toolkit installed successfully.

:: Step 4: Install cuDNN from conda-forge
echo Installing cuDNN from conda-forge...
conda install -c conda-forge cudnn -y
if %ERRORLEVEL% neq 0 (
    echo Failed to install cuDNN. Exiting...
    pause
    exit /b %ERRORLEVEL%
)
echo cuDNN installed successfully.

:: Step 5: Install Python dependencies from requirements.txt
echo Installing Python dependencies from requirements.txt...
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo Failed to install Python dependencies. Exiting...
    pause
    exit /b %ERRORLEVEL%
)
echo Python dependencies installed successfully.

:: Completion message
echo Setup completed successfully. The 'tf' environment is ready to use.
pause