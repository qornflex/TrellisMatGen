@echo off

REM ===================================================================================================================
REM  Created by: Quentin Lengele (16/10/2025)
REM ===================================================================================================================

echo.
echo =============================================
echo  Trellis and MatGen installation for Windows
echo =============================================
echo.

REM ===================================================================================================================
REM Ask for Python 3.10 folder
REM ===================================================================================================================

echo Enter the path to your Python 3.10 directory (e.g. C:\Python310\)
echo.
:get_python_path
set /p PYTHON_FOLDER=Python 3.10 Directory: 

if not exist "%PYTHON_FOLDER%\python.exe" (
    echo.
	echo   Can't find any python executable here '%PYTHON_FOLDER%\python.exe'.
    echo.
	goto get_python_path    
) else (
	set PYTHON_MAIN=%PYTHON_FOLDER%\python.exe
)

REM ===================================================================================================================
REM Check Python Version
REM ===================================================================================================================

for /f "tokens=2 delims= " %%v in ('%PYTHON_MAIN% --version') do set PYVER=%%v
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do set PYSHORT=%%a.%%b

if "%PYSHORT%"=="3.10" (
    echo.
    echo   Python %PYVER% is installed.
	echo.
) else (
    echo.
    echo   =====================================================================================
    echo   The provided Python executable is %PYVER% (%PYTHON_MAIN%^)
    echo.
    echo   Please install Python 3.10: 
    echo   https://www.python.org/ftp/python/3.10.9/python-3.10.9-amd64.exe
    echo   =====================================================================================
    echo.
    pause
    exit /b 1    
)

REM ===================================================================================================================
REM Ask for CUDA ToolKit 12.8 folder
REM ===================================================================================================================

echo Enter the path to your CUDA Toolkit 12.8 directory
echo (e.g. C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\)
echo.
:get_cuda_path
set /p CUDA_FOLDER=Cuda TookKit Directory: 

if not exist "%CUDA_FOLDER%\bin\nvcc.exe" (
    echo.
	echo   Can't find any CUDA executable here '%CUDA_FOLDER%\bin\nvcc.exe'.
    echo.
	goto get_cuda_path    
) else (
	set CUDA_NVCC="%CUDA_FOLDER%\bin\nvcc.exe"
)

REM ===================================================================================================================
REM Check CUDA ToolKit 12.8 Installation
REM ===================================================================================================================

setlocal enabledelayedexpansion

set "CUDA_PATH=%CUDA_FOLDER%"

set PATH=%CUDA_FOLDER%\bin;%CUDA_FOLDER%\libnvvp;%PATH%

nvcc --version >nul 2>&1
if %errorlevel%==0 (
    set "CUDA_VERSION="
    for /f "tokens=6" %%v in ('nvcc --version ^| findstr "release"') do set "CUDA_VERSION=%%v"

    rem Remove trailing comma
    set "CUDA_VERSION=!CUDA_VERSION:,=!"
    rem Remove leading V if present
    set "CUDA_VERSION=!CUDA_VERSION:V=!"
    rem Trim spaces
    for /f "tokens=* delims= " %%a in ("!CUDA_VERSION!") do set "CUDA_VERSION=%%a"

    rem Keep only major.minor
    set "CUDA_MAJOR=!CUDA_VERSION:~0,4!"
            
    if "!CUDA_MAJOR!"=="12.8" (
        echo.
        echo   CUDA Toolkit 12.8 is installed
        echo.
    ) else (
        echo.
        echo   =========================================================================================================
        echo   Your CUDA ToolKit version is !CUDA_VERSION!
        echo   You need to install CUDA Toolkit 12.8
        echo   Please install it from here: 
        echo   https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_571.96_windows.exe
        echo   =========================================================================================================
        echo.
        pause
        exit /b 1
    )
) else (
    echo.
    echo   ===========================================================================================================
    echo   CUDA Toolkit 12.8 not installed or not found in your PATH environment variable
    echo   Please install it from here: 
    echo   https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_571.96_windows.exe 
    echo   ===========================================================================================================
    echo.
    pause
    exit /b 1
)

REM ===================================================================================================================
REM Check for Git Installation
REM ===================================================================================================================

REM Check if Git is available in PATH
echo Checking Git Installation...
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo   =====================================================================
    echo   Git is not installed or not found in your PATH environment variable.
    echo   Please install Git for Windows:
    echo   https://gitforwindows.org
    echo   =====================================================================
    echo.
    pause
    exit /b 1
)

git --version >nul 2>&1
if %errorlevel%==0 (
    REM Install and enable Git LFS
    echo   Git is installed
    git lfs install
	echo.
) else (
    echo.
    echo   =====================================================================
    echo   Git is not installed or not found in your PATH environment variable
    echo   Please install Git for Windows: 
    echo   https://gitforwindows.org
    echo   =====================================================================
    echo.
    pause
    exit /b 1
)

REM -------------------------------------------------------------------------------------------------------------------
REM CLONE MODELS
REM -------------------------------------------------------------------------------------------------------------------

set TRELLIS_MODELS_DIR="%CD%\models"
if not exist "%TRELLIS_MODELS_DIR%\" (
    mkdir "%TRELLIS_MODELS_DIR%"
)

git clone https://huggingface.co/microsoft/TRELLIS-image-large ./models/TRELLIS-image-large
cd ./models/TRELLIS-image-large
git config advice.detachedHead false
git checkout 25e0d31ffbebe4b5a97464dd851910efc3002d96

cd ../..

git clone https://huggingface.co/microsoft/TRELLIS-text-base ./models/TRELLIS-text-base
cd ./models/TRELLIS-text-base
git config advice.detachedHead false
git checkout f8e8cf00c40d53dea26b718e49169ce83cf24c67

cd ../..

git clone https://huggingface.co/microsoft/TRELLIS-text-large ./models/TRELLIS-text-large
cd ./models/TRELLIS-text-large
git config advice.detachedHead false
git checkout 4aad9f4a110329a410974d7f41ce5333a9a1fc87

cd ../..

git clone https://huggingface.co/microsoft/TRELLIS-text-xlarge ./models/TRELLIS-text-xlarge
cd ./models/TRELLIS-text-xlarge
git config advice.detachedHead false
git checkout e0b00432b8e3a8ecee0df806ab1df9f7281f2be4

cd ../..

REM -------------------------------------------------------------------------------------------------------------------
REM VENV & PIP
REM -------------------------------------------------------------------------------------------------------------------

%PYTHON_MAIN% -m venv venv

call venv\Scripts\activate

call python -m pip install --upgrade pip
call python -m pip install wheel

REM -------------------------------------------------------------------------------------------------------------------
REM PYTORCH
REM -------------------------------------------------------------------------------------------------------------------

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

REM -------------------------------------------------------------------------------------------------------------------
REM REQUIREMENTS
REM -------------------------------------------------------------------------------------------------------------------

pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh xatlas pyvista pymeshfix igraph transformers

REM utils3d

pip install -e ./tmp/extensions/utils3d

REM Blender

pip install bpy==4.0 --extra-index-url https://download.blender.org/pypi/

REM MathUtils

pip install mathutils

REM Open3D

pip install open3d

REM pygltflib

pip install pygltflib

REM KAOLIN

pip install ./tmp/wheels/kaolin-0.18.0-cp310-cp310-win_amd64.whl

REM NVIDIA FRAST

pip install ./tmp/extensions/nvdiffrast --no-build-isolation

REM DIFFOCTREERAST

pip install ./tmp/extensions/diffoctreerast --no-build-isolation

REM MIP GAUSSIAN

pip install ./tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/ --no-build-isolation

REM SPCONV

pip install spconv-cu120

REM VOX2SEQ
set VOX2SEQ_EXT_DIR="%CD%\extensions\vox2seq"
if exist %VOX2SEQ_EXT_DIR% (
	xcopy /E /I extensions\vox2seq .\tmp\extensions\vox2seq /Y
	pip install ./tmp/extensions/vox2seq
)

REM GRADIO

pip install gradio==4.44.1 gradio_litmodel3d==0.0.1

REM FLASH-ATTN

pip install https://huggingface.co/marcorez8/flash-attn-windows-blackwell/resolve/e1480e12fc744c1edf2f50831a5363d0faef45e4/flash_attn-2.7.4.post1-cp310-cp310-win_amd64-torch2.7.0-cu128/flash_attn-2.7.4.post1-cp310-cp310-win_amd64.whl

echo.
echo ============================================
echo  Trellis and MatGen installation completed!
echo ============================================
echo.

pause
