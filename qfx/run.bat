@echo off

REM ===================================================================================================================
REM  Created by: Quentin Lengele (16/10/2025)
REM ===================================================================================================================

set PY_LIB_PATH=%1
set PY_FILE=%2
set INPUT_FILELIST=%3

set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0
set TORCH_CUDA_ARCH_LIST=8.0;8.6+PTX
set SPCONV_ALGO=native

REM SPCONV_ALGO can be 'native' or 'auto', default is 'auto'.
REM 'auto' is faster but will do benchmarking at the beginning.
REM Recommended to set to 'native' if run only once.

set ATTN_BACKEND=flash-attn
REM ATTN_BACKEND can be 'flash-attn' or 'xformers', default is 'flash-attn'

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.3,max_split_size_mb:128

set PYTHONPATH=%PY_LIB_PATH%;%PY_LIB_PATH%\matgen;%PYTHONPATH%

cd /D "%PY_LIB_PATH%"
call venv\Scripts\activate

call python qfx/%PY_FILE% %INPUT_FILELIST%