@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Starts the vision language model server that World Generation stages 1 and 2 talk to.
rem Leave this window open while you run the stages. Stop it with Ctrl+C.
rem
rem vLLM is installed by setup_vlm.bat (also run by setup_worldgen.bat) into the conda env
rem named vllm. Run setup_vlm.bat first if that env does not exist.

rem Starting values to tune on the 12 GB card. Stage 1 also loads SAM3, MoGe, Grounding DINO
rem and ZIM on the same GPU, so the server must not take all of the VRAM.
rem   MAX_LEN       longest prompt plus answer in tokens. Lower it to save VRAM.
rem   GPU_UTIL      share of GPU memory vLLM may use (0.45 is about 5.4 GB of 12 GB).
rem   CPU_OFFLOAD   GB of model weights kept in system RAM instead of VRAM.
set "VLM_ENV=vllm"
set "MODEL=Qwen/Qwen3-VL-8B-Instruct-FP8"
set "HOST=127.0.0.1"
set "PORT=8000"
set "MAX_LEN=8192"
set "GPU_UTIL=0.45"
set "CPU_OFFLOAD=6"

set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
rem Same model cache as setup.bat.
if not defined HF_HOME set "HF_HOME=%~dp0hf_cache"

rem ---- 1. Find conda (same places as setup.bat) ----
set "CONDA_ROOT="
for %%D in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\miniforge3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%~dp0tools\miniforge3") do if not defined CONDA_ROOT if exist "%%~D\Scripts\activate.bat" set "CONDA_ROOT=%%~D"
if not defined CONDA_ROOT goto err_conda
echo Using conda at: %CONDA_ROOT%

rem ---- 2. Activate the vllm env ----
if not exist "%CONDA_ROOT%\envs\%VLM_ENV%\python.exe" goto err_env
call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ROOT%"
if errorlevel 1 goto err_activate
call conda activate %VLM_ENV%
if errorlevel 1 goto err_activate

rem ---- 3. Serve ----
echo.
echo Serving %MODEL% on http://%HOST%:%PORT% (max len %MAX_LEN%, GPU share %GPU_UTIL%, CPU offload %CPU_OFFLOAD% GB)
echo Use these for the worldgen scripts: --llm_addr %HOST% --llm_port %PORT% --llm_name %MODEL%
echo.
vllm serve %MODEL% --served-model-name %MODEL% --host %HOST% --port %PORT% --max-model-len %MAX_LEN% --gpu-memory-utilization %GPU_UTIL% --cpu-offload-gb %CPU_OFFLOAD% --trust-remote-code
echo.
echo The vLLM server stopped (exit code %ERRORLEVEL%).
pause
exit /b 0

:err_conda
echo.
echo ERROR: No conda found. Run setup.bat once first, it installs Miniforge when needed.
pause
exit /b 1

:err_env
echo.
echo ERROR: conda env "%VLM_ENV%" not found at %CONDA_ROOT%\envs\%VLM_ENV%
echo Run setup_vlm.bat first, it creates the env and installs vLLM.
pause
exit /b 1

:err_activate
echo.
echo ERROR: Could not activate conda env %VLM_ENV%.
pause
exit /b 1
