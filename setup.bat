@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem One click setup for the HY-World 2.0 lab PC. Safe to run again: it only
rem installs what is missing. Optional argument:
rem   pano      HY-Pano environment (setup_pano.bat)
rem   worldgen  everything for World Generation: recon env, MSVC, CUDA toolchain, builds,
rem             then the vLLM env (setup_worldgen.bat)
rem   vlm       only the vLLM env for start_vlm.bat (setup_vlm.bat)

rem Branch or commit hash of Tencent-Hunyuan/HY-World-2.0. Pin a commit hash here for repeatable installs.
set "UPSTREAM_REF=main"
set "REPO=%~dp0HY-World-2.0"
set "TOOLS=%~dp0tools"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
rem Keep model weights next to this folder, so a wiped user profile does not mean downloading them again.
if not defined HF_HOME set "HF_HOME=%~dp0hf_cache"

set "MODE=recon"
set "ENV=hyworld2"
set "PYVER=3.11.15"
if /i "%~1"=="pano" set "MODE=pano"
if /i "%~1"=="pano" set "ENV=hyworld2-pano"
if /i "%~1"=="pano" set "PYVER=3.10"
rem worldgen shares the recon env (hyworld2) and adds the build tools on top of it.
if /i "%~1"=="worldgen" set "MODE=worldgen"
rem vlm is its own env: vLLM needs Python 3.12 and its own torch.
if /i "%~1"=="vlm" set "MODE=vlm"
if /i "%~1"=="vlm" set "ENV=vllm"
if /i "%~1"=="vlm" set "PYVER=3.12"

echo ==========================================
echo  HY-World 2.0 setup, mode: %MODE%
echo ==========================================

rem ---- 0. Checks that assume nothing about this PC ----
where curl.exe >nul 2>&1
if errorlevel 1 set "FAILED=curl.exe not found (it ships with Windows 10 version 1803 and newer)" & goto fail
where tar.exe >nul 2>&1
if errorlevel 1 set "FAILED=tar.exe not found (it ships with Windows 10 version 1803 and newer)" & goto fail
set "HERE=%~dp0"
if not "%HERE: =%"=="%HERE%" goto err_path_space
if not "%HERE:~40,1%"=="" echo WARNING: this folder path is long (%HERE%). The pytorch3d build can fail above the 260 character path limit, and long paths cannot be enabled without admin. A path like D:\hyw is safer.
set "MINGB=30"
if /i "%MODE%"=="worldgen" set "MINGB=150"
set "FREEGB="
set "DRV=%~d0"
for /f %%A in ('powershell -NoProfile -Command "[math]::Floor((Get-PSDrive %DRV:~0,1%).Free/1GB)" 2^>nul') do set "FREEGB=%%A"
if defined FREEGB if %FREEGB% LSS %MINGB% echo WARNING: only %FREEGB% GB free on %DRV%, mode %MODE% wants at least %MINGB% GB.
rem Short, user writable temp folder for pip and compiler builds.
if not exist "%~dp0tmp" mkdir "%~dp0tmp"
set "TMP=%~dp0tmp"
set "TEMP=%~dp0tmp"

rem ---- 1. Find conda ----
set "CONDA_ROOT="
for %%D in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\miniforge3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%~dp0tools\miniforge3") do if not defined CONDA_ROOT if exist "%%~D\Scripts\activate.bat" set "CONDA_ROOT=%%~D"
if defined CONDA_ROOT goto have_conda

echo No conda found. Installing Miniforge for this user, no admin needed.
rem The installer /D option must be last and unquoted, so the folder path cannot contain spaces.
set "MFDIR=%TOOLS%\miniforge3"
set "NOSPACE=%MFDIR: =%"
if not "%NOSPACE%"=="%MFDIR%" goto err_space
if not exist "%TOOLS%" mkdir "%TOOLS%"
echo Downloading Miniforge...
curl.exe -L --fail -o "%TOOLS%\Miniforge3-Windows-x86_64.exe" "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"
if errorlevel 1 set "FAILED=Downloading Miniforge" & goto fail
echo Running the Miniforge installer, this takes a few minutes...
start /wait "" "%TOOLS%\Miniforge3-Windows-x86_64.exe" /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /S /D=%MFDIR%
if errorlevel 1 set "FAILED=Installing Miniforge" & goto fail
if not exist "%MFDIR%\Scripts\activate.bat" set "FAILED=Installing Miniforge (activate.bat missing afterwards)" & goto fail
del "%TOOLS%\Miniforge3-Windows-x86_64.exe" >nul 2>&1
set "CONDA_ROOT=%MFDIR%"

:have_conda
echo Using conda at: %CONDA_ROOT%

rem ---- 2. Activate conda base ----
call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ROOT%"
if errorlevel 1 set "FAILED=Activating conda base" & goto fail

rem ---- 3. Get the HY-World-2.0 repo if missing ----
if /i "%MODE%"=="vlm" goto have_repo
if exist "%REPO%\requirements.txt" goto have_repo
echo HY-World-2.0 not found. Downloading from GitHub (ref: %UPSTREAM_REF%)...
if not exist "%TOOLS%" mkdir "%TOOLS%"
if exist "%TOOLS%\upstream_extract" rmdir /s /q "%TOOLS%\upstream_extract"
mkdir "%TOOLS%\upstream_extract"
curl.exe -L --fail -o "%TOOLS%\upstream.zip" "https://github.com/Tencent-Hunyuan/HY-World-2.0/archive/%UPSTREAM_REF%.zip"
if errorlevel 1 set "FAILED=Downloading the HY-World-2.0 source" & goto fail
tar -xf "%TOOLS%\upstream.zip" -C "%TOOLS%\upstream_extract"
if errorlevel 1 set "FAILED=Extracting the HY-World-2.0 source" & goto fail
rem The zip holds one top folder named HY-World-2.0-<ref>. Move it to the repo path.
for /d %%F in ("%TOOLS%\upstream_extract\*") do move "%%~F" "%REPO%" >nul
if not exist "%REPO%\requirements.txt" set "FAILED=Moving the HY-World-2.0 source into place" & goto fail
del "%TOOLS%\upstream.zip" >nul 2>&1
rmdir /s /q "%TOOLS%\upstream_extract" >nul 2>&1
:have_repo
echo Repo: %REPO%

rem ---- 4. Create the conda env if missing ----
rem The vlm mode checks the NVIDIA driver first, with the base python, before creating anything.
if /i not "%MODE%"=="vlm" goto after_driver
python "%~dp0setup_env.py" driver "%REPO%"
if errorlevel 1 set "FAILED=NVIDIA driver check (see messages above)" & goto fail
:after_driver
if exist "%CONDA_ROOT%\envs\%ENV%\python.exe" goto have_env
echo Creating conda env %ENV% (python %PYVER%)...
call conda create -y -n %ENV% -c conda-forge --override-channels python=%PYVER%
if errorlevel 1 set "FAILED=Creating conda env %ENV%" & goto fail
:have_env

rem ---- 5. Activate it and run the Python setup ----
call conda activate %ENV%
if errorlevel 1 set "FAILED=Activating conda env %ENV%" & goto fail
set "PYMODE=%MODE%"
if /i "%MODE%"=="worldgen" set "PYMODE=recon"
python "%~dp0setup_env.py" %PYMODE% "%REPO%"
if errorlevel 1 set "FAILED=setup_env.py %PYMODE% (see messages above)" & goto fail
if /i not "%MODE%"=="worldgen" goto after_python
python "%~dp0setup_env.py" worldgen "%REPO%"
if errorlevel 1 set "FAILED=setup_env.py worldgen (see messages above)" & goto fail
rem vLLM last, so a driver that is too old does not block the rest. The nested run
rem reports its own failure and returns instead of stopping this window.
set "VLM_STATUS=OK"
set "NESTED=1"
call "%~dp0setup.bat" vlm
if errorlevel 1 set "VLM_STATUS=FAILED"
set "NESTED="
:after_python

rem ---- 6. Done: leave a ready Command Prompt ----
if /i "%MODE%"=="pano" goto done_pano
if /i "%MODE%"=="worldgen" goto done_worldgen
if /i "%MODE%"=="vlm" goto done_vlm
cd /d "%REPO%"
echo.
echo ==========================================
echo  Setup complete. Environment: %ENV%
echo  You are in the repo folder. Next steps:
echo    Read README.md, section World Reconstruction.
echo    Example: python -m hyworld2.worldrecon.pipeline --help
echo ==========================================
cmd /k
exit /b 0

:done_pano
cd /d "%REPO%\hyworld2\panogen"
echo.
echo ==========================================
echo  Setup complete. Environment: %ENV%
echo  You are in hyworld2\panogen. Next steps:
echo    Read README.md here. No models were downloaded yet.
echo ==========================================
cmd /k
exit /b 0

:done_worldgen
rem setup_env.py wrote the path of the MSVC environment .bat it used into tools\msvc_env.txt.
if not exist "%TOOLS%\msvc_env.txt" set "FAILED=Reading tools\msvc_env.txt (written by setup_env.py)" & goto fail
set /p MSVC_BAT=<"%TOOLS%\msvc_env.txt"
if not exist "%MSVC_BAT%" set "FAILED=Finding the MSVC environment file %MSVC_BAT%" & goto fail
call "%MSVC_BAT%" >nul
where cl.exe >nul 2>&1
if errorlevel 1 set "FAILED=Loading the MSVC environment from %MSVC_BAT% (cl.exe not found afterwards)" & goto fail
set "CUDA_HOME=%CONDA_PREFIX%\Library"
set "CUDA_PATH=%CONDA_PREFIX%\Library"
set "INCLUDE=%CONDA_PREFIX%\Library\include;%CONDA_PREFIX%\Library\include\targets\x64;%INCLUDE%"
set "LIB=%CONDA_PREFIX%\Library\lib;%CONDA_PREFIX%\Library\lib\x64;%LIB%"
set "DISTUTILS_USE_SDK=1"
set "MSSdk=1"
set "TORCH_CUDA_ARCH_LIST=12.0"
set "USE_LIBUV=0"
rem CMake builds must use Ninja: the Visual Studio generator needs an installed Visual Studio.
set "CMAKE_GENERATOR=Ninja"
rem Windows defaults to cp1252. UTF-8 keeps reading meta_info.json and printing paths safe.
set "PYTHONUTF8=1"
cd /d "%REPO%\hyworld2\worldgen"
echo.
echo ==========================================
echo  Setup complete. Environment: %ENV% with the World Generation build tools
echo  You are in hyworld2\worldgen. MSVC, CUDA_HOME, CUDA_PATH,
echo  DISTUTILS_USE_SDK, TORCH_CUDA_ARCH_LIST=12.0, USE_LIBUV=0 and HF_HOME are set.
echo  Next steps:
echo    Read README.md, section World generation, and start_vlm.bat for stages 1 and 2.
if /i "%VLM_STATUS%"=="FAILED" echo.
if /i "%VLM_STATUS%"=="FAILED" echo  WARNING: the vLLM setup FAILED, see the messages above. Everything else is ready.
if /i "%VLM_STATUS%"=="FAILED" echo  Fix the cause (an NVIDIA driver older than 580 needs IT) and run setup_vlm.bat.
echo ==========================================
cmd /k
exit /b 0

:done_vlm
echo.
echo ==========================================
echo  vLLM env ready: %ENV%. Start the server with start_vlm.bat.
echo ==========================================
if defined NESTED exit /b 0
cmd /k
exit /b 0

:err_path_space
echo.
echo ERROR: this folder path contains spaces:
echo   %HERE%
echo Move this whole folder to a short path without spaces (for example D:\hyw) and run again.
if not defined NESTED pause
exit /b 1

:err_space
echo.
echo ERROR: Miniforge cannot be installed into a folder path that contains spaces:
echo   %MFDIR%
echo Move this whole folder to a path without spaces (for example D:\HY-World-Experimentation) and run setup.bat again.
if not defined NESTED pause
exit /b 1

:fail
echo.
echo ERROR: setup failed at step: %FAILED%
if not defined NESTED pause
exit /b 1
