@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem One click setup for the HY-World 2.0 lab PC. Safe to run again: it only
rem installs what is missing. Optional argument: pano (HY-Pano environment).

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

echo ==========================================
echo  HY-World 2.0 setup, mode: %MODE%
echo ==========================================

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
if exist "%CONDA_ROOT%\envs\%ENV%\python.exe" goto have_env
echo Creating conda env %ENV% (python %PYVER%)...
call conda create -y -n %ENV% -c conda-forge --override-channels python=%PYVER%
if errorlevel 1 set "FAILED=Creating conda env %ENV%" & goto fail
:have_env

rem ---- 5. Activate it and run the Python setup ----
call conda activate %ENV%
if errorlevel 1 set "FAILED=Activating conda env %ENV%" & goto fail
python "%~dp0setup_env.py" %MODE% "%REPO%"
if errorlevel 1 set "FAILED=setup_env.py %MODE% (see messages above)" & goto fail

rem ---- 6. Done: leave a ready Command Prompt ----
if /i "%MODE%"=="pano" goto done_pano
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

:err_space
echo.
echo ERROR: Miniforge cannot be installed into a folder path that contains spaces:
echo   %MFDIR%
echo Move this whole folder to a path without spaces (for example D:\HY-World-Experimentation) and run setup.bat again.
pause
exit /b 1

:fail
echo.
echo ERROR: setup failed at step: %FAILED%
pause
exit /b 1
