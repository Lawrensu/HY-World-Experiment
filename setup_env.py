"""Set up the Python side of the HY-World 2.0 environment on the Windows lab PC.

Runs inside the activated conda env (setup.bat does that). Stdlib only at top level.
Every step checks first and only installs what is missing, so reruns are fast.

Usage: python setup_env.py {recon,pano,worldgen,vlm,driver} REPO
(vlm and driver do not need the repo, pass any path)
"""

import argparse
import ctypes
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path




# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CU128_INDEX = "https://download.pytorch.org/whl/cu128"

# Edits needed so the official requirements.txt installs on Windows without a compiler.
#   cupy: the source build needs nvcc, the prebuilt cuda12x wheel does not.
#   pycolmap: the 3.10.0 Windows wheel fails to import (DLL initialization routine failed).
REQ_EDITS = [
    ("cupy==13.6.0", "cupy-cuda12x==13.6.0"),
    ("pycolmap==3.10.0", "pycolmap==3.11.1"),
]

# World generation (setup_worldgen.bat)
EXP_DIR = Path(__file__).resolve().parent
TOOLS_DIR = EXP_DIR / "tools"
PATCH_PATH = EXP_DIR / "patches" / "worldgen_windows.patch"
PATCH_MARKER = "HY-World Windows patch"
MSVC_ENV_FILE = TOOLS_DIR / "msvc_env.txt"
PORTABLE_MSVC_BAT = TOOLS_DIR / "msvc" / "setup_x64.bat"
WORLDGEN_REL = "hyworld2/worldgen"
RECAST_REL = "hyworld2/worldgen/third_party/recastnavigation"
NAVMESH_REL = "hyworld2/worldgen/third_party/navmesh"
GLM_REL = "hyworld2/worldgen/third_party/gsplat_maskgaussian/gsplat/cuda/csrc/third_party/glm"
# The commit the upstream repo pins as its recastnavigation submodule (the ZIP has no submodules).
RECAST_COMMIT = "9f4ce64458dfae86e1239c525ddc219c4e9e06f1"
RECAST_URL = f"https://github.com/recastnavigation/recastnavigation/archive/{RECAST_COMMIT}.zip"
# gsplat_maskgaussian (a copy of gsplat 1.5.3) includes <glm/glm.hpp>, but the HY-World copy
# leaves out gsplat's glm submodule. Use the exact commit gsplat 1.5.3 pins for it.
GLM_COMMIT = "33b4a621a697a305bc3a7610d290677b96beb181"
GLM_URL = f"https://github.com/g-truc/glm/archive/{GLM_COMMIT}.zip"
# Community script that downloads MSVC from Microsoft and extracts it without admin rights
# (gist by mmozeiko). Pinned to one revision and checked by hash before it is run.
MSVC_SCRIPT_URL = (
    "https://gist.githubusercontent.com/mmozeiko/7f3162ec2988e81e56d5c4e22cde9977/raw/"
    "553d55bb7f0405a2fbd0228b8ed87c2b8ad200f8/portable-msvc.py"
)
MSVC_SCRIPT_SHA256 = "1791cccca594ff083ce8d3e0a139e6156d2c312e24332a909e608010f0230a51"
MSVC_SCRIPT_PATH = TOOLS_DIR / "portable-msvc.py"
MSVC_VERSION = "14.44"  # the newest toolset CUDA 12.8 supports

# vLLM for Windows (community build by SystemPanic). Needs CUDA 13, so NVIDIA driver R580+.
VLLM_VERSION = "0.32.0"
VLLM_WHEEL_NAME = "vllm-0.32.0+cu132-cp312-cp312-win_amd64.whl"
VLLM_WHEEL_URL = (
    "https://github.com/SystemPanic/vllm-windows/releases/download/v0.32.0/"
    "vllm-0.32.0%2Bcu132-cp312-cp312-win_amd64.whl"
)
VLLM_WHEEL_SHA256 = "a5b91bcbce16ab4b0fa8c84852fc76c921d641b2e11091a8f7be269ef983cf38"
VLLM_EXTRA_INDEX = "https://download.pytorch.org/whl/cu130"
MIN_DRIVER_MAJOR = 580

ATTN_REL = "hyworld2/worldrecon/hyworldmirror/models/layers/attention.py"
GSPLAT_REL = "hyworld2/worldgen/third_party/gsplat_maskgaussian"

ATTN_OLD_IMPORT = [
    "try:",
    "    from flash_attn_interface import flash_attn_func as flash_attn_func_v3",
    "    _USE_FLASH_ATTN_V3 = True",
    "except ImportError:",
    "    from flash_attn.flash_attn_interface import flash_attn_func as flash_attn_func_v2",
    "    _USE_FLASH_ATTN_V3 = False",
]

ATTN_NEW_IMPORT = [
    "try:",
    "    from flash_attn_interface import flash_attn_func as flash_attn_func_v3",
    "    _USE_FLASH_ATTN_V3 = True",
    "    _HAS_FLASH_ATTN = True",
    "except ImportError:",
    "    _USE_FLASH_ATTN_V3 = False",
    "    try:",
    "        from flash_attn.flash_attn_interface import flash_attn_func as flash_attn_func_v2",
    "        _HAS_FLASH_ATTN = True",
    "    except ImportError:",
    "        _HAS_FLASH_ATTN = False",
]

ATTN_OLD_LINE = "        if q.dtype==torch.bfloat16 or q.dtype==torch.float16:"
ATTN_NEW_LINE = "        if _HAS_FLASH_ATTN and (q.dtype==torch.bfloat16 or q.dtype==torch.float16):"




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def step(i, total, title):
    print()
    print(f"[{i}/{total}] {title}")


def fail(msg):
    print(f"ERROR: {msg}")
    sys.exit(1)


def pip(*args, cwd=None, env=None):
    cmd = [sys.executable, "-m", "pip", *args]
    print("  >", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def run_py(code, cwd=None, capture=True):
    """Run code in a fresh interpreter so a broken import cannot kill this script."""
    return subprocess.run(
        [sys.executable, "-c", code], cwd=cwd, capture_output=capture, text=True
    )


def read_text(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return f.read()


def write_text(path, text):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module_available(name):
    import importlib.util
    return importlib.util.find_spec(name) is not None


def torch_is_cu128(with_audio=False):
    code = (
        "import sys, torch\n"
        "ok = torch.__version__.startswith('2.7.1') and torch.version.cuda == '12.8'\n"
        "if " + str(with_audio) + ":\n"
        "    import torchaudio\n"
        "    ok = ok and torchaudio.__version__.startswith('2.7.1')\n"
        "print(torch.__version__, torch.version.cuda)\n"
        "sys.exit(0 if ok else 1)\n"
    )
    r = run_py(code)
    if r.stdout.strip():
        print("  found torch:", r.stdout.strip())
    return r.returncode == 0


def install_torch(with_audio=False, force=False):
    pkgs = ["torch==2.7.1", "torchvision==0.22.1"]
    if with_audio:
        pkgs.append("torchaudio==2.7.1")
    flags = ["--force-reinstall", "--no-deps"] if force else []
    pip("install", *flags, *pkgs, "--index-url", CU128_INDEX)
    if not torch_is_cu128(with_audio):
        fail("torch is still not the 2.7.1 CUDA 12.8 build after installing")


def ensure_torch(with_audio=False):
    if torch_is_cu128(with_audio):
        print("  OK, already done")
        return
    print("  installing torch cu128 build")
    install_torch(with_audio)


def install_requirements(req_path, marker_name):
    """Run pip install -r when the file changed since the last successful run."""
    marker = Path(sys.prefix) / marker_name
    digest = sha256_of(req_path)
    if marker.exists() and marker.read_text().strip() == digest:
        print("  OK, already done")
        return False
    print(f"  installing {req_path}")
    pip("install", "-r", str(req_path))
    marker.write_text(digest + "\n")
    return True


def require_cuda(cwd=None):
    code = (
        "import sys, torch\n"
        "print('torch', torch.__version__, 'cuda', torch.version.cuda)\n"
        "avail = torch.cuda.is_available()\n"
        "print('torch.cuda.is_available():', avail)\n"
        "if not avail:\n"
        "    sys.exit(1)\n"
        "p = torch.cuda.get_device_properties(0)\n"
        "print('GPU:', p.name, '|', round(p.total_memory / 1024**3, 1), 'GB VRAM')\n"
    )
    r = run_py(code, cwd=cwd, capture=False)
    if r.returncode != 0:
        fail("CUDA is not available to torch (see output above)")


def memory_gb():
    """Return (free, total) system RAM in GB using the Windows API, or None."""
    if os.name != "nt":
        return None

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    gb = 1024 ** 3
    return stat.ullAvailPhys / gb, stat.ullTotalPhys / gb




# ---------------------------------------------------------------------------
# Edits to upstream files (both idempotent, both keep the file's line endings)
# ---------------------------------------------------------------------------

def edit_requirements(req_path):
    text = read_text(req_path)
    changed = False
    for old, new in REQ_EDITS:
        old_re = re.compile(r"^" + re.escape(old) + r"[ \t]*(?=\r?$)", re.M)
        new_re = re.compile(r"^" + re.escape(new) + r"[ \t]*(?=\r?$)", re.M)
        if old_re.search(text):
            text = old_re.sub(new, text)
            changed = True
            print(f"  {old} -> {new}")
        elif not new_re.search(text):
            fail(f"neither '{old}' nor '{new}' found in {req_path} (upstream changed?)")
    if changed:
        write_text(req_path, text)
    else:
        print("  OK, already done")
    return changed


def patch_attention(attn_path):
    text = read_text(attn_path)
    if "_HAS_FLASH_ATTN" in text:
        print("  OK, already done")
        return False
    nl = "\r\n" if "\r\n" in text else "\n"
    old_import = nl.join(ATTN_OLD_IMPORT)
    new_import = nl.join(ATTN_NEW_IMPORT)
    if old_import not in text:
        fail(f"flash_attn import block not found in {attn_path} (upstream changed?)")
    if text.count(ATTN_OLD_LINE) != 1:
        fail(f"dtype check line not found exactly once in {attn_path} (upstream changed?)")
    text = text.replace(old_import, new_import).replace(ATTN_OLD_LINE, ATTN_NEW_LINE)
    write_text(attn_path, text)
    print("  patched attention.py to fall back to torch SDPA")
    return True




# ---------------------------------------------------------------------------
# World Reconstruction (official README steps plus the forced deviations)
# ---------------------------------------------------------------------------

def setup_recon(repo):
    total = 8
    req_path = repo / "requirements.txt"

    step(1, total, "Edit requirements.txt for Windows (cupy-cuda12x, pycolmap 3.11.1)")
    edit_requirements(req_path)

    step(2, total, "torch 2.7.1 + torchvision 0.22.1, CUDA 12.8 build")
    ensure_torch()

    step(3, total, "pip install -r requirements.txt")
    pip_ran = install_requirements(req_path, "hyworld_requirements.sha256")

    step(4, total, "Check pip did not swap torch for another build")
    if not pip_ran:
        print("  OK, nothing was installed in step 3")
    elif torch_is_cu128():
        print("  OK, torch is still the cu128 build")
    else:
        print("  torch was swapped, reinstalling the cu128 build")
        install_torch(force=True)

    step(5, total, "gsplat_maskgaussian (editable, BUILD_NO_CUDA=1)")
    gsplat_dir = repo / GSPLAT_REL
    r = run_py("import gsplat; print(gsplat.__file__)", cwd=sys.prefix)
    inside = False
    if r.returncode == 0:
        inside = Path(r.stdout.strip()).resolve().is_relative_to(gsplat_dir.resolve())
    if inside:
        print("  OK, already done")
    else:
        env = dict(os.environ, BUILD_NO_CUDA="1")
        pip("install", "-e", ".", "--no-build-isolation", cwd=gsplat_dir, env=env)

    step(6, total, "onnxruntime and requests (used by the sky mask code)")
    missing = [m for m in ("onnxruntime", "requests") if not module_available(m)]
    if missing:
        pip("install", *missing)
    else:
        print("  OK, already done")

    step(7, total, "Patch attention.py to fall back to torch SDPA (no FlashAttention)")
    patch_attention(repo / ATTN_REL)

    step(8, total, "Verify")
    require_cuda(cwd=repo)
    r = run_py(
        "import importlib\n"
        "m = importlib.import_module('hyworld2.worldrecon.hyworldmirror.models.layers.attention')\n"
        "print('flash attention:', m._HAS_FLASH_ATTN)\n",
        cwd=repo, capture=False,
    )
    if r.returncode != 0:
        fail("could not import the patched attention module")
    r = run_py(
        "from hyworld2.worldrecon.pipeline import WorldMirrorPipeline\n"
        "print('PIPELINE OK')\n",
        cwd=repo, capture=False,
    )
    if r.returncode != 0:
        fail("WorldMirrorPipeline import failed (see output above)")




# ---------------------------------------------------------------------------
# HY-Pano 2.0 (official README steps, cu128 instead of cu118)
# ---------------------------------------------------------------------------

def setup_pano(repo):
    total = 5
    # The README says "cd HY-World-2.0" then "pip install -r requirements.txt", but it
    # lives in hyworld2/panogen/ next to that README. The root requirements.txt is the
    # World Reconstruction one (numpy 1.26.4, transformers 5.2.0) and conflicts with the
    # pano pins (numpy 2.2.0, transformers 4.57.1), so the panogen file is the right one.
    req_path = repo / "hyworld2" / "panogen" / "requirements.txt"
    if not req_path.exists():
        fail(f"{req_path} not found (upstream changed?)")

    step(1, total, "torch 2.7.1 + torchvision 0.22.1 + torchaudio 2.7.1, CUDA 12.8 build")
    # Deviation: the README uses the cu118 index, which has no RTX 5070 (sm_120) support.
    ensure_torch(with_audio=True)

    step(2, total, "pip install -r hyworld2/panogen/requirements.txt")
    pip_ran = install_requirements(req_path, "hyworld_pano_requirements.sha256")

    step(3, total, "Check pip did not swap torch for another build")
    if not pip_ran:
        print("  OK, nothing was installed in step 2")
    elif torch_is_cu128(with_audio=True):
        print("  OK, torch is still the cu128 build")
    else:
        print("  torch was swapped, reinstalling the cu128 build")
        install_torch(with_audio=True, force=True)

    step(4, total, "peft (diffusers needs it to load the HY-Pano LoRA)")
    # The panogen requirements file does not list peft. Use the pin from the root
    # requirements.txt, which the official README points at.
    if module_available("peft"):
        print("  OK, already done")
    else:
        pip("install", "peft==0.18.1")

    step(5, total, "Verify")
    require_cuda(cwd=repo)
    mem = memory_gb()
    if mem is None:
        print("system RAM: unknown (not Windows)")
    else:
        print(f"system RAM: {mem[0]:.1f} GB free of {mem[1]:.1f} GB total")
    print("PANO ENV OK (no models were downloaded)")




# ---------------------------------------------------------------------------
# World Generation (official README step 3 plus the forced deviations)
# ---------------------------------------------------------------------------

def lib_dir():
    """The conda Library folder, which is where the conda CUDA toolkit lives on Windows."""
    return Path(sys.prefix) / "Library"


def prepend_path(var, dirs):
    """Put existing dirs at the front of a ';' separated variable, without duplicates."""
    parts = [p for p in os.environ.get(var, "").split(";") if p]
    have = {os.path.normcase(os.path.normpath(p)) for p in parts}
    new = []
    for d in dirs:
        d = Path(d)
        if d.is_dir() and os.path.normcase(os.path.normpath(str(d))) not in have:
            new.append(str(d))
    if new:
        os.environ[var] = ";".join(new + parts)


def conda_exe():
    exe = os.environ.get("CONDA_EXE")
    if exe and Path(exe).exists():
        return exe
    found = shutil.which("conda")
    if not found:
        fail("conda not found (CONDA_EXE is not set). Run this through setup_worldgen.bat.")
    return found


def conda_install(*pkgs):
    cmd = [conda_exe(), "install", "-y", "-p", sys.prefix,
           "-c", "conda-forge", "--override-channels", *pkgs]
    print("  >", " ".join(cmd))
    return subprocess.run(cmd).returncode == 0


def nvcc_is_128():
    nvcc = lib_dir() / "bin" / "nvcc.exe"
    if not nvcc.exists():
        return False
    try:
        r = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True)
    except OSError:
        return False
    return "release 12.8" in r.stdout


def download(url, dest):
    cmd = ["curl.exe", "-L", "--fail", "-o", str(dest), url]
    print("  >", " ".join(cmd))
    if subprocess.run(cmd).returncode != 0:
        fail(f"download failed: {url}")


def download_verified(url, dest, sha256):
    """Download url to dest and check its sha256. A file with the right hash is reused."""
    dest = Path(dest)
    if dest.is_file() and sha256_of(dest) == sha256:
        print(f"  {dest.name}: already downloaded, hash OK")
        return
    download(url, dest)
    got = sha256_of(dest)
    if got != sha256:
        dest.unlink(missing_ok=True)
        fail(f"sha256 mismatch for {dest.name}\n  expected {sha256}\n  got      {got}\n  The file was deleted.")
    print(f"  {dest.name}: hash OK")


def fetch_zip_folder(url, dest, name):
    """Download a GitHub ZIP and move its single top folder to dest."""
    TOOLS_DIR.mkdir(exist_ok=True)
    zip_path = TOOLS_DIR / f"{name}.zip"
    tmp = TOOLS_DIR / f"{name}_extract"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    download(url, zip_path)
    if subprocess.run(["tar", "-xf", str(zip_path), "-C", str(tmp)]).returncode != 0:
        fail(f"could not extract {zip_path}")
    tops = [p for p in tmp.iterdir() if p.is_dir()]
    if len(tops) != 1:
        fail(f"expected one top folder in {zip_path}, found {len(tops)}")
    if dest.exists():
        if any(dest.iterdir()):
            fail(f"{dest} exists and is not empty, remove it and run again")
        dest.rmdir()  # empty placeholder
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tops[0]), str(dest))
    shutil.rmtree(tmp, ignore_errors=True)
    zip_path.unlink(missing_ok=True)


def find_vcvars():
    """vcvars64.bat of an installed Visual Studio, else the portable MSVC, else None."""
    pf86 = os.environ.get("ProgramFiles(x86)")
    if pf86:
        vswhere = Path(pf86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if vswhere.exists():
            r = subprocess.run(
                [str(vswhere), "-latest", "-products", "*",
                 "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                 "-property", "installationPath"],
                capture_output=True, text=True,
            )
            lines = r.stdout.strip().splitlines()
            if r.returncode == 0 and lines:
                bat = Path(lines[0].strip()) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                if bat.exists():
                    return bat
    if PORTABLE_MSVC_BAT.exists():
        return PORTABLE_MSVC_BAT
    return None


def parse_set_output(text):
    """Parse the output of the cmd 'set' command into a dict."""
    env = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if not sep or not key or key.startswith("="):
            continue
        env[key] = value
    return env


def load_env_from_bat(bat):
    """Run a .bat in cmd and copy the environment it leaves behind into this process."""
    # The whole string is passed as is. /s makes cmd strip the outer quotes only.
    cmd = f'cmd /s /c "call "{bat}" >nul && set"'
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        fail(f"{bat} failed (exit {r.returncode}):\n{r.stdout}{r.stderr}")
    env = parse_set_output(r.stdout)
    if not any(k.upper() == "PATH" for k in env):
        fail(f"could not read the environment after running {bat}")
    os.environ.update(env)
    return env


def check_cl():
    cl = shutil.which("cl")
    if not cl:
        fail("cl.exe is not on PATH after loading the MSVC environment")
    r = subprocess.run([cl], capture_output=True, text=True, errors="replace")
    m = re.search(r"Version (\d+)\.(\d+)\.(\d+)", r.stdout + r.stderr)
    if not m:
        fail(f"could not read the cl version from: {cl}")
    major, minor = int(m.group(1)), int(m.group(2))
    print(f"  cl {m.group(1)}.{m.group(2)}.{m.group(3)} (MSVC toolset 14.{minor}) at {cl}")
    if (major, minor) >= (19, 50):
        print("  WARNING: CUDA 12.8 supports MSVC toolsets up to 14.44. This cl is newer,")
        print("  so nvcc builds may fail. Use portable-msvc.py with --msvc-version 14.44.")


def install_portable_msvc():
    """Download the pinned portable-msvc.py, check its hash, run it. Returns the setup bat."""
    print("  No Visual Studio C++ tools found. Setting up a portable MSVC, no admin needed.")
    TOOLS_DIR.mkdir(exist_ok=True)
    download_verified(MSVC_SCRIPT_URL, MSVC_SCRIPT_PATH, MSVC_SCRIPT_SHA256)
    base = [sys.executable, str(MSVC_SCRIPT_PATH), "--vs", "2022"]
    print()
    print("  The next step downloads MSVC from Microsoft. The script asks you to read and")
    print("  accept the Microsoft Visual Studio license. Answer in this window (Y or N).")
    print()
    # No stdin or stdout redirection on purpose: the user answers the license question.
    r = subprocess.run([*base, "--msvc-version", MSVC_VERSION], cwd=TOOLS_DIR)
    if r.returncode != 0 or not PORTABLE_MSVC_BAT.exists():
        print()
        print(f"  portable-msvc.py did not finish (exit {r.returncode}). Versions it offers:")
        subprocess.run([*base, "--show-versions"], cwd=TOOLS_DIR)
        fail(
            f"could not set up MSVC {MSVC_VERSION}. If you declined the license, run again and accept it.\n"
            "  If the version is not offered, edit MSVC_VERSION in setup_env.py to the highest 14.4x listed above."
        )
    shutil.rmtree(TOOLS_DIR / "downloads", ignore_errors=True)
    return PORTABLE_MSVC_BAT


def set_build_env():
    lib = lib_dir()
    os.environ.update(
        CUDA_HOME=str(lib),
        CUDA_PATH=str(lib),
        DISTUTILS_USE_SDK="1",
        MSSdk="1",
        TORCH_CUDA_ARCH_LIST="12.0",
        FORCE_CUDA="1",
    )
    # The recon mode builds gsplat without kernels; the world generation build needs them.
    os.environ.pop("BUILD_NO_CUDA", None)
    prepend_path("PATH", [lib / "bin", lib / "cmd"])
    # conda-forge puts the CUDA headers in Library\include and the .lib files in Library\lib.
    # torch only adds CUDA_HOME\lib\x64 to the linker, so add both through INCLUDE and LIB.
    prepend_path("INCLUDE", [lib / "include"])
    prepend_path("LIB", [lib / "lib", lib / "lib" / "x64"])


def patch_target_files(patch_text):
    """Paths from the '+++ b/...' lines of a unified diff."""
    files = []
    for line in patch_text.splitlines():
        if line.startswith("+++ "):
            path = line[4:].split("\t")[0].strip()
            if path == "/dev/null":
                continue
            if path.startswith("b/"):
                path = path[2:]
            files.append(path)
    return files


def files_missing_marker(repo, files):
    missing = []
    for rel in files:
        path = Path(repo) / rel
        if not path.is_file() or PATCH_MARKER not in read_text(path):
            missing.append(rel)
    return missing


def git_run(repo, *args):
    # Stop git from looking above the repo, so a repository around this folder
    # cannot make git apply skip files.
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=Path(repo).parent.as_posix())
    return subprocess.run(
        ["git", *args], cwd=repo, env=env, capture_output=True, text=True, errors="replace"
    )


def apply_worldgen_patch(repo, patch_path=None):
    patch_path = Path(patch_path or PATCH_PATH)
    if not patch_path.is_file():
        fail(f"patch file missing: {patch_path}")
    files = patch_target_files(read_text(patch_path))
    if not files:
        fail(f"no '+++ b/...' lines found in {patch_path}")
    patch = str(patch_path)
    if git_run(repo, "apply", "--check", "-R", patch).returncode == 0 \
            and not files_missing_marker(repo, files):
        print("  OK, already done")
        return
    r = git_run(repo, "apply", "--check", patch)
    if r.returncode != 0:
        fail(f"the patch does not apply to {repo} (upstream changed, or half applied):\n{r.stderr}")
    r = git_run(repo, "apply", patch)
    if r.returncode != 0:
        fail(f"git apply failed:\n{r.stderr}")
    missing = files_missing_marker(repo, files)
    if missing:
        fail(
            f"git apply ran but '{PATCH_MARKER}' is missing in: {', '.join(missing)}\n"
            "  git may have skipped files because the folder is inside another git repository."
        )
    print(f"  applied the patch to {len(files)} file(s)")


def gsplat_has_kernels():
    """True when the compiled gsplat.csrc file exists. Does not import gsplat, which could start a JIT build."""
    code = (
        "import glob, importlib.util, os, sys\n"
        "spec = importlib.util.find_spec('gsplat')\n"
        "if spec is None or not spec.submodule_search_locations:\n"
        "    sys.exit(1)\n"
        "d = list(spec.submodule_search_locations)[0]\n"
        "found = glob.glob(os.path.join(d, 'csrc*.pyd')) + glob.glob(os.path.join(d, 'csrc*.so'))\n"
        "sys.exit(0 if found else 1)\n"
    )
    return run_py(code, cwd=sys.prefix).returncode == 0


def pip_or_fail(*args, cwd=None, env=None):
    try:
        pip(*args, cwd=cwd, env=env)
    except subprocess.CalledProcessError:
        fail("pip failed (see output above). Send the last 40 lines of this window.")


def hf_status():
    code = (
        "import sys\n"
        "try:\n"
        "    from huggingface_hub import HfApi\n"
        "except ImportError:\n"
        "    print('NOHUB'); sys.exit(0)\n"
        "api = HfApi()\n"
        "try:\n"
        "    print('LOGIN', api.whoami().get('name'))\n"
        "except Exception:\n"
        "    print('NOLOGIN'); sys.exit(0)\n"
        "try:\n"
        "    check = getattr(api, 'auth_check', None)\n"
        "    if check:\n"
        "        check('facebook/sam3')\n"
        "    else:\n"
        "        api.model_info('facebook/sam3')\n"
        "    print('ACCESS')\n"
        "except Exception as e:\n"
        "    print('NOACCESS', type(e).__name__)\n"
    )
    r = run_py(code, cwd=sys.prefix)
    return r.stdout.split()


def hf_login():
    """Run the Hugging Face login attached to the console. This script never sees the token."""
    scripts = Path(sys.prefix) / "Scripts"
    hf = shutil.which("hf") or (str(scripts / "hf.exe") if (scripts / "hf.exe").exists() else None)
    if hf:
        cmd = [hf, "auth", "login"]
    else:
        old = shutil.which("huggingface-cli") or str(scripts / "huggingface-cli.exe")
        cmd = [old, "login"]
    print("  >", " ".join(cmd))
    try:
        subprocess.run(cmd)
    except OSError as e:
        print(f"  could not start the login command: {e}")


WORLDGEN_CHECKS = [
    ("recast", "import recast"),
    ("pytorch3d (with _C)", "import torch, pytorch3d, pytorch3d._C"),
    ("fused_ssim", "import torch, fused_ssim"),
    ("moge", "import moge"),
    ("nerfview", "import nerfview"),
    ("gsplat compiled module", "import torch\nfrom gsplat import csrc\nprint(csrc.__file__)"),
    ("cupy kernel (NVRTC)",
     "import cupy as cp\n"
     "k = cp.ElementwiseKernel('float32 x', 'float32 y', 'y = x * 2', 'wg_setup_check')\n"
     "total = float(k(cp.arange(8, dtype=cp.float32)).sum())\n"
     "assert total == 56.0, total"),
]


def setup_worldgen(repo):
    total = 10
    if os.name != "nt":
        fail("worldgen setup is for Windows only")
    prepend_path("PATH", [lib_dir() / "bin", lib_dir() / "cmd"])

    step(1, total, "conda packages: git and the CUDA 12.8 toolchain (conda-forge)")
    # cuda-nvcc, cuda-cudart-dev and cuda-libraries-dev are the conda-forge split packages
    # for CUDA 12 (compiler, runtime headers and libs, library headers and libs incl. NVRTC).
    # cuda-version=12.8 stops the solver from picking a newer CUDA. If the solver rejects the
    # split packages, fall back to the cuda-toolkit metapackage, which pulls in the same set.
    pkgs = []
    if shutil.which("git") is None:
        pkgs.append("git")
    need_cuda = not nvcc_is_128()
    if pkgs or need_cuda:
        split = ["cuda-version=12.8", "cuda-nvcc=12.8", "cuda-cudart-dev=12.8", "cuda-libraries-dev=12.8"]
        ok = conda_install(*pkgs, *(split if need_cuda else []))
        if not ok and need_cuda:
            print("  split packages failed, trying cuda-toolkit=12.8")
            ok = conda_install(*pkgs, "cuda-version=12.8", "cuda-toolkit=12.8")
        if not ok:
            fail("conda install failed (see output above)")
        prepend_path("PATH", [lib_dir() / "bin", lib_dir() / "cmd"])
        if shutil.which("git") is None:
            fail("git is still not on PATH after installing it")
        if not nvcc_is_128():
            fail(f"nvcc 12.8 not found at {lib_dir() / 'bin' / 'nvcc.exe'} after installing")
    else:
        print("  OK, already done")

    step(2, total, "MSVC (C++ compiler for the CUDA and pybind11 builds)")
    bat = find_vcvars()
    if bat is None:
        bat = install_portable_msvc()
    print(f"  using {bat}")
    load_env_from_bat(bat)
    check_cl()
    TOOLS_DIR.mkdir(exist_ok=True)
    write_text(MSVC_ENV_FILE, str(bat) + "\n")

    step(3, total, "Build environment (CUDA_HOME, CUDA_PATH, DISTUTILS_USE_SDK, arch 12.0)")
    set_build_env()
    for var in ("CUDA_HOME", "CUDA_PATH", "DISTUTILS_USE_SDK", "TORCH_CUDA_ARCH_LIST"):
        print(f"  {var}={os.environ[var]}")
    if not (lib_dir() / "include" / "cuda_runtime.h").exists():
        print(f"  WARNING: {lib_dir() / 'include' / 'cuda_runtime.h'} not found, CUDA builds will fail")

    step(4, total, "Apply patches/worldgen_windows.patch")
    apply_worldgen_patch(repo)

    step(5, total, "recastnavigation sources (the upstream ZIP has no submodules)")
    recast_dir = repo / RECAST_REL
    if (recast_dir / "Recast" / "Source" / "Recast.cpp").exists():
        print("  OK, already done")
    else:
        print(f"  downloading commit {RECAST_COMMIT[:7]}")
        fetch_zip_folder(RECAST_URL, recast_dir, "recastnavigation")
        if not (recast_dir / "Recast" / "Source" / "Recast.cpp").exists():
            fail(f"Recast.cpp not found under {recast_dir} after extracting")

    step(6, total, "pip install --no-build-isolation -r requirements_git.txt")
    req_git = repo / "requirements_git.txt"
    if not req_git.exists():
        fail(f"{req_git} not found (upstream changed?)")
    marker = Path(sys.prefix) / "hyworld_worldgen_git.sha256"
    if marker.exists() and marker.read_text().strip() == sha256_of(req_git):
        print("  OK, already done")
    else:
        print("  NOTE: pytorch3d and fused-ssim compile CUDA code. This takes a long time")
        print("  (often 20 to 60 minutes) and the window can look stuck. Let it run.")
        # With --no-build-isolation pip uses the build tools already in the env.
        # requirements.txt brings scikit-build-core, nanobind and pybind11, but spz also needs
        # CMake, which upstream expects on the system. Install it from pip (no admin).
        if not module_available("cmake"):
            pip_or_fail("install", "cmake")
        pip_or_fail("install", "--no-build-isolation", "-r", str(req_git))
        marker.write_text(sha256_of(req_git) + "\n")

    step(7, total, "navmesh (recast python module)")
    if run_py("import recast", cwd=sys.prefix).returncode == 0:
        print("  OK, already done")
    else:
        # Official command. setup.py finds recastnavigation at its default relative path.
        pip_or_fail("install", ".", "--no-build-isolation", cwd=repo / NAVMESH_REL)

    step(8, total, "gsplat_maskgaussian with CUDA kernels (official build, needed by 3DGS training)")
    # The worldgen README says gsplat_maskgaussian must be compiled from source. The recon
    # mode skipped the kernels with BUILD_NO_CUDA=1; here the official command runs without it.
    if gsplat_has_kernels():
        print("  OK, already done")
    else:
        glm_dir = repo / GLM_REL
        if not (glm_dir / "glm" / "glm.hpp").exists():
            print(f"  glm headers are missing in the upstream copy, downloading gsplat's pin {GLM_COMMIT[:7]}")
            fetch_zip_folder(GLM_URL, glm_dir, "glm")
        print("  building the kernels, this takes several minutes")
        pip_or_fail("install", "-e", ".", "--no-build-isolation",
                    cwd=repo / GSPLAT_REL, env=dict(os.environ))
        if not gsplat_has_kernels():
            fail("gsplat was rebuilt but no compiled csrc module was found")

    step(9, total, "Hugging Face login and SAM3 access (needed for stages 1 and 3)")
    status = hf_status()
    if "NOLOGIN" in status:
        print("  Not logged in to Hugging Face. Starting the login, paste your own token when asked")
        print("  (create a read token at https://huggingface.co/settings/tokens).")
        hf_login()
        status = hf_status()
    if "ACCESS" in status:
        print(f"  OK, logged in as {status[1] if len(status) > 1 else '?'} with access to facebook/sam3")
    elif "NOACCESS" in status:
        print("  Logged in, but no access to facebook/sam3 yet. Setup continues.")
        print("  Request access once at https://huggingface.co/facebook/sam3 (gated, approved by Meta).")
        print("  Stages 1 and 3 will fail until it is approved.")
    else:
        print(f"  Hugging Face is not ready (status: {' '.join(status) or 'unknown'}). Setup continues.")
        print("  Run  hf auth login  later, and request access at https://huggingface.co/facebook/sam3")

    step(10, total, "Verify imports")
    failed = []
    for label, code in WORLDGEN_CHECKS:
        r = run_py(code, cwd=sys.prefix)
        if r.returncode == 0:
            print(f"  OK      {label}")
        else:
            failed.append(label)
            tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
            print(f"  FAILED  {label}")
            for line in tail:
                print(f"          {line}")
    if failed:
        fail("verification failed for: " + ", ".join(failed))
    print("WORLDGEN ENV OK (no models were downloaded)")




# ---------------------------------------------------------------------------
# vLLM for stages 1 and 2 (community Windows build, own conda env)
# ---------------------------------------------------------------------------

def check_driver():
    """The Windows vLLM build needs CUDA 13, which needs NVIDIA driver R580 or newer."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
    except OSError:
        fail("nvidia-smi not found. The NVIDIA driver is missing or not on PATH.")
    lines = r.stdout.strip().splitlines()
    if r.returncode != 0 or not lines:
        fail(f"nvidia-smi failed: {r.stderr.strip() or r.stdout.strip()}")
    version = lines[0].strip()
    m = re.match(r"(\d+)", version)
    if not m:
        fail(f"could not read the driver version from: {version}")
    print(f"  NVIDIA driver {version}")
    if int(m.group(1)) < MIN_DRIVER_MAJOR:
        fail(
            f"NVIDIA driver {version} is too old for the Windows vLLM build (needs {MIN_DRIVER_MAJOR} or newer, for CUDA 13).\n"
            "  The driver cannot be updated without admin rights. Ask IT for driver 580 or newer,\n"
            "  then run setup_vlm.bat again. Everything else in the setup is not affected."
        )


def vllm_installed_version():
    r = run_py("import importlib.metadata as m\nprint(m.version('vllm'))", cwd=sys.prefix)
    return r.stdout.strip() if r.returncode == 0 else None


def setup_vlm():
    total = 3
    step(1, total, "NVIDIA driver (R580 or newer needed)")
    check_driver()

    step(2, total, f"vLLM {VLLM_VERSION} Windows build")
    have = vllm_installed_version()
    if have and have.startswith(VLLM_VERSION):
        print(f"  OK, already done (vllm {have})")
    else:
        TOOLS_DIR.mkdir(exist_ok=True)
        wheel = TOOLS_DIR / VLLM_WHEEL_NAME
        print("  NOTE: this downloads about 220 MB for vLLM plus PyTorch 2.11 (cu130), several GB.")
        download_verified(VLLM_WHEEL_URL, wheel, VLLM_WHEEL_SHA256)
        pip_or_fail("install", str(wheel), "--extra-index-url", VLLM_EXTRA_INDEX)

    step(3, total, "Verify")
    r = run_py("import vllm; print(vllm.__version__)", cwd=sys.prefix)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout).strip().splitlines()[-5:]
        fail("import vllm failed:\n  " + "\n  ".join(tail))
    print(f"  vllm {r.stdout.strip()}")
    exe = Path(sys.prefix) / "Scripts" / "vllm.exe"
    r = subprocess.run([str(exe), "--help"], capture_output=True, text=True, cwd=sys.prefix)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout).strip().splitlines()[-5:]
        fail("vllm --help failed:\n  " + "\n  ".join(tail))
    print("  vllm --help OK")
    print("VLLM ENV OK (no models were downloaded)")




# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Set up the HY-World 2.0 conda env.")
    parser.add_argument("mode", choices=["recon", "pano", "worldgen", "vlm", "driver"])
    parser.add_argument("repo", help="path to the HY-World-2.0 checkout")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    if args.mode not in ("vlm", "driver") and not repo.is_dir():
        fail(f"repo folder not found: {repo}")

    print(f"Python {sys.version.split()[0]} in {sys.prefix}")
    if args.mode == "driver":
        check_driver()
        return
    if args.mode == "vlm":
        setup_vlm()
        print()
        print("Setup finished.")
        return
    print(f"Repo: {repo}")
    if args.mode == "recon":
        setup_recon(repo)
    elif args.mode == "pano":
        setup_pano(repo)
    else:
        setup_worldgen(repo)
    print()
    print("Setup finished.")


if __name__ == "__main__":
    main()
