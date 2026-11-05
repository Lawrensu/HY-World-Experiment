"""Set up the Python side of the HY-World 2.0 environment on the Windows lab PC.

Runs inside the activated conda env (setup.bat does that). Stdlib only at top level.
Every step checks first and only installs what is missing, so reruns are fast.

Usage: python setup_env.py {recon,pano} REPO
"""

import argparse
import ctypes
import hashlib
import os
import re
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
    total = 4
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

    step(4, total, "Verify")
    require_cuda(cwd=repo)
    mem = memory_gb()
    if mem is None:
        print("system RAM: unknown (not Windows)")
    else:
        print(f"system RAM: {mem[0]:.1f} GB free of {mem[1]:.1f} GB total")
    print("PANO ENV OK (no models were downloaded)")




# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Set up the HY-World 2.0 conda env.")
    parser.add_argument("mode", choices=["recon", "pano"])
    parser.add_argument("repo", help="path to the HY-World-2.0 checkout")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        fail(f"repo folder not found: {repo}")

    print(f"Python {sys.version.split()[0]} in {sys.prefix}")
    print(f"Repo: {repo}")
    if args.mode == "recon":
        setup_recon(repo)
    else:
        setup_pano(repo)
    print()
    print("Setup finished.")


if __name__ == "__main__":
    main()
