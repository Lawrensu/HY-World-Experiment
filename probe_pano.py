"""
Probe: run Tencent HY-World 2.0 HY-Pano 2.0 (Qwen-Image-Edit backend) on one
12 GB GPU and record whether it fits.

It reuses the official code in <repo>/hyworld2/panogen/pipeline_with_qwen_image.py
(HunyuanPanoPipeline, PanoDiffusionPipeline, the default model id, LoRA path and
subfolder, prompt templates, height, width, steps, guidance, blend, crop and
seed). Nothing is copied from it and nothing in the repo is modified.

The single deviation: upstream's from_pretrained ends with .to("cuda"), which
puts the whole ~20B transformer plus the 7B text encoder on the GPU and runs out
of memory on 12 GB. Here the same pipeline is built the same way, but the .to("cuda")
is replaced by diffusers CPU offloading (--offload sequential or model).
The LoRA is loaded before the offload hooks are attached, because the hooks
move weights around and the LoRA call expects normal modules.

Usage:
    python probe_pano.py --image input.png
    python probe_pano.py --image input.png --offload model --steps 20 --seed 1
"""

import argparse
import ctypes
import json
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ============================================================
# System info
# ============================================================

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


def get_ram_gb():
    """Return (total, available) system RAM in GB using GlobalMemoryStatusEx."""
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return stat.ullTotalPhys / 1024 ** 3, stat.ullAvailPhys / 1024 ** 3


def parse_args():
    parser = argparse.ArgumentParser("Probe HY-Pano 2.0 (Qwen-Image-Edit backend) on a small GPU")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path (default: runs/pano/<image stem>_<offload>.png next to this script)")
    parser.add_argument("--offload", type=str, default="sequential", choices=["sequential", "model"],
                        help="sequential: layer by layer, slowest, smallest GPU use. "
                             "model: whole components, likely OOM on 12 GB, kept for comparison")
    parser.add_argument("--repo", type=str, default=str(HERE / "HY-World-2.0"),
                        help="Path to the official HY-World-2.0 repo")
    parser.add_argument("--steps", type=int, default=None, help="Default: upstream's default")
    parser.add_argument("--seed", type=int, default=None, help="Default: upstream's default")
    parser.add_argument("--prompt", type=str, default=None, help="Default: upstream's default")
    return parser.parse_args()


def write_log(log_path, record):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="ascii") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    # Resolve every user path to absolute before anything else
    image_path = Path(args.image).resolve()
    repo = Path(args.repo).resolve()
    panogen = repo / "hyworld2" / "panogen"
    out_path = Path(args.out).resolve() if args.out else (
        HERE / "runs" / "pano" / f"{image_path.stem}_{args.offload}.png")
    log_path = HERE / "runs" / "pano" / "probe_log.jsonl"

    if not image_path.is_file():
        sys.exit(f"probe stopped: input image not found: {image_path}")
    if not (panogen / "pipeline_with_qwen_image.py").is_file():
        sys.exit(f"probe stopped: official panogen code not found at {panogen}")

    # Upstream imports its siblings as top level modules (from qwen_image import ...),
    # so panogen must be on sys.path, as when running its CLI from that folder.
    sys.path.insert(0, str(panogen))

    import torch
    from pipeline_with_qwen_image import HunyuanPanoPipeline
    from qwen_image import PanoDiffusionPipeline

    total_ram, avail_ram = get_ram_gb()
    print(f"System RAM: {total_ram:.1f} GB total, {avail_ram:.1f} GB available")
    if not torch.cuda.is_available():
        sys.exit("probe stopped: CUDA is not available in this environment")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, {props.total_memory / 1024 ** 3:.1f} GB VRAM")
    if total_ram < 64:
        print("WARNING: RAM is below 64 GB. Sequential offload keeps the full bf16 model "
              "in RAM and may fail.")

    # Only pass through what the user gave, so upstream defaults apply otherwise
    kwargs = {}
    if args.steps is not None:
        kwargs["num_inference_steps"] = args.steps
    if args.seed is not None:
        kwargs["seed"] = args.seed
    if args.prompt is not None:
        kwargs["prompt"] = args.prompt

    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "image": str(image_path),
        "offload": args.offload,
        "steps": args.steps if args.steps is not None else "upstream_default",
        "seed": args.seed if args.seed is not None else "upstream_default",
        "ram_total_gb": round(total_ram, 1),
        "ram_available_gb": round(avail_ram, 1),
        "gpu": props.name,
        "vram_gb": round(props.total_memory / 1024 ** 3, 1),
        "success": False,
        "stage": None,
        "error_type": None,
    }

    stage = "load"
    t_load = t_infer = None
    try:
        torch.cuda.reset_peak_memory_stats()

        # Same as HunyuanPanoPipeline.from_pretrained, minus .to("cuda")
        t0 = time.time()
        model_id = HunyuanPanoPipeline.DEFAULT_MODEL_ID
        print(f"[Probe] Loading base model {model_id} ...")
        pipe = PanoDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

        stage = "LoRA"
        print("[Probe] Loading LoRA weights ...")
        pipe.load_lora_weights(
            HunyuanPanoPipeline.DEFAULT_LORA_PATH,
            subfolder=HunyuanPanoPipeline.DEFAULT_LORA_SUBFOLDER,
            weight_name="pytorch_lora_weights.safetensors",
            torch_dtype=torch.bfloat16,
        )

        stage = "offload setup"
        # The one deviation: offload instead of .to("cuda")
        if args.offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.enable_model_cpu_offload()
        # Built the way from_pretrained does it: cls(pipe)
        pipeline = HunyuanPanoPipeline(pipe)
        t_load = time.time() - t0
        print(f"[Probe] Model load time: {t_load:.1f} s")

        stage = "inference"
        t1 = time.time()
        output = pipeline(image_path, **kwargs)
        t_infer = time.time() - t1

        out_path.parent.mkdir(parents=True, exist_ok=True)
        output.save(out_path)
        record["success"] = True
        record["stage"] = "done"
    except torch.cuda.OutOfMemoryError as e:
        record["stage"], record["error_type"] = stage, type(e).__name__
        print(f"probe stopped: GPU out of memory at {stage}")
    except MemoryError as e:
        record["stage"], record["error_type"] = stage, type(e).__name__
        print(f"probe stopped: system RAM out of memory at {stage}")
    except Exception as e:
        record["stage"], record["error_type"] = stage, type(e).__name__
        print(f"probe stopped: {type(e).__name__} at {stage}: {e}")

    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    record.update({
        "load_seconds": round(t_load, 1) if t_load is not None else None,
        "inference_seconds": round(t_infer, 1) if t_infer is not None else None,
        "peak_vram_gb": round(peak, 2),
        "output": str(out_path) if record["success"] else None,
    })
    write_log(log_path, record)

    print(f"Model load time:  {record['load_seconds']} s")
    print(f"Inference time:   {record['inference_seconds']} s")
    print(f"Peak GPU memory:  {peak:.2f} GB (max_memory_allocated)")
    if record["success"]:
        print(f"Output saved to {out_path}")
    print(f"Log line appended to {log_path}")
    sys.exit(0 if record["success"] else 1)


if __name__ == "__main__":
    main()
