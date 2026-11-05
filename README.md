# HY-World 2.0 Experimentation

Experiments with the official Tencent HY-World 2.0 on the Windows lab PC (RTX 5070, no admin rights). Everything follows the official docs, except the changes the lab PC forces, which are listed under Deviations.

This document covers:

* Setting up the lab PC with one click
* Experiment 1: how many stump images WorldMirror needs
* Experiment 2: WorldMirror on the Amorphophallus 360 from the frontend
* Experiment 3: probing the full pipeline on the Amorphophallus front view
* Deviations from the official docs



## Related

* Official repo: https://github.com/Tencent-Hunyuan/HY-World-2.0
* WorldMirror docs: `HY-World-2.0/DOCUMENTATION.md`
* HY-Pano docs: `HY-World-2.0/hyworld2/panogen/README.md`
* World generation docs: `HY-World-2.0/hyworld2/worldgen/README.md`
* How the stump sets were picked: `stump_selection/stump_selection.json`, `stump_selection/camera_map.png`



## Setup

1. Copy this whole folder to the lab PC, to a path with no spaces, for example `D:\hyw\HY-World-Experimentation`
2. Double click `setup.bat`. The first run downloads and installs everything and takes a while. Later runs only check, then open a ready Command Prompt in `HY-World-2.0` with `(hyworld2)` active
3. The setup ends with `PIPELINE OK`. If a step fails, the window stays open and names the step

For Experiment 3 only, also double click `setup_pano.bat` once. It creates the separate `hyworld2-pano` env from the HY-Pano docs.


### Notes

* Run `setup.bat` after every reboot. It installs only what is missing
* Model weights download on first use into `hf_cache\` in this folder
* World generation weights are not downloaded by the setup



## Experiment 1: stump image count

Question: does adding images reconstruct the whole surroundings, or only what the cameras saw?

Every Gaussian comes from a pixel of an input image, so unseen areas stay empty. The sets test how coverage and quality grow with more images, and where the 12 GB limit is.

* `full_orbit_010` to `full_orbit_109`: nested sets spread evenly around the stump. Up to 48 they hold only stump-centred photos, from 64 on they add background photos
* `half_orbit_010` to `half_orbit_032`: one side only, the same counts. The far side should come out empty
* `matched_048_with_background`: 32 stump photos plus 16 background, compared with `full_orbit_048`
* `test_016`: held-out views, never used as input

Run each set from the ready prompt, smallest first:

```bat
python -m hyworld2.worldrecon.pipeline --input_path ..\inputs\stump\full_orbit_010 --enable_bf16 --no_interactive
```

When a set runs out of GPU memory, repeat it with `--target_size 518` and stop that ladder.


### Notes

* Record per run: images, target size, time, peak GPU memory (Task Manager), and how it looks in the viewer
* Results land in `inference_output\` inside `HY-World-2.0`



## Experiment 2: Amorphophallus 360

The input is the exact 360 the frontend serves: `inputs\amorphophallus\orbit_front_centre.mp4` (1280x720, 240 frames, 238 degrees), and the served frames at 8, 16, 24 and 32.

From the frames:

```bat
python -m hyworld2.worldrecon.pipeline --input_path ..\inputs\amorphophallus\frames_016 --enable_bf16 --no_interactive
```

From the video, with the official frame selection:

```bat
python -m hyworld2.worldrecon.pipeline --input_path ..\inputs\amorphophallus\orbit_front_centre.mp4 --video_max_frames 32 --enable_bf16 --no_interactive
```

To view it in the frontend, upload `gaussians.ply` with Upload .ply splat on the Amorphophallus admin page.


### Notes

* The footage is AI generated, and the plant faces the camera in every frame while the background turns. Expect a clean background arc and a smeared plant
* If the splat looks hazy or see-through, the opacity needs converting before upload



## Experiment 3: full pipeline probe

The full pipeline starts from one image: the front view `inputs\amorphophallus\front_frame_0148.png`. HY-Pano turns it into a panorama, then WorldNav, WorldStereo and WorldMirror with 3DGS build the world.

From the `setup_pano.bat` prompt:

```bat
python ..\..\..\probe_pano.py --image ..\..\..\inputs\amorphophallus\front_frame_0148.png
```

It prints RAM, GPU memory, time and the result, and logs to `runs\pano\probe_log.jsonl`.


### Notes

* HY-Pano with CPU offload needs about 64 GB of system RAM and runs slowly
* The later stages cannot run on this PC, because:
  * the vLLM server they need does not run on Windows
  * the navmesh extension and the 3DGS trainer must be compiled, which needs admin
  * several stages need NCCL multi-GPU communication, which does not exist on Windows
  * WorldStereo needs about 68 GB of weights
  * the official docs test on 8x H20 GPUs



## Deviations

* conda-forge channel for the envs, to avoid the Anaconda terms of service prompt
* cu128 torch installed first, because the Windows PyPI torch is CPU only and the 5070 needs CUDA 12.8
* `requirements.txt`: `cupy-cuda12x==13.6.0` instead of `cupy==13.6.0`, and `pycolmap==3.11.1` instead of `3.10.0`
* `gsplat_maskgaussian` installed with `BUILD_NO_CUDA=1`, since reconstruction only imports gsplat
* FlashAttention skipped, and `attention.py` patched to use torch SDPA
* `onnxruntime` and `requests` installed, because the sky mask code imports them
* HY-Pano: cu128 instead of cu118, `hyworld2\panogen\requirements.txt`, and CPU offload instead of `.to("cuda")` in `probe_pano.py`
