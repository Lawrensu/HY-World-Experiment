# HY-World 2.0 Experimentation

Experiments with the official Tencent HY-World 2.0 on the Windows lab PC (RTX 5070, no admin rights). Everything follows the official docs, except the changes the lab PC forces, which are listed under Deviations.

This document covers:

* Setting up the lab PC with one click
* Experiment 1: how many stump images WorldMirror needs
* Experiment 2: WorldMirror on the Amorphophallus 360 from the frontend
* Experiment 3: the full pipeline on the Amorphophallus front view
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

For Experiment 3, double click `setup_worldgen.bat` instead of `setup.bat`. It runs the World Reconstruction setup, then installs everything World Generation needs without admin rights, then the vLLM server env, and opens a ready prompt in `HY-World-2.0\hyworld2\worldgen`. The first run takes hours, mostly compiling pytorch3d and gsplat.

* MSVC: uses Visual Studio if installed, otherwise downloads Microsoft's build tools into `tools\msvc`. You accept the Microsoft license in the window
* CUDA 12.8 compiler from conda-forge, into the env
* The official `requirements_git.txt`, navmesh and gsplat builds, and recastnavigation at the commit upstream pins
* Hugging Face login: paste your own read token when asked
* vLLM: needs NVIDIA driver 580 or newer. An older driver only skips this part, everything else still installs

Before the first run, request access once at https://huggingface.co/facebook/sam3. Stages 1 and 3 need it.


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
* Above 24 images the lab PC froze and restarted with DPC_WATCHDOG_VIOLATION. Stay at 24 or below, or use `--target_size 518`
* To get an out of memory error instead of a freeze, set NVIDIA Control Panel > Manage 3D settings > CUDA Sysmem Fallback Policy to Prefer No Sysmem Fallback, if the PC allows it



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



## Experiment 3: full pipeline

The official World Generation pipeline, run on one 12 GB GPU. It starts from one image, the front view `inputs\amorphophallus\front_frame_0148.png`. Each stage writes into one scene folder that the next stage reads.

1. HY-Pano (stage 0): from the `setup_pano.bat` prompt, run the probe. It prints RAM, GPU memory, time and the result, and logs to `runs\pano\probe_log.jsonl`

```bat
python ..\..\..\probe_pano.py --image ..\..\..\inputs\amorphophallus\front_frame_0148.png
```

2. Create `scenes\amorphophallus\` in this folder and copy the panorama into it as `panorama.png`
3. Start the model server: double click `start_vlm.bat` and leave that window open
4. In the `setup_worldgen.bat` prompt, set the scene path once:

```bat
set "TARGET_PATH=D:\hyw\HY-World-Experimentation\scenes\amorphophallus"
```

Stage 1, WorldNav:

```bat
python traj_generate.py --target_path %TARGET_PATH% --llm_addr 127.0.0.1 --llm_port 8000 --llm_name Qwen/Qwen3-VL-8B-Instruct-FP8 --apply_nav_traj --apply_up_route --apply_recon_iteration --force_vlm
```

Stage 2, trajectory rendering:

```bat
torchrun --nproc_per_node 1 traj_render.py --target_path %TARGET_PATH% --llm_addr 127.0.0.1 --llm_port 8000 --llm_name Qwen/Qwen3-VL-8B-Instruct-FP8
```

Stage 3, WorldStereo. Close the `start_vlm.bat` window first to free the GPU:

```bat
set "HYWORLD_LOWMEM=1"
```

```bat
torchrun --nproc_per_node 1 video_gen.py --target_path %TARGET_PATH%
```

Stage 4, GS training data:

```bat
torchrun --nproc_per_node 1 gen_gs_data.py --root_path %TARGET_PATH% --save_normal --split_sky
```

Stage 5, 3DGS training, with the single GPU step counts from the official notes (every step count scaled by 8000 / 1500):

```bat
python -m world_gs_trainer default --data_dir %TARGET_PATH%\gs_data --result_dir %TARGET_PATH%\gs_result --max_steps 8000 --save_steps 8000 --eval_steps 8000 --ply_steps 8000 --save_ply --convert_to_spz --disable_video --use_scale_regularization --antialiased --depth_loss --normal_loss --sky_depth_from_pcd --use_mask_gaussian --mask_export_stochastic --no-mask-export-anchor-protection --use_anchor_protection --export_mesh --strategy.refine-start-iter 800 --strategy.refine-stop-iter 4000 --strategy.refine-every 533 --strategy.refine-scale2d-stop-iter 4000 --strategy.reset-every 99990 --strategy.grow-grad2d 0.0001 --strategy.prune-scale3d 0.1
```


### Notes

* HY-Pano with CPU offload needs about 64 GB of system RAM and runs slowly
* Keep Task Manager open on Memory. If RAM stays at 100 percent, press Ctrl + C before the PC locks up
* Stage 3 is the risky one: a 14B to 17B video model on 12 GB of GPU memory and 64 GB of RAM. Expect hours, or a stop at loading. Record where it stops
* Stage 2 captions use a prompt upstream hardcoded for a campfire scene, so they will describe the plant scene oddly. This is upstream behaviour and is left as is
* If `start_vlm.bat` runs out of memory, lower `MAX_LEN` or `GPU_UTIL`, or raise `CPU_OFFLOAD`, at the top of the file
* Note per stage: time, peak GPU memory, peak RAM, and the last lines if it stops



## Deviations

* conda-forge channel for the envs, to avoid the Anaconda terms of service prompt
* cu128 torch installed first, because the Windows PyPI torch is CPU only and the 5070 needs CUDA 12.8
* `requirements.txt`: `cupy-cuda12x==13.6.0` instead of `cupy==13.6.0`, and `pycolmap==3.11.1` instead of `3.10.0`
* `gsplat_maskgaussian` installed with `BUILD_NO_CUDA=1` for World Reconstruction only, since reconstruction only imports gsplat. `setup_worldgen.bat` builds it with the official command
* FlashAttention skipped, and `attention.py` patched to use torch SDPA
* `onnxruntime` and `requests` installed, because the sky mask code imports them
* HY-Pano: cu128 instead of cu118, `hyworld2\panogen\requirements.txt`, and CPU offload instead of `.to("cuda")` in `probe_pano.py`
* HY-Pano: `peft==0.18.1` added, the pin from the root `requirements.txt`, because the panogen file leaves it out and diffusers needs it to load the LoRA
* World Generation: `patches\worldgen_windows.patch` (12 files, each change marked `HY-World Windows patch`): gloo when NCCL is missing, Windows path handling, single process runs without device mesh or FSDP, no DataLoader workers on Windows
* Stage 3 low memory mode, only with `HYWORLD_LOWMEM=1`: CPU offload, bf16 text and image encoders, no `torch.compile`
* Build tools without admin: portable MSVC 14.44 and the conda CUDA 12.8 compiler instead of Visual Studio and the CUDA installer
* gsplat glm headers at the commit gsplat 1.5.3 pins, because the HY-World copy leaves them out
* CMake and Ninja from pip, because spz needs CMake. CMake uses the Ninja generator, because its default Visual Studio generator needs an installed Visual Studio
* navmesh is built with the official command from a short copy in `%USERPROFILE%\hywb`, with `RECAST_PATH` pointing at it, because its object file paths pass the 260 character limit in deep folders such as the Desktop
* zlib 1.3.1 for spz from zlib's GitHub release, checked against the hash spz pins, because spz's zlib.net link is dead
* VLM: Qwen3-VL-8B in Qwen's official FP8 version, served by the community Windows build of vLLM 0.32.0 with CPU offload
* `torchrun --nproc_per_node 1` and no `--fsdp`, because there is one GPU
