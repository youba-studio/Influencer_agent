"""
make_notebook.py — Generates the production Kaggle notebook.

Run locally (no GPU needed):
    python make_notebook.py

This writes cloud_notebooks/kaggle_wan21_i2v.ipynb
Then upload THAT file to Kaggle.
"""

import json
import pathlib

OUTPUT_PATH = pathlib.Path("cloud_notebooks/kaggle_wan21_i2v.ipynb")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# All cells as plain strings — NO hardcoded tokens anywhere
# ---------------------------------------------------------------------------

CELL_0_MARKDOWN = """\
# Anaya Sharma — Wan2.1 I2V Fashion Reel Generator
## Production Kaggle Notebook (v3.0)

**Before running:**
1. Add your `anaya-assets` dataset (contains outfit image + anaya_avatar.jpg)
2. Set Accelerator → **GPU T4 x2**
3. Set Internet → **ON**
4. Add your HuggingFace token as a **Kaggle Secret** named `HF_TOKEN`
   (Notebook sidebar → Add-ons → Secrets → Add New Secret)

**This notebook generates REAL AI video using Wan2.1 Image-to-Video diffusion.**
It does NOT use pixel warping, Ken Burns zoom, or image crossfading.
If GPU inference fails for any reason, the notebook will report failure clearly.
"""

CELL_1_GPU_CHECK = """\
# CELL 1: GPU & Environment Check
import subprocess, sys, os, platform

print("=" * 60)
print("  ENVIRONMENT CHECK")
print("=" * 60)

result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
if result.returncode != 0:
    raise RuntimeError(
        "No GPU detected. Enable GPU: Settings → Accelerator → GPU T4 x2"
    )
print(result.stdout)

import torch
print(f"PyTorch     : {torch.__version__}")
print(f"CUDA avail  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    total_vram_gb = props.total_memory / 1e9
    print(f"GPU         : {props.name}")
    print(f"VRAM        : {total_vram_gb:.1f} GB")
    if total_vram_gb < 14:
        print("WARNING: Less than 14 GB VRAM — will force 1.3B model.")
    elif total_vram_gb < 18:
        print("NOTE: ~16 GB VRAM — will use 14B with CPU offloading.")
        print("      If OOM occurs, set USE_SMALL_MODEL = True in Cell 2.")
print("=" * 60)
"""

CELL_2_CONFIG = """\
# CELL 2: Configuration  (single source of truth — matches pipeline_config.py)
# ──────────────────────────────────────────────────────────────────────────────
# Set USE_SMALL_MODEL = True if you get CUDA out-of-memory on T4.
# The 1.3B model is faster and fits easily; quality is lower but still real AI motion.
USE_SMALL_MODEL = False

import os
from pathlib import Path

# ── Model ──────────────────────────────────────────────────────────────────
MODEL_ID_14B  = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
MODEL_ID_1_3B = "Wan-AI/Wan2.1-I2V-1.3B-480P-Diffusers"
MODEL_ID = MODEL_ID_1_3B if USE_SMALL_MODEL else MODEL_ID_14B

# ── Resolution (Wan2.1 480P official supported dimensions) ─────────────────
W, H = 480, 832      # 9:16 portrait — do NOT change to arbitrary values

# ── Generation parameters ──────────────────────────────────────────────────
NUM_FRAMES        = 81      # ~5.1 s at 16 fps
OUTPUT_FPS        = 16
GUIDANCE_SCALE    = 5.0     # 5.0 for 480P; higher values suppress motion
NUM_STEPS         = 50      # 50 for quality; 30 if time-constrained
SEED              = 42      # change to 123 / 777 if result is too static

# ── Scheduler ──────────────────────────────────────────────────────────────
# flow_shift MUST be applied via from_config(), not by mutating config dict.
FLOW_SHIFT = 3.0   # 3.0 recommended for 480P

# ── Paths ───────────────────────────────────────────────────────────────────
ASSET_DIR  = Path("/kaggle/input/anaya-assets")
OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Auto-detect if running on Colab instead
if not ASSET_DIR.exists():
    ASSET_DIR  = Path("/content")
    OUTPUT_DIR = Path("/content/output")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Colab path detected.")

AVATAR_PATH      = ASSET_DIR / "anaya_avatar.jpg"
OUTPUT_VIDEO     = OUTPUT_DIR / "anaya_fashion_reel.mp4"
CONTACT_SHEET    = OUTPUT_DIR / "contact_sheet.jpg"
RUN_REPORT_PATH  = OUTPUT_DIR / "run_report.json"

# ── Prompts ─────────────────────────────────────────────────────────────────
# Describe TEMPORAL ACTION, not static appearance.
# Motion verbs must come first. ~90 words is the effective range.
MOTION_PROMPT = (
    "An elegant young Indian woman presents her outfit in a natural full-body fashion shot. "
    "She maintains a relaxed standing pose, gently shifts her weight once, makes a subtle "
    "head movement, softly smiles, and lets the skirt and loose fabric respond naturally "
    "to the small body motion. Her hair moves slightly with the motion. "
    "The camera stays nearly fixed. Continuous realistic human motion, coherent anatomy, "
    "natural lighting, realistic fabric drape and folds. "
    "Preserve the exact outfit design, colour and pattern from the input image."
)

NEGATIVE_PROMPT = (
    "frozen, static image, no movement, slideshow, Ken Burns zoom, artificial pan, "
    "sudden pose change, camera cut, identity drift, facial flicker, face shape change, "
    "clothing redesign, changing prints or colours, disappearing fabric, unstable hem, "
    "extra limbs, missing limbs, deformed hands, floating body parts, feet sliding, "
    "phone, props, additional people, text, logo, border, watermark, "
    "excessive hair motion, implausible physics, blurry, low quality, cartoon, CGI"
)

print(f"Model      : {MODEL_ID}")
print(f"Resolution : {W}x{H}  |  Frames: {NUM_FRAMES}  |  FPS: {OUTPUT_FPS}")
print(f"Guidance   : {GUIDANCE_SCALE}  |  Steps: {NUM_STEPS}  |  Seed: {SEED}")
print(f"Flow shift : {FLOW_SHIFT}  (applied via scheduler.from_config)")
print(f"Asset dir  : {ASSET_DIR}  (exists: {ASSET_DIR.exists()})")
print(f"Avatar     : {AVATAR_PATH}  (exists: {AVATAR_PATH.exists()})")
print(f"Output dir : {OUTPUT_DIR}")
print()
print(f"Prompt ({len(MOTION_PROMPT.split())} words):")
print(MOTION_PROMPT)
"""

CELL_3_INSTALL = """\
# CELL 3: Install Dependencies  (~3-5 minutes on first run)
import subprocess

def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:])
    return r.returncode == 0

print("Installing diffusers and video stack...")
assert run("pip install -q --upgrade diffusers transformers accelerate sentencepiece")
assert run("pip install -q imageio imageio-ffmpeg pillow numpy scipy opencv-python-headless")
print("All dependencies installed.")
"""

CELL_4_HF_AUTH = """\
# CELL 4: HuggingFace Authentication
# Reads HF_TOKEN from Kaggle Secrets (recommended) or environment variable.
# Never hardcode your token here.
from huggingface_hub import login
import os

# Try Kaggle secret first, then env var
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle Secrets.")
except Exception:
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        print("HF_TOKEN loaded from environment variable.")
    else:
        raise RuntimeError(
            "HF_TOKEN not found!\\n"
            "Add it: Notebook sidebar → Add-ons → Secrets → HF_TOKEN"
        )

login(token=hf_token, add_to_git_credential=False)
print("HuggingFace authentication: OK")
"""

CELL_5_LOAD_MODEL = """\
# CELL 5: Load Wan2.1 I2V Model
# Applies the correct flow_shift via scheduler.from_config() — NOT dict mutation.
import torch
import gc
from diffusers import WanImageToVideoPipeline, FlowMatchEulerDiscreteScheduler

print(f"Loading model: {MODEL_ID}")
print("First run downloads ~15-30 GB. Subsequent runs use cache.")
print("This may take 5-15 minutes...")

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

try:
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        token=hf_token,
    )
except Exception as e:
    raise RuntimeError(
        f"Model load failed: {e}\\n"
        f"If authentication error: verify HF_TOKEN has read access to {MODEL_ID}\\n"
        f"If disk/memory error: set USE_SMALL_MODEL = True in Cell 2"
    )

# ── Apply flow_shift correctly ──────────────────────────────────────────────
# The ONLY correct way: reconstruct scheduler via from_config with shift param.
# Mutating pipe.scheduler.config['shift'] after-the-fact does NOT work reliably.
try:
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        shift=FLOW_SHIFT,
    )
    print(f"Scheduler: FlowMatchEulerDiscreteScheduler  flow_shift={FLOW_SHIFT}")
except Exception as e:
    print(f"Warning: Could not set flow_shift ({e}). Proceeding with default scheduler.")

# ── Memory optimisations (critical for T4 16 GB) ───────────────────────────
pipe.enable_model_cpu_offload()   # offloads unused layers to CPU RAM
pipe.enable_vae_slicing()          # decode VAE slice-by-slice to save VRAM

gc.collect()
torch.cuda.empty_cache()

vram_used  = round(torch.cuda.memory_allocated() / 1e9, 2)
vram_total = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
print(f"Model loaded. VRAM used: {vram_used} / {vram_total} GB")
"""

CELL_6_PREP_IMAGE = """\
# CELL 6: Prepare Conditioning Image (letterbox — preserves full outfit)
import glob
from PIL import Image
import numpy as np

# Find dress images; fall back to avatar if none found
dress_paths = sorted(glob.glob(str(ASSET_DIR / "lakdata_Women_Tiered_Pur_hd_angle_*.jpg")))
print(f"Found {len(dress_paths)} dress reference images.")
for p in dress_paths:
    print(f"  {p}")

PRIMARY_PATH = dress_paths[0] if dress_paths else str(AVATAR_PATH)
print(f"\\nUsing conditioning image: {PRIMARY_PATH}")

raw = Image.open(PRIMARY_PATH).convert("RGB")
orig_w, orig_h = raw.size
print(f"Original size: {orig_w}x{orig_h}")

# Letterbox: scale to fit WxH while preserving aspect ratio
# Never stretch — squishing distorts body proportions and confuses the model
scale = min(W / orig_w, H / orig_h)
new_w, new_h = int(orig_w * scale), int(orig_h * scale)
resized = raw.resize((new_w, new_h), Image.Resampling.LANCZOS)

# Neutral warm background — avoids stark black bars that confuse diffusion
canvas = Image.new("RGB", (W, H), (245, 240, 235))
paste_x = (W - new_w) // 2
paste_y = (H - new_h) // 2
canvas.paste(resized, (paste_x, paste_y))
conditioning_image = canvas

canvas.save(str(OUTPUT_DIR / "conditioning_frame.jpg"))
print(f"Conditioning image prepared: {W}x{H} (letterboxed, full outfit preserved)")
"""

CELL_7_GENERATE = """\
# CELL 7: Generate Video — Real Wan2.1 I2V Diffusion
# ─────────────────────────────────────────────────────────────────────────────
# This cell runs the actual neural network inference.
# Expected time: 15-30 min on T4 (14B), 5-10 min (1.3B)
# If CUDA OOM: set USE_SMALL_MODEL = True in Cell 2 and re-run from Cell 5.
# ─────────────────────────────────────────────────────────────────────────────
import torch, time

print("Starting Wan2.1 I2V generation...")
print(f"  Frames : {NUM_FRAMES}  |  Steps : {NUM_STEPS}  |  Guidance : {GUIDANCE_SCALE}")
print(f"  Seed   : {SEED}")
print(f"  Prompt : {MOTION_PROMPT[:120]}...")
print()

generation_start = time.time()
generation_error = None
raw_frames = None

try:
    with torch.inference_mode():
        output = pipe(
            image=conditioning_image,
            prompt=MOTION_PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            height=H,
            width=W,
            num_frames=NUM_FRAMES,
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=NUM_STEPS,
            generator=torch.Generator("cuda").manual_seed(SEED),
        )
    raw_frames = output.frames[0]  # list of PIL Images
    generation_time = round(time.time() - generation_start, 1)
    print(f"Generation complete in {generation_time}s — {len(raw_frames)} frames produced.")

except torch.cuda.OutOfMemoryError as e:
    generation_error = f"CUDA OOM: {e}"
    print(f"\\nOUT OF MEMORY ERROR: {generation_error}")
    print("Fix: set USE_SMALL_MODEL = True in Cell 2, then re-run from Cell 5.")
    raise

except Exception as e:
    generation_error = str(e)
    print(f"\\nGENERATION FAILED: {generation_error}")
    raise
"""

CELL_8_VALIDATE = """\
# CELL 8: Quality Validation
# Distinguishes too-little-motion from plausible motion from excessive drift.
# These checks are DIAGNOSTIC — they do not guarantee perceptual quality.
import numpy as np

if raw_frames is None:
    raise RuntimeError("No frames to validate — generation did not complete.")

frames_np = [np.array(f).astype(np.float32) for f in raw_frames]

# ── 1. Motion score: mean absolute pixel difference first→last ────────────
f0, f_last = frames_np[0], frames_np[-1]
motion_score = float(np.mean(np.abs(f_last - f0)))
print(f"Motion score (first→last frame diff): {motion_score:.2f}")
print(f"  < {2.0} = likely frozen/slideshow")
print(f"  {2.0}–{40.0} = plausible subject movement")
print(f"  > {40.0} = possible identity drift or camera jitter")

# ── 2. Temporal consistency: frame-to-frame mean diff ────────────────────
frame_diffs = [
    float(np.mean(np.abs(frames_np[i+1] - frames_np[i])))
    for i in range(0, len(frames_np) - 1, 5)   # sample every 5 frames
]
mean_jitter = float(np.mean(frame_diffs))
max_jitter  = float(np.max(frame_diffs))
print(f"\\nTemporal consistency (sampled every 5 frames):")
print(f"  Mean frame-to-frame diff : {mean_jitter:.2f}")
print(f"  Max  frame-to-frame diff : {max_jitter:.2f}")
if max_jitter > 30:
    print("  WARNING: High max jitter — may indicate a scene cut or identity flip.")

# ── 3. Determine review status ────────────────────────────────────────────
if motion_score < 2.0:
    review_status = "NEEDS_REVIEW__TOO_STATIC"
    print(f"\\nStatus: {review_status}")
    print("  Retry: change SEED to 123 or 777, lower GUIDANCE_SCALE to 4.0")
elif motion_score > 40.0:
    review_status = "NEEDS_REVIEW__EXCESSIVE_MOTION"
    print(f"\\nStatus: {review_status}")
    print("  Retry: raise GUIDANCE_SCALE to 6.0, change SEED")
else:
    review_status = "MOTION_OK__REVIEW_VISUALLY"
    print(f"\\nStatus: {review_status}")
    print("  Motion is in the plausible range. Review the contact sheet below.")

print("\\n  *** Always review the video manually before publishing. ***")
"""

CELL_9_EXPORT = """\
# CELL 9: Export Clean MP4
import imageio, numpy as np

if raw_frames is None:
    raise RuntimeError("No frames — cannot export. Fix generation errors first.")

print(f"Exporting {len(raw_frames)} frames to {OUTPUT_VIDEO} ...")

writer = imageio.get_writer(
    str(OUTPUT_VIDEO),
    fps=OUTPUT_FPS,
    codec="libx264",
    output_params=["-pix_fmt", "yuv420p", "-crf", "18"],
)
for frame in raw_frames:
    writer.append_data(np.array(frame))
writer.close()

import os
size_mb = round(os.path.getsize(str(OUTPUT_VIDEO)) / 1e6, 2)
print(f"Exported: {OUTPUT_VIDEO}  ({size_mb} MB)")
print("No text overlays, no watermarks, no borders — clean 9:16 footage only.")
"""

CELL_10_CONTACT_SHEET = """\
# CELL 10: Save Contact Sheet (representative frames for visual review)
import numpy as np
from PIL import Image

STRIDE = 10   # one frame every STRIDE frames
sampled = raw_frames[::STRIDE]
print(f"Contact sheet: {len(sampled)} frames sampled every {STRIDE} frames")

cols = min(4, len(sampled))
rows = (len(sampled) + cols - 1) // cols
thumb_w, thumb_h = W // 2, H // 2

sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (30, 30, 30))
for idx, frame in enumerate(sampled):
    thumb = frame.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    r, c = divmod(idx, cols)
    sheet.paste(thumb, (c * thumb_w, r * thumb_h))

sheet.save(str(CONTACT_SHEET))
print(f"Contact sheet saved: {CONTACT_SHEET}")

# Display inline
from IPython.display import display
display(sheet.resize((sheet.width // 2, sheet.height // 2)))
"""

CELL_11_RUN_REPORT = """\
# CELL 11: Save Run Report JSON
import json, datetime, torch

report = {
    "schema_version": "3.0",
    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "model": {
        "model_id": MODEL_ID,
        "dtype": str(dtype),
    },
    "scheduler": {
        "class": "FlowMatchEulerDiscreteScheduler",
        "flow_shift": FLOW_SHIFT,
    },
    "generation": {
        "prompt_word_count": len(MOTION_PROMPT.split()),
        "width": W,
        "height": H,
        "num_frames": NUM_FRAMES,
        "fps": OUTPUT_FPS,
        "guidance_scale": GUIDANCE_SCALE,
        "num_inference_steps": NUM_STEPS,
        "seed": SEED,
        "generation_time_s": generation_time if 'generation_time' in dir() else None,
    },
    "hardware": {
        "gpu_name": torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else "N/A",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else 0,
    },
    "quality": {
        "motion_score": round(motion_score, 3) if 'motion_score' in dir() else None,
        "mean_jitter": round(mean_jitter, 3) if 'mean_jitter' in dir() else None,
        "max_jitter": round(max_jitter, 3) if 'max_jitter' in dir() else None,
        "review_status": review_status if 'review_status' in dir() else "UNKNOWN",
    },
    "output": {
        "video": str(OUTPUT_VIDEO),
        "contact_sheet": str(CONTACT_SHEET),
    },
    "generation_method": "WAN2.1_I2V_DIFFUSION__GENUINE_AI_VIDEO",
}

with open(str(RUN_REPORT_PATH), "w") as f:
    json.dump(report, f, indent=2)

print("Run report saved:", RUN_REPORT_PATH)
print()
print(json.dumps(report, indent=2))
"""

CELL_12_DISPLAY = """\
# CELL 12: Display Output
from IPython.display import Video, display

print("Final output video:")
print(f"  {OUTPUT_VIDEO}")
print()
print("Download from the Output tab (right sidebar) when notebook finishes.")
print()
# Inline preview (may not work in all Kaggle versions)
try:
    display(Video(str(OUTPUT_VIDEO), width=320))
except Exception:
    print("(Inline preview not available — download from Output tab)")
"""

# ---------------------------------------------------------------------------
# Assemble notebook
# ---------------------------------------------------------------------------

def make_cell(source_str: str, cell_type: str = "code") -> dict:
    lines = [line + "\\n" for line in source_str.splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\\n")
    if cell_type == "markdown":
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": lines,
        }
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


notebook = {
    "metadata": {
        "accelerator": "GPU",
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
    "cells": [
        make_cell(CELL_0_MARKDOWN, "markdown"),
        make_cell(CELL_1_GPU_CHECK),
        make_cell(CELL_2_CONFIG),
        make_cell(CELL_3_INSTALL),
        make_cell(CELL_4_HF_AUTH),
        make_cell(CELL_5_LOAD_MODEL),
        make_cell(CELL_6_PREP_IMAGE),
        make_cell(CELL_7_GENERATE),
        make_cell(CELL_8_VALIDATE),
        make_cell(CELL_9_EXPORT),
        make_cell(CELL_10_CONTACT_SHEET),
        make_cell(CELL_11_RUN_REPORT),
        make_cell(CELL_12_DISPLAY),
    ],
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

nb_bytes = OUTPUT_PATH.stat().st_size
print(f"Notebook written: {OUTPUT_PATH}  ({nb_bytes:,} bytes, {len(notebook['cells'])} cells)")
print()
print("Upload this file to Kaggle, then follow the steps printed by main.py.")
