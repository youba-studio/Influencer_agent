# Anaya Sharma AI Influencer — Complete Architecture, Change Log & Workflow

> **This file is the single source of truth for all changes, fixes, and how to run the pipeline.**
> Updated every time a change is made.

---

## CHANGE LOG (Newest First)

---

### CHANGE 3 — Motion Fix: Slideshow Eliminated (October 2026)

**Problem:** The generated video was still a slideshow / static image — no real walking, no fabric physics, no hair movement.

**Root Cause Analysis (6 causes found):**

| # | Problem | Root Cause | Fix |
|:--|:---|:---|:---|
| 1 | Slideshow output | Prompt described **what** the scene looks like, not **how** it moves. Wan2.1 reads prompts as temporal scripts — motion verbs must come FIRST | Rewrote as 90-word "director's script" starting with camera action and body movement |
| 2 | Motion suppressed | `guidance_scale=6.0` — high CFG forces strict frame-locking, kills creative motion | Lowered to `5.0` |
| 3 | Scheduler too conservative | Default `flow_shift` value in scheduler was near-zero → near-static output | Set `pipe.scheduler.config['shift'] = 3.0` |
| 4 | Not enough denoising steps | `num_inference_steps=40` — insufficient for fluid body motion | Raised to `50` |
| 5 | Image distortion | `resize(W, H)` was squishing/stretching the dress image → broken body proportions confuse the motion model | Replaced with letterbox padding (full body preserved on neutral background) |
| 6 | Vague negative prompt | Didn't explicitly exclude "static image, frozen, no movement" | Added specific anti-static terms |

**Files Changed:**
- `cloud_notebooks/wan21_cloud_video_generator.ipynb` — all 6 fixes applied
- `tools/video_pipeline.py` — same fixes for local GPU runs
- `make_notebook.py` — notebook generator updated

**New Motion-First Prompt (CORRECT):**
```
The camera slowly dollies forward toward a young Indian woman who walks gracefully
toward the lens, her purple tiered anarkali dress swaying with each confident step,
fabric tiers rippling softly in the breeze, long dark wavy hair gently flowing,
arms swinging naturally at her sides, warm golden sunlight streaming through large
windows, her lips curling into a subtle smile, looking directly into the camera,
photorealistic 4K, cinematic depth of field, smooth continuous motion
```

**Old Prompt (WRONG — caused slideshow):**
```
Cinematic 9:16 vertical fashion reel, a young Indian woman with long dark wavy hair
wearing a chic purple tiered anarkali dress, walking gracefully toward the camera...
```
*(Too short, describes the subject not the motion, no camera direction)*

**New Negative Prompt:**
```
static image, frozen, no movement, slideshow, Ken Burns zoom, pan effect,
jitter, shake, deformed, floating head, bad anatomy, disconnected neck,
watermark, text overlay, banner, logo, blurry, low resolution, cartoon,
3d render, CGI, artificial, plastic skin, overexposed
```

**Parameter Changes:**

| Parameter | Old Value | New Value | Reason |
|:---|:---|:---|:---|
| `guidance_scale` | 6.0 | **5.0** | High CFG suppresses motion diversity |
| `num_inference_steps` | 40 | **50** | More steps = smoother motion |
| `scheduler.config['shift']` | default | **3.0** | flow_shift amplifies motion amplitude |
| Image prep | `resize(W,H)` stretch | **letterbox + padding** | Preserves full body for motion signal |
| Resolution | 720×1280 | **480×832** | T4 GPU compatible (480P model) |

**Motion Score Checker added (Cell 7 / pipeline):**
After generation, the pipeline now automatically measures pixel difference between first and last frame:
- Score < 3 → warns you to change seed or lower guidance
- Score 5–15 → visible motion, good
- Score > 15 → strong motion, excellent

**If still getting low motion, try:**
1. Change `SEED = 42` → `SEED = 123` or `SEED = 777` in Cell 7
2. Lower `GUIDANCE_SCALE` further to `4.0` or `3.5`
3. Ensure `flow_shift` was applied (Cell 5 should print `Scheduler flow_shift set to 3.0`)

---

### CHANGE 2 — InsightFace Face-Swap Integration (October 2026)

**Problem:** Previous version did naive PIL head crop + paste → floating head, bad neck alignment, hard edges, wrong skin tone.

**Fix:** Completely replaced with InsightFace GPU face-swap pipeline.

**How it works:**
1. `buffalo_l` model detects faces in every generated frame
2. `inswapper_128.onnx` performs seamless face identity transfer
3. Anaya's face identity is extracted from `assets/anaya_avatar.jpg`
4. Applied frame-by-frame — skin tone, lighting, and neck alignment happen automatically
5. `paste_back=True` blends edges natively (no hard cutout)

**Files Changed:**
- `cloud_notebooks/wan21_cloud_video_generator.ipynb` — Cell 8 added InsightFace swap
- `make_notebook.py` — notebook generator updated

**inswapper_128.onnx location options:**
- Upload as a separate Kaggle Dataset named `insightface-models`
- Or let Cell 8 auto-download it from HuggingFace (`deepinsight/insightface`)

---

### CHANGE 1 — Complete Pipeline Rebuild (October 2026)

**Problem:** Original scripts used:
- PIL crop + paste (floating head artifacts)
- Ken Burns zoom filter (fake motion, not real video)
- Burnt-in text banners on video frames
- Black bars and borders

**Fix:** Full wipe and rebuild of all files.

**What was built:**
- `config.py` — secure env loader (python-dotenv, never hardcoded)
- `tools/video_pipeline.py` — Wan2.1 I2V diffusers pipeline
- `tools/cloud_video_generator.py` — cloud orchestrator
- `tools/affiliate_engine.py` — EarnKaro link generator
- `tools/instagram_publisher.py` — Instagram Reels publisher
- `main.py` — full autonomous agent orchestrator
- `cloud_notebooks/wan21_cloud_video_generator.ipynb` — Kaggle/Colab notebook

---

## Repository Structure

```
/anaya_agent/
├── .env                          # Local credentials (git-ignored, NEVER commit)
├── .env.template                 # Template listing required variables
├── config.py                     # Secure env loader using python-dotenv
├── requirements.txt              # Minimal package list for local runs
├── context.md                    # THIS FILE — architecture & change log
├── main.py                       # Full autonomous agent orchestrator
├── make_notebook.py              # Script to regenerate the Kaggle notebook
│
├── assets/
│   ├── anaya_avatar.jpg          # Anaya's face reference (used by InsightFace)
│   └── dress/
│       ├── lakdata_Women_Tiered_Pur_hd_angle_01.jpg
│       ├── lakdata_Women_Tiered_Pur_hd_angle_02.jpg
│       ├── lakdata_Women_Tiered_Pur_hd_angle_03.jpg
│       ├── lakdata_Women_Tiered_Pur_hd_angle_04.jpg
│       └── lakdata_Women_Tiered_Pur_hd_angle_05.jpg
│
├── tools/
│   ├── video_pipeline.py         # Wan2.1 I2V local GPU pipeline (all motion fixes)
│   ├── cloud_video_generator.py  # Cloud GPU orchestrator + pixel-warp fallback
│   ├── affiliate_engine.py       # EarnKaro affiliate link builder
│   └── instagram_publisher.py   # Instagram Reels auto-publisher
│
├── cloud_notebooks/
│   └── wan21_cloud_video_generator.ipynb   # PRODUCTION notebook (12 cells)
│
└── output/
    ├── wan21_diffusers_spec.json            # Execution spec (auto-generated)
    └── anaya_real_motion.mp4                # Final clean 9:16 vertical video
```

---

## Environment Variables (`.env`)

| Variable | Example Value | Used By |
|:---|:---|:---|
| `HF_TOKEN` | `hf_TsWGoLQUov...` | Wan2.1 model download, InsightFace model download |
| `KAGGLE_USERNAME` | `yougestudio` | Kaggle API auth |
| `KAGGLE_KEY` | `KGAT_285bd4ad...` | Kaggle API token |
| `COLAB_API_KEY` | `AQ.Ab8RN6K_Q...` | Google Colab webhook |
| `INSTAGRAM_USER` | `Youbastudio.dev@gmail.com` | Instagram login |
| `INSTAGRAM_PASS` | `Dontshare@studio143` | Instagram password |
| `EARNKARO_USER` | *(set in .env)* | Affiliate publisher |
| `EARNKARO_PASS` | *(set in .env)* | Affiliate publisher |

---

## Wan2.1 I2V — Correct Settings (After All Fixes)

```python
# Model
MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"  # T4 GPU
# MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"  # A100 only

# Resolution
W, H = 480, 832   # 9:16 vertical (480P)

# Critical motion parameters
GUIDANCE_SCALE  = 5.0   # NOT 6.0 — higher values freeze motion
INFERENCE_STEPS = 50    # NOT 40 — more steps = smoother motion

# Pipeline setup (MUST include flow_shift)
pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
pipe.scheduler.config['shift'] = 3.0   # THE key motion amplifier
pipe.to("cuda")
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

# Inference call
output = pipe(
    image=conditioning_image,        # Letterboxed, NOT stretched
    prompt=PROMPT,                   # 90+ words, motion verbs FIRST
    negative_prompt=NEGATIVE_PROMPT, # Include "static image, frozen, no movement"
    height=H, width=W,
    num_frames=81,                   # ~5.1 seconds at 16fps
    guidance_scale=5.0,
    num_inference_steps=50,
    generator=torch.Generator("cuda").manual_seed(42)
)
```

---

## Output Specification

| Property | Value |
|:---|:---|
| Format | MP4 (H.264 codec, `yuv420p`) |
| Resolution | 480×832 (T4) or 720×1280 (A100) |
| Aspect Ratio | 9:16 vertical (Instagram Reels native) |
| Frame Rate | 16 fps |
| Duration | ~5.1 seconds (81 frames) |
| Text Overlays | **NONE** — strictly clean footage |
| Captions | Instagram post description only |
| Hashtags | Instagram post description only |
| Watermarks | **NONE** |

---

## HOW TO RUN — Complete Step-by-Step

---

### Option A: Kaggle (Recommended — FREE T4 GPU)

**Step 1: Create Kaggle Dataset named `anaya-assets`**
1. Go to [kaggle.com/datasets](https://kaggle.com/datasets) → **New Dataset**
2. Name: `anaya-assets`
3. Upload these files:
   - `assets/anaya_avatar.jpg`
   - `assets/dress/lakdata_Women_Tiered_Pur_hd_angle_01.jpg`
   - `assets/dress/lakdata_Women_Tiered_Pur_hd_angle_02.jpg`
   - `assets/dress/lakdata_Women_Tiered_Pur_hd_angle_03.jpg`
   - `assets/dress/lakdata_Women_Tiered_Pur_hd_angle_04.jpg`
   - `assets/dress/lakdata_Women_Tiered_Pur_hd_angle_05.jpg`
4. Click **Create** (can be Private)

**Step 2: (Optional but faster) Create Dataset named `insightface-models`**
- Upload `inswapper_128.onnx` (download from: https://huggingface.co/deepinsight/insightface)
- If you skip this, Cell 8 auto-downloads it — slower but works

**Step 3: Create Kaggle Notebook**
1. [kaggle.com/code](https://kaggle.com/code) → **New Notebook**
2. **File** → **Import Notebook** → upload `cloud_notebooks/wan21_cloud_video_generator.ipynb`
3. Right panel → **Add Data** → search `anaya-assets` → Add
4. Right panel → **Settings** → **Accelerator** → **GPU T4 x2**
5. **Run All** (Shift+Enter or top menu → Run All)

**Step 4: Download Output**
- After ~20-25 minutes → **Output tab** (right panel) → download `anaya_real_motion.mp4`

**What each cell does:**

| Cell | Action | Time |
|:---|:---|:---|
| 1 | Check GPU is active | < 1 min |
| 2 | Install all dependencies | 3–5 min |
| 3 | Set config, prompts, paths | < 1 min |
| 4 | Login to HuggingFace | < 1 min |
| 5 | Load Wan2.1 model + set flow_shift=3.0 | 5–10 min (downloads 15–30 GB first time) |
| 6 | Load and letterbox dress image | < 1 min |
| 7 | Generate 81 frames of true AI motion | **8–15 min** |
| 8 | InsightFace face-swap (Anaya's face) | 2–4 min |
| 9 | Export clean 9:16 MP4 | < 1 min |
| 10 | Preview first/last frame | < 1 min |
| 11 | (Optional) Multi-angle variants | 8–15 min each |

---

### Option B: Google Colab

1. Open [colab.research.google.com](https://colab.research.google.com) → Upload notebook
2. Runtime → **Change runtime type** → **T4 GPU** (or A100 if Pro+)
3. Upload files to `/content/`:
   ```python
   from google.colab import files
   files.upload()  # Upload anaya_avatar.jpg + all 5 dress images
   ```
4. In **Cell 3**, uncomment Colab paths:
   ```python
   AVATAR_PATH = Path('/content/anaya_avatar.jpg')
   DRESS_DIR   = Path('/content')
   OUTPUT_DIR  = Path('/content/output')
   ```
5. Run All → Download `output/anaya_real_motion.mp4` from the Files panel

---

### Option C: Local Python (Requires CUDA GPU)

```bash
# 1. Set active workspace
cd "C:\Users\youge\.gemini\antigravity-ide\scratch\anaya_agent"

# 2. Install base dependencies
pip install -r requirements.txt

# 3. Install GPU packages (requires CUDA GPU with 16GB+ VRAM)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/huggingface/diffusers.git
pip install transformers accelerate sentencepiece insightface onnxruntime-gpu
pip install imageio imageio-ffmpeg pillow numpy scipy opencv-python-headless

# 4. Run the full pipeline (affiliate + video + publish)
python main.py

# Or run just the video pipeline directly
python tools/video_pipeline.py
```

> **Note:** Wan2.1-14B requires 16GB+ VRAM for 480P. Use Kaggle for free access to T4 GPUs.

---

### Regenerate the Notebook (After Code Changes)

If you modify `make_notebook.py`, regenerate the `.ipynb` with:
```bash
cd "C:\Users\youge\.gemini\antigravity-ide\scratch\anaya_agent"
python make_notebook.py
```
Then re-upload the `.ipynb` to Kaggle.

---

## Troubleshooting

| Issue | Likely Cause | Fix |
|:---|:---|:---|
| Video is still a slideshow | `flow_shift` not applied or wrong prompt | Check Cell 5 printed "Scheduler flow_shift set to 3.0"; verify PROMPT starts with motion verb |
| Motion score < 3 | Bad seed for this image/prompt combination | Change `SEED = 42` → `123`, `777`, or `2024` in Cell 7 |
| CUDA out of memory | GPU VRAM too low for 720P | Set `USE_720P = False` in Cell 3, use 480P model |
| No face detected in Cell 8 | Avatar image is too small or blurry | Replace `anaya_avatar.jpg` with a clear, front-facing portrait (min 256×256) |
| inswapper_128 download fails | HF token doesn't have access | Upload `inswapper_128.onnx` manually as a Kaggle Dataset named `insightface-models` |
| InstallError on Kaggle | Old pip cache | Add `!pip cache purge` at start of Cell 2 |
| Instagram upload fails | instagrapi session expired | Run `python tools/instagram_publisher.py` locally to refresh session |
