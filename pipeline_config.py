"""
pipeline_config.py — Single Authoritative Configuration for Anaya Sharma Video Pipeline.

ALL model IDs, dimensions, frame counts, FPS, prompts, scheduler settings, and
seed values live HERE. The Kaggle notebook and the Python tools both import or
copy-paste from this source so they can never drift apart.

Do NOT hardcode these values anywhere else.
"""

# ---------------------------------------------------------------------------
# MODEL SELECTION
# ---------------------------------------------------------------------------
# Wan2.1-I2V-14B-480P requires ~18-22 GB VRAM at bfloat16 — exceeds T4 (16 GB).
# Use the officially supported diffusers pipeline with CPU offloading + VAE slicing.
# If OOM persists, the notebook will fall back to the 1.3B model.
# Never silently fall back to pixel-warp synthesis.

MODEL_ID_480P    = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
MODEL_ID_1_3B    = "Wan-AI/Wan2.1-I2V-1.3B-480P-Diffusers"

# Which model to use for the Kaggle T4 run.
# Start with 14B + offloading; switch to 1_3B if OOM.
ACTIVE_MODEL_ID  = MODEL_ID_480P

# ---------------------------------------------------------------------------
# RESOLUTION  (must match model's supported dimensions exactly)
# ---------------------------------------------------------------------------
# Wan2.1-I2V-480P officially supports 480×832 (portrait 9:16).
# Do NOT pass arbitrary dimensions — the model's VAE tiling is fixed.
VIDEO_WIDTH      = 480
VIDEO_HEIGHT     = 832

# ---------------------------------------------------------------------------
# GENERATION PARAMETERS
# ---------------------------------------------------------------------------
NUM_FRAMES       = 81       # ~5.1 s at 16 fps  (model default, fits T4 RAM)
OUTPUT_FPS       = 16

# guidance_scale: 5.0 recommended for 480P I2V.
# Higher values (≥6) suppress motion diversity → slideshow-like output.
GUIDANCE_SCALE   = 5.0

# Steps: 50 gives smoother motion than 40; 30 if time-constrained.
NUM_INFERENCE_STEPS = 50

# Fixed seed for reproducibility. Change only when retrying a bad generation.
SEED             = 42

# ---------------------------------------------------------------------------
# SCHEDULER — flow_shift (VERIFIED against HF docs)
# ---------------------------------------------------------------------------
# flow_shift must be applied via FlowMatchEulerDiscreteScheduler.from_config()
# NOT via pipe.scheduler.config['shift'] = … (that does nothing at runtime).
# Recommended value: 3.0 for 480P, 5.0 for 720P.
FLOW_SHIFT       = 3.0

# ---------------------------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------------------------
# Rule: describe TEMPORAL ACTION, not scene appearance.
# Motion verbs must lead. Target ~90 words for sufficient context.

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

# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------
OUTPUT_FILENAME  = "anaya_fashion_reel.mp4"
SPEC_FILENAME    = "run_report.json"

# Contact sheet: save one frame every N frames for visual review
CONTACT_SHEET_STRIDE = 10   # saves ~8 frames for a 81-frame clip

# Quality gate thresholds (diagnostic, not guarantees)
MIN_MOTION_SCORE  = 2.0    # mean abs pixel diff first→last; below = likely frozen
MAX_MOTION_SCORE  = 40.0   # above = likely identity drift / camera jitter

# ---------------------------------------------------------------------------
# PLATFORM PATHS  (isolated here so notebook and Python can both reference them)
# ---------------------------------------------------------------------------
KAGGLE_INPUT_DATASET = "anaya-assets"          # name of your Kaggle dataset
KAGGLE_ASSET_DIR  = f"/kaggle/input/{KAGGLE_INPUT_DATASET}"
KAGGLE_OUTPUT_DIR = "/kaggle/working"

COLAB_ASSET_DIR   = "/content"
COLAB_OUTPUT_DIR  = "/content/output"

LOCAL_ASSET_DIR   = "assets"
LOCAL_OUTPUT_DIR  = "output"
