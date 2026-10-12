"""
tools/cloud_video_generator.py

Builds and saves the execution specification (JSON) for the Kaggle/Colab
cloud notebook. This module does NOT run any video inference locally.

It no longer contains a pixel-warp fallback.  If the cloud notebook cannot
run, we return a clear error — we never silently substitute warped still
images for genuine AI-generated video.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def build_run_spec(
    reference_image_path: str,
    output_dir: str = "./output",
) -> Dict[str, Any]:
    """
    Builds a JSON specification that the cloud notebook reads at startup.

    The spec encodes every parameter needed for reproducible generation.
    Both the Python orchestrator and the notebook use pipeline_config.py
    as their single source of truth — they do not duplicate constants.

    Returns the spec dict and writes it to output_dir/run_report.json.
    """
    import pipeline_config as cfg

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_path = Path(reference_image_path)
    if not image_path.exists():
        raise FileNotFoundError(
            f"Reference image not found: {image_path}\n"
            "Upload your Anaya outfit image to assets/ before running."
        )

    spec: Dict[str, Any] = {
        "schema_version": "3.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "model_id": cfg.ACTIVE_MODEL_ID,
            "fallback_model_id": cfg.MODEL_ID_1_3B,
            "framework": "diffusers.WanImageToVideoPipeline",
            "dtype": "torch.bfloat16",
        },
        "scheduler": {
            "class": "FlowMatchEulerDiscreteScheduler",
            "flow_shift": cfg.FLOW_SHIFT,
            "note": (
                "flow_shift is applied via FlowMatchEulerDiscreteScheduler.from_config("
                "pipe.scheduler.config, shift=FLOW_SHIFT), NOT via config dict mutation."
            ),
        },
        "conditioning": {
            "reference_image": str(image_path.resolve()),
            "resize_strategy": "letterbox",
            "target_width": cfg.VIDEO_WIDTH,
            "target_height": cfg.VIDEO_HEIGHT,
            "background_fill_rgb": [245, 240, 235],
        },
        "generation": {
            "prompt": cfg.MOTION_PROMPT,
            "negative_prompt": cfg.NEGATIVE_PROMPT,
            "width": cfg.VIDEO_WIDTH,
            "height": cfg.VIDEO_HEIGHT,
            "num_frames": cfg.NUM_FRAMES,
            "fps": cfg.OUTPUT_FPS,
            "guidance_scale": cfg.GUIDANCE_SCALE,
            "num_inference_steps": cfg.NUM_INFERENCE_STEPS,
            "seed": cfg.SEED,
        },
        "output": {
            "directory": str(output_path.resolve()),
            "filename": cfg.OUTPUT_FILENAME,
            "codec": "libx264",
            "pixel_format": "yuv420p",
            "contact_sheet_stride": cfg.CONTACT_SHEET_STRIDE,
        },
        "quality_gates": {
            "min_motion_score": cfg.MIN_MOTION_SCORE,
            "max_motion_score": cfg.MAX_MOTION_SCORE,
        },
        "status": "SPEC_READY__AWAITING_CLOUD_EXECUTION",
        "result": None,
    }

    spec_file = output_path / cfg.SPEC_FILENAME
    with open(spec_file, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)

    logger.info(f"Run specification written to: {spec_file}")
    logger.info("Next step: upload and run cloud_notebooks/kaggle_wan21_i2v.ipynb on Kaggle.")
    logger.info("Manual 'Run All' is required — no automated remote execution is configured.")
    return spec


def print_run_instructions(spec: Dict[str, Any]) -> None:
    """Prints honest, actionable instructions for running the notebook."""
    print("\n" + "=" * 70)
    print("  CLOUD VIDEO GENERATION — MANUAL STEPS REQUIRED")
    print("=" * 70)
    print(f"\n  Model : {spec['model']['model_id']}")
    print(f"  Size  : {spec['generation']['width']}×{spec['generation']['height']}")
    print(f"  Frames: {spec['generation']['num_frames']} @ {spec['generation']['fps']} fps")
    print(f"  Steps : {spec['generation']['num_inference_steps']}")
    print(f"  CFG   : {spec['generation']['guidance_scale']}")
    print(f"  Seed  : {spec['generation']['seed']}")
    print()
    print("  STEPS:")
    print("  1. Go to https://www.kaggle.com/code → New Notebook")
    print("  2. Import: cloud_notebooks/kaggle_wan21_i2v.ipynb")
    print("  3. Add Data → Dataset 'anaya-assets'  (upload if not created yet)")
    print("  4. Settings → Accelerator → GPU T4 x2")
    print("  5. Settings → Internet → ON")
    print("  6. Click Run All  (expect 20-35 minutes on T4)")
    print("  7. Output tab → download anaya_fashion_reel.mp4")
    print()
    print("  ⚠  T4 (16 GB VRAM) is marginal for the 14B model.")
    print("     The notebook will attempt offloading. If OOM occurs,")
    print("     set USE_SMALL_MODEL = True in Cell 2 to use the 1.3B model.")
    print("=" * 70 + "\n")
