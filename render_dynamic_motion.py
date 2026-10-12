"""
render_dynamic_motion.py

⚠  SYNTHETIC IMAGE WARP — NOT AI-GENERATED VIDEO ⚠

This script applies pixel-displacement warping to still images to produce
a fake-motion MP4.  It is retained for debugging and visual reference ONLY.

Output of this script must NEVER be:
  - Labelled as AI-generated video
  - Used as a substitute for genuine diffusion model output
  - Published to social media as a fashion reel
  - Returned from the pipeline with status="success"

To generate real AI video, run the cloud notebook:
  cloud_notebooks/kaggle_wan21_i2v.ipynb

To run this debug warp:
  python render_dynamic_motion.py --confirm-synthetic
"""

import argparse
import glob
import sys
from pathlib import Path

import imageio
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates


def parse_args():
    p = argparse.ArgumentParser(
        description="Synthetic pixel-warp debug renderer (NOT real AI video)"
    )
    p.add_argument(
        "--confirm-synthetic",
        action="store_true",
        required=True,
        help="Required flag. Confirms you understand this is NOT generative AI video.",
    )
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=832)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--frames", type=int, default=81)
    p.add_argument("--output", type=str, default="output/DEBUG_SYNTHETIC_WARP.mp4")
    return p.parse_args()


def run_synthetic_warp(W: int, H: int, FPS: int, NUM_FRAMES: int, output_path: Path):
    dress_paths = sorted(glob.glob("assets/dress/lakdata_Women_Tiered_Pur_hd_angle_*.jpg"))
    if not dress_paths:
        raise FileNotFoundError("No dress images found in assets/dress/")

    base_poses = []
    for p in dress_paths:
        img = Image.open(p).convert("RGB")
        dw, dh = img.size
        # Letterbox (preserve full body)
        scale = min(W / dw, H / dh)
        nw, nh = int(dw * scale), int(dh * scale)
        resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (W, H), (245, 240, 235))
        canvas.paste(resized, ((W - nw) // 2, (H - nh) // 2))
        base_poses.append(np.array(canvas, dtype=np.float32))

    num_angles = len(base_poses)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n⚠  SYNTHETIC WARP — output is NOT AI-generated video")
    print(f"   Writing to: {output_path}")

    writer = imageio.get_writer(str(output_path), fps=FPS, codec="libx264",
                                output_params=["-pix_fmt", "yuv420p"])
    grid_y, grid_x = np.mgrid[0:H, 0:W].astype(np.float32)

    for f in range(NUM_FRAMES):
        t = f / FPS
        prog = f / max(NUM_FRAMES - 1, 1)
        ai = prog * (num_angles - 1)
        bi, ni = int(np.floor(ai)), min(int(np.floor(ai)) + 1, num_angles - 1)
        bw = ai - bi
        bs = bw * bw * (3.0 - 2.0 * bw)  # smooth-step
        frame = base_poses[bi] * (1.0 - bs) + base_poses[ni] * bs

        y_frac = grid_y / H
        sway_mask = np.clip((y_frac - 0.45) / 0.55, 0.0, 1.0)
        fabric = np.sin(grid_y * 0.015 + t * 4.0) * (4.0 * sway_mask)
        hair_mask = np.clip((0.35 - y_frac) / 0.35, 0.0, 1.0)
        hair = np.sin(grid_x * 0.02 + t * 3.5) * (2.5 * hair_mask)
        bob = -np.abs(np.sin(t * np.pi * 1.6)) * 8.0
        sway = np.sin(t * np.pi * 0.8) * 6.0

        wx = np.clip(grid_x - sway - fabric - hair, 0, W - 1)
        wy = np.clip(grid_y - bob, 0, H - 1)

        channels = [
            map_coordinates(frame[:, :, c], [wy, wx], order=1, mode="nearest")
            for c in range(3)
        ]
        warped = np.clip(np.stack(channels, axis=-1), 0, 255).astype(np.uint8)
        writer.append_data(warped)

    writer.close()
    print(f"   Done. Label: SYNTHETIC_IMAGE_WARP — do not publish or report as AI video.\n")


if __name__ == "__main__":
    args = parse_args()
    run_synthetic_warp(
        W=args.width,
        H=args.height,
        FPS=args.fps,
        NUM_FRAMES=args.frames,
        output_path=Path(args.output),
    )
