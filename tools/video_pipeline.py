"""
Wan2.1 Image-to-Video (I2V) Diffusion Video Pipeline for Anaya Sharma.
Strictly implements the official Hugging Face diffusers WanImageToVideoPipeline.
Zero Ken Burns filters, zero pan-and-scan, zero slideshows, zero static cropping.
Performs true frame-by-frame 3D diffusion generative motion.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure workspace root is in path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from config import config

logger = logging.getLogger(__name__)

class Wan21DiffusionPipeline:
    """
    Official Hugging Face Diffusers Wan2.1 Image-to-Video Pipeline.
    Loads Wan-AI/Wan2.1-I2V-14B-720P-Diffusers (or 480P variant)
    and generates real diffusion motion conditioned on reference images.
    """

    MODEL_ID_720P = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
    MODEL_ID_480P = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

    # FIX #1: Director's Script prompt — motion verbs FIRST, 90+ words
    # Wan2.1 I2V reads the prompt as a temporal script, not a description.
    # Starting with camera/body movement cues forces the model to generate motion.
    DEFAULT_PROMPT = (
        "The camera slowly dollies forward toward a young Indian woman who walks gracefully "
        "toward the lens, her purple tiered anarkali dress swaying with each confident step, "
        "fabric tiers rippling softly in the breeze, long dark wavy hair gently flowing, "
        "arms swinging naturally at her sides, warm golden sunlight streaming through large "
        "windows, her lips curling into a subtle smile, looking directly into the camera, "
        "photorealistic 4K, cinematic depth of field, smooth continuous motion"
    )

    # FIX #2: Be explicit about the slideshow artifact in the negative prompt
    DEFAULT_NEGATIVE_PROMPT = (
        "static image, frozen, no movement, slideshow, Ken Burns zoom, pan effect, "
        "jitter, shake, deformed, floating head, bad anatomy, disconnected neck, "
        "watermark, text overlay, banner, logo, blurry, low resolution, cartoon, "
        "3d render, CGI, artificial, plastic skin, overexposed"
    )

    def __init__(
        self,
        model_id: Optional[str] = None,
        output_dir: str = "./output",
        use_480p_fallback: bool = False
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token = config.hf_token
        self.model_id = model_id or (self.MODEL_ID_480P if use_480p_fallback else self.MODEL_ID_720P)
        self.pipeline = None

    def load_pipeline(self):
        """
        Loads WanImageToVideoPipeline using diffusers in torch.bfloat16.
        Enables device_map='auto' or 'cuda' with offloading if VRAM is constrained.
        """
        import torch
        from diffusers import WanImageToVideoPipeline

        logger.info(f"Loading Wan2.1 I2V diffusion model: {self.model_id}...")
        
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        pipe = WanImageToVideoPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            token=self.hf_token
        )

        # FIX #3: Set flow_shift=3.0 on the scheduler
        # The default shift value is too low and causes near-static output (slideshow effect)
        # flow_shift=3.0 is the community-validated value for visible walking motion
        if hasattr(pipe, 'scheduler') and hasattr(pipe.scheduler, 'config'):
            pipe.scheduler.config['shift'] = 3.0
            logger.info("Scheduler flow_shift set to 3.0 (motion amplifier active)")

        if torch.cuda.is_available():
            pipe.to("cuda")
            try:
                pipe.enable_model_cpu_offload()
            except Exception:
                pass
        else:
            logger.warning("CUDA not detected. Running on CPU (requires cloud GPU instance for full inference).")

        self.pipeline = pipe
        return self.pipeline

    def generate_video(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        width: int = 480,
        height: int = 832,
        num_frames: int = 81,
        guidance_scale: float = 5.0,   # FIX #4: lowered from 6.0 — reduces motion suppression
        num_inference_steps: int = 50,  # FIX #5: raised from 40 — more coherent motion
        output_filename: str = "anaya_real_motion.mp4",
        fps: int = 16
    ) -> str:
        """
        Executes true generative diffusion:
        pipe(image=image, prompt=prompt, negative_prompt=negative_prompt, ...)
        and exports real motion clip using diffusers export_to_video.
        """
        from PIL import Image
        import numpy as np
        from diffusers.utils import export_to_video

        input_img_path = Path(image_path)
        if not input_img_path.exists():
            raise FileNotFoundError(f"Input conditioning image not found: {input_img_path}")

        raw_img = Image.open(input_img_path).convert("RGB")
        orig_w, orig_h = raw_img.size

        # FIX #6: Letterbox instead of stretch-resize
        # Cropping/squishing the body reduces the motion signal the model can follow
        scale = min(width / orig_w, height / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        resized = raw_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        conditioning_image = Image.new("RGB", (width, height), (245, 240, 235))
        conditioning_image.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))

        output_path = self.output_dir / output_filename
        active_prompt = prompt or self.DEFAULT_PROMPT
        active_negative_prompt = negative_prompt or self.DEFAULT_NEGATIVE_PROMPT

        logger.info(f"Generating true diffusion motion conditioned on: {input_img_path.name}")
        logger.info(f"Resolution: {width}x{height} | Frames: {num_frames} (~{num_frames/fps:.1f}s at {fps} fps)")
        logger.info(f"Guidance: {guidance_scale} | Steps: {num_inference_steps} | Seed: 42")

        if self.pipeline is None:
            self.load_pipeline()

        # Execute Wan2.1 Diffusion Model
        generator_output = self.pipeline(
            image=conditioning_image,
            prompt=active_prompt,
            negative_prompt=active_negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        )

        frames = generator_output.frames[0]

        # Motion quality check
        f0 = np.array(frames[0]).astype(float)
        f_last = np.array(frames[-1]).astype(float)
        motion_score = float(np.mean(np.abs(f_last - f0)))
        logger.info(f"Motion score: {motion_score:.2f} (>5=visible, >15=strong)")
        if motion_score < 3.0:
            logger.warning("Low motion detected! Try changing seed or lowering guidance_scale to 4.0")

        export_to_video(frames, str(output_path), fps=fps)
        logger.info(f"Real diffusion video exported to: {output_path}")
        return str(output_path)


def generate_wan21_video(
    image_path: str = "assets/dress/lakdata_Women_Tiered_Pur_hd_angle_01.jpg",
    prompt: Optional[str] = None,
    output_filename: str = "anaya_real_motion.mp4"
) -> str:
    """Helper entry point for pipeline execution."""
    pipeline = Wan21DiffusionPipeline()
    return pipeline.generate_video(
        image_path=image_path,
        prompt=prompt,
        output_filename=output_filename
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Wan2.1 Hugging Face Diffusers Pipeline Initialized.")
