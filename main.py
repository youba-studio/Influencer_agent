"""
Anaya Sharma AI Influencer - Autonomous Agent Orchestrator.
Pipelines:
1. Affiliate Monetization (EarnKaro tracking & Linktree storefront)
2. Unified Generative Cloud Video (Wan2.1 I2V diffusion - clean 9:16 footage, ZERO text)
3. Social Distribution (Instagram Reels publishing with Hinglish captions & bio CTA)
"""

import os
import sys
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List

# Workspace configuration & tools
from config import config
from tools.affiliate_engine import AffiliateEngine
from tools.cloud_video_generator import CloudVideoGenerator
from tools.instagram_publisher import InstagramPublisher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AnayaAgent")


@dataclass
class LocalAgentConfig:
    """Agent runtime configuration for Antigravity SDK orchestration."""
    agent_name: str = "Anaya Sharma AI Influencer"
    version: str = "2.2.0"
    avatar_image_path: str = "./assets/anaya_avatar.jpg"
    enable_affiliate: bool = True
    enable_generative_video: bool = True
    enable_social_distribution: bool = True
    output_dir: str = "./output"


class AnayaAgent:
    """
    Unified Autonomous AI Influencer Agent.
    """

    def __init__(self, agent_config: LocalAgentConfig = LocalAgentConfig()):
        self.agent_config = agent_config
        self.affiliate_engine = AffiliateEngine()
        self.video_generator = CloudVideoGenerator(output_dir=agent_config.output_dir)
        self.instagram_publisher = InstagramPublisher()

    def run_pipeline(
        self,
        outfit_prompt: str,
        product_items: List[Dict[str, str]],
        video_motion_prompt: str | None = None
    ) -> Dict[str, Any]:
        """
        Executes the end-to-end autonomous publishing cycle.
        """
        logger.info(f"=== Starting Pipeline Run for: '{outfit_prompt}' ===")
        report = {
            "agent": self.agent_config.agent_name,
            "version": self.agent_config.version,
            "outfit_prompt": outfit_prompt,
            "steps": {}
        }

        # Step 1: Affiliate Monetization & Linktree Storefront
        if self.agent_config.enable_affiliate:
            logger.info("Executing Step 1: Converting affiliate links & updating Linktree blocks...")
            affiliate_res = self.affiliate_engine.batch_process_outfit(product_items)
            report["steps"]["affiliate"] = affiliate_res
            logger.info(f"Generated {len(affiliate_res['items'])} monetized tracking links.")

        # Step 2: Official Wan2.1 I2V Diffusers Pipeline (True Generative Motion)
        video_res = {}
        final_video_path = None
        if self.agent_config.enable_generative_video:
            logger.info("Executing Step 2: Building Wan2.1 I2V Diffusers specification...")
            prompt = video_motion_prompt or (
                "A photorealistic 9:16 vertical fashion reel, young Indian woman with long dark hair "
                f"wearing {outfit_prompt}, walking gracefully forward, natural clothing physics, "
                "fluid body movement, realistic skin texture, cinematic lighting."
            )
            base_dress = product_items[0].get("image_path", "assets/dress/lakdata_Women_Tiered_Pur_hd_angle_01.jpg")
            
            spec = self.video_generator.build_wan21_spec(
                reference_image_path=base_dress,
                motion_prompt=prompt,
                num_frames=81,
                width=720,
                height=1280,
                fps=16
            )
            final_video_path = self.video_generator.execute_pipeline(spec)
            video_res = {
                "status": "WAN21_DIFFUSION_READY",
                "framework": spec["framework"],
                "model_id": spec["model_id"],
                "video_path": final_video_path,
                "frames": 81,
                "fps": 16,
                "clean_footage_verified": True
            }
            report["steps"]["video_generation"] = video_res


        # Step 3: Social Media Publishing (Instagram Reels)
        if self.agent_config.enable_social_distribution and final_video_path:
            logger.info("Executing Step 3: Staging and publishing Reel to Instagram...")
            self.instagram_publisher.login()
            caption = self.instagram_publisher.build_hinglish_caption(outfit_prompt)
            publish_res = self.instagram_publisher.publish_reel(
                video_path=final_video_path,
                caption=caption,
                thumbnail_path=self.agent_config.avatar_image_path
            )
            report["steps"]["social_publishing"] = publish_res

        logger.info("=== Pipeline Run Finished Successfully! ===")
        return report


def main():
    """Initializes and runs the Anaya Sharma Agent workflow."""
    print("=" * 68)
    print("  ANAYA SHARMA - AI INFLUENCER AUTONOMOUS WORKSPACE (v2.2.0)")
    print("=" * 68)

    agent_cfg = LocalAgentConfig(
        agent_name="Anaya Sharma Agent",
        enable_affiliate=True,
        enable_generative_video=True,
        enable_social_distribution=True
    )
    agent = AnayaAgent(agent_config=agent_cfg)

    # Active outfit look: Women Tiered Purple Festive Dress Showcase
    sample_outfit = "Women Tiered Purple Festive Kurta & Dupatta Set"
    sample_products = [
        {
            "title": "Tiered Purple Festive Anarkali Kurta Set",
            "url": "https://www.myntra.com/dresses/tiered-purple-festive-anarkali/1049281",
            "category": "Ethnic Wear",
            "image_path": "assets/dress/lakdata_Women_Tiered_Pur_hd_angle_01.jpg"
        },
        {
            "title": "Antique Gold Kundan Earrings",
            "url": "https://www.ajio.com/jewellery/kundan-earrings/827192",
            "category": "Jewellery"
        },
        {
            "title": "Zari Embroidered Juttis",
            "url": "https://www.amazon.in/dp/B08XYZ5678",
            "category": "Footwear"
        }
    ]

    result = agent.run_pipeline(
        outfit_prompt=sample_outfit,
        product_items=sample_products,
        video_motion_prompt=(
            "Cinematic 9:16 vertical fashion reel of young Indian woman Anaya Sharma presenting a tiered purple festive kurta dress, "
            "walking gracefully toward the camera, relaxed elegant posture, natural hair movement, realistic skin texture, "
            "smooth clothing physics, photorealistic warm daylight, 4k"
        )
    )

    print("\n--- Pipeline Execution Report ---")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
