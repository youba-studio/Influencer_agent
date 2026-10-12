"""
main.py — Anaya Sharma AI Influencer Agent Orchestrator.

Pipelines (each can be run independently):
  1. Spec Builder  — prepares the cloud notebook run specification JSON.
  2. Affiliate     — generates EarnKaro links (disabled by default).
  3. Publisher     — publishes to Instagram (disabled by default; requires
                     a validated cloud-generated video first).

Video generation itself always happens in the cloud notebook.
This script does NOT run or fake local video generation.

Usage:
  python main.py --build-spec --image assets/dress/angle_01.jpg
  python main.py --build-spec --image my_outfit.jpg --skip-affiliate --skip-publish
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("AnayaAgent")

BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="Anaya Sharma AI Influencer Agent")
    p.add_argument(
        "--build-spec",
        action="store_true",
        default=True,
        help="Build the cloud notebook run spec JSON (default: True)",
    )
    p.add_argument(
        "--image",
        type=str,
        default="assets/dress/lakdata_Women_Tiered_Pur_hd_angle_01.jpg",
        help="Path to the Anaya outfit reference image",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for spec JSON and downloaded video",
    )
    p.add_argument(
        "--skip-affiliate",
        action="store_true",
        default=True,
        help="Skip affiliate link generation (default: skip)",
    )
    p.add_argument(
        "--skip-publish",
        action="store_true",
        default=True,
        help="Skip Instagram publishing (default: skip; enable only after"
             " verifying the generated video)",
    )
    return p.parse_args()


def run_build_spec(image_path: str, output_dir: str) -> dict:
    """Builds and saves the cloud notebook execution specification."""
    logger.info("Building cloud notebook run specification...")

    # Import here so other modules do not need GPU deps
    sys.path.insert(0, str(BASE_DIR))
    from tools.cloud_video_generator import build_run_spec, print_run_instructions

    try:
        spec = build_run_spec(
            reference_image_path=image_path,
            output_dir=output_dir,
        )
        print_run_instructions(spec)
        return spec
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)


def run_affiliate(spec: dict) -> dict:
    """Generates EarnKaro affiliate links for the outfit."""
    logger.info("Running affiliate link generation...")
    try:
        from tools.affiliate_engine import AffiliateEngine
        engine = AffiliateEngine()
        sample_products = [
            {
                "title": "Tiered Purple Festive Anarkali Kurta Set",
                "url": "https://www.myntra.com/dresses/tiered-purple-festive-anarkali/1049281",
                "category": "Ethnic Wear",
                "image_path": spec["conditioning"]["reference_image"],
            },
        ]
        result = engine.batch_process_outfit(sample_products)
        logger.info(f"Affiliate: {len(result.get('items', []))} links generated.")
        return result
    except Exception as e:
        logger.warning(f"Affiliate step failed (non-fatal): {e}")
        return {"error": str(e)}


def run_publish(video_path: str) -> dict:
    """Publishes the validated video to Instagram Reels."""
    if not Path(video_path).exists():
        logger.error(
            f"Video file not found: {video_path}\n"
            "  Run the Kaggle notebook first and download the output video."
        )
        return {"error": "video_not_found", "path": video_path}

    logger.info("Running Instagram publisher...")
    try:
        from tools.instagram_publisher import InstagramPublisher
        publisher = InstagramPublisher()
        publisher.login()
        caption = publisher.build_hinglish_caption("purple tiered anarkali dress")
        result = publisher.publish_reel(video_path=video_path, caption=caption)
        logger.info("Instagram publish complete.")
        return result
    except Exception as e:
        logger.warning(f"Publish step failed: {e}")
        return {"error": str(e)}


def main():
    args = parse_args()

    print("=" * 70)
    print("  ANAYA SHARMA — AI INFLUENCER AGENT  (v3.0)")
    print("=" * 70)

    report = {"steps": {}}

    # Step 1: Always build the cloud spec first
    if args.build_spec:
        spec = run_build_spec(args.image, args.output_dir)
        report["steps"]["spec"] = {
            "status": "SPEC_WRITTEN",
            "model": spec["model"]["model_id"],
            "spec_file": str(Path(args.output_dir) / "run_report.json"),
        }

    # Step 2: Optional affiliate generation
    if not args.skip_affiliate and args.build_spec:
        report["steps"]["affiliate"] = run_affiliate(spec)

    # Step 3: Optional publish (only after downloading real cloud video)
    if not args.skip_publish:
        import pipeline_config as cfg
        video_path = str(Path(args.output_dir) / cfg.OUTPUT_FILENAME)
        report["steps"]["publish"] = run_publish(video_path)

    print("\n--- Agent Report ---")
    print(json.dumps(report, indent=2))
    print()
    logger.info("Done. Next step: run the Kaggle notebook (see instructions above).")


if __name__ == "__main__":
    main()
