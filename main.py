import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = ROOT_DIR / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import importlib

trend_hunter = importlib.import_module("3_trend_hunter")
story_planner = importlib.import_module("1_plan_story")
video_renderer = importlib.import_module("2_render_video")
drive_uploader = importlib.import_module("4_drive_uploader")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated Faceless YouTube Shorts Factory Orchestrator"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Custom topic to override automatic trend hunting.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip Google Drive & Sheets upload (for local testing).",
    )
    args = parser.parse_args()

    try:
        # Step 1: Trend Hunting
        if args.topic:
            topic = args.topic.strip()
            print(f"[Main] Using custom topic: {topic}")
        else:
            print("[Main] Step 1/4: Hunting trending US Tech & AI topic...")
            topic = trend_hunter.hunt_trending_topic()

        # Step 2: Story Planning
        print(f"[Main] Step 2/4: Generating story plan for '{topic}'...")
        plan = story_planner.generate_story_plan(topic)
        plan_path = ROOT_DIR / "output" / "story_plan.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(
            json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[Main] Saved story plan ({len(plan['episodes'])} part(s)) -> {plan_path}")

        # Step 3: Video Rendering
        print("[Main] Step 3/4: Rendering 1080x1920 60fps video(s)...")
        video_renderer.main()

        # Step 4: Drive & Sheets Upload
        if args.skip_upload:
            print("[Main] Step 4/4: Skipping Google Drive upload (--skip-upload).")
        else:
            print("[Main] Step 4/4: Uploading to Google Drive & logging to Sheets...")
            drive_uploader.upload_story_outputs()

        print("\n[Main] Pipeline completed successfully!")
    except Exception as exc:
        print(f"\n[Main] Pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
