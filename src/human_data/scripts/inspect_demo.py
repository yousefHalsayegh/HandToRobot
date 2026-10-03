import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Inspect a raw human demonstration before retargeting.")
    p.add_argument("video", type=Path)
    args = p.parse_args()
    if not args.video.exists():
        raise SystemExit(f"missing video: {args.video}")
    print(f"demo: {args.video}")
    print("Next: connect hand_tracking.extract_hand_trajectory() and object tracking.")

if __name__ == "__main__":
    main()
