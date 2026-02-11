import argparse
from pathlib import Path
import cv2
import re
import unicodedata
from tqdm import tqdm

def sanitize_name(name):
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r'\s+', "_", name)
    return name

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--base_out", required=True)
    p.add_argument("--fps", type=float, default=0)
    args = p.parse_args()

    video_path = Path(args.input)
    if not video_path.exists():
        raise SystemExit("Video not found")

    clean_name = sanitize_name(video_path.stem)

    root_out = Path(args.base_out) / clean_name
    frames_dir = root_out / "_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    target_fps = original_fps if args.fps == 0 else args.fps

    frame_interval = int(round(original_fps / target_fps)) if target_fps < original_fps else 1
    frame_interval = max(1, frame_interval)

    saved_idx = 0

    print("Video:", video_path.name)
    print("Total frames:", total_frames)
    print("Output:", frames_dir)
    print("Extracting...")

    with tqdm(total=total_frames) as pbar:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                out_path = frames_dir / f"frame_{saved_idx:06d}.png"
                cv2.imwrite(str(out_path), frame)
                saved_idx += 1

            frame_idx += 1
            pbar.update(1)

    cap.release()

    print("Done.")
    print("Frames saved:", saved_idx)

if __name__ == "__main__":
    main()
