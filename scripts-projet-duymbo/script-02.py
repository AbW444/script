import argparse
from pathlib import Path
import shutil
import cv2
import numpy as np
from tqdm import tqdm

def frame_signature(img_bgr, size=160):
    # Downscale -> gray
    h, w = img_bgr.shape[:2]
    scale_w = size
    scale_h = int(h * (scale_w / w))
    small = cv2.resize(img_bgr, (scale_w, max(1, scale_h)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Histogram (stable for cut detection)
    hist = cv2.calcHist([gray],[0],None,[32],[0,256]).flatten()
    hist = hist / (hist.sum() + 1e-9)

    # Edge energy (helps when exposure changes)
    edges = cv2.Canny(gray, 60, 140)
    edge_energy = float(edges.mean())  # 0..255

    return hist, edge_energy

def hist_distance(h1, h2):
    # L1 distance (fast + robust)
    return float(np.abs(h1 - h2).sum())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True, help="Folder with frame_000001.png etc")
    p.add_argument("--out", required=True, help="Output base folder for scenes")
    p.add_argument("--threshold", type=float, default=0.35, help="Cut sensitivity (lower = more cuts)")
    p.add_argument("--min_len", type=int, default=18, help="Minimum frames per scene (avoid micro cuts)")
    p.add_argument("--copy", action="store_true", help="Copy instead of move")
    p.add_argument("--pattern", default="frame_*.png")
    args = p.parse_args()

    frames_dir = Path(args.frames)
    out_base = Path(args.out)
    out_base.mkdir(parents=True, exist_ok=True)

    files = sorted(frames_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No frames found in {frames_dir}")

    # Prepare first scene
    scene_idx = 1
    scene_dir = out_base / f"scene_{scene_idx:04d}"
    scene_dir.mkdir(parents=True, exist_ok=True)

    prev_hist = None
    prev_edge = None
    scene_start_i = 0

    def new_scene(i):
        nonlocal scene_idx, scene_dir, scene_start_i
        scene_idx += 1
        scene_dir = out_base / f"scene_{scene_idx:04d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        scene_start_i = i

    def write_frame(src: Path, dst_dir: Path):
        # Keep original filename
        dst = dst_dir / src.name
        if args.copy:
            shutil.copy2(src, dst)
        else:
            shutil.move(src, dst)

    # Prime with first frame
    first = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise SystemExit(f"Cannot read {files[0]}")
    prev_hist, prev_edge = frame_signature(first)
    write_frame(files[0], scene_dir)

    # Iterate
    for i in tqdm(range(1, len(files)), desc="Splitting"):
        fp = files[i]
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            # If unreadable, skip (or you could move to a 'bad' folder)
            continue

        h, e = frame_signature(img)
        d = hist_distance(prev_hist, h)
        ed = abs(prev_edge - e) / 255.0  # normalize

        score = d + 0.35 * ed  # mix

        # Cut decision with min length guard
        if score >= args.threshold and (i - scene_start_i) >= args.min_len:
            new_scene(i)

        write_frame(fp, scene_dir)

        prev_hist, prev_edge = h, e

    print(f"OK - scenes: {scene_idx}")
    print(f"Output: {out_base}")

if __name__ == "__main__":
    main()
