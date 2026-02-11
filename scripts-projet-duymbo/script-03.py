import argparse
from pathlib import Path
import cv2
from tqdm import tqdm

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trier", required=True, help="Path to trier folder")
    args = p.parse_args()

    trier_path = Path(args.trier)
    if not trier_path.exists():
        raise SystemExit("Folder not found")

    scenes = sorted([d for d in trier_path.iterdir() if d.is_dir()])

    print("Scenes found:", len(scenes))

    for scene in tqdm(scenes, desc="Generating previews"):
        frames = sorted(scene.glob("frame_*.png"))
        if not frames:
            continue

        first_frame = frames[0]
        img = cv2.imread(str(first_frame))
        if img is None:
            continue

        preview_path = scene / "_preview.jpg"
        cv2.imwrite(str(preview_path), img)

    print("Done.")

if __name__ == "__main__":
    main()
