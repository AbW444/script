import argparse
from pathlib import Path
import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# ----------------------------
# LOGO DETECTION (bottom-right)
# ----------------------------

def detect_logo_mask(img):
    h, w = img.shape[:2]

    y1 = int(h * 0.6)
    x1 = int(w * 0.6)

    roi = img[y1:h, x1:w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    _, bright = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray, 80, 150)

    mask_small = cv2.bitwise_or(bright, edges)

    kernel = np.ones((5, 5), np.uint8)
    mask_small = cv2.dilate(mask_small, kernel, iterations=2)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:h, x1:w] = mask_small

    return mask


def remove_logo(img, mask):
    # INPAINT_TELEA est OK ici
    return cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)


# ----------------------------
# VIGNETTE (ancienne version)
# ----------------------------

def apply_vignette(img, size=0.05, strength=0.85):
    h, w = img.shape[:2]
    fade = int(min(h, w) * size)
    fade = max(fade, 1)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    dist_left   = xx
    dist_right  = (w - 1) - xx
    dist_top    = yy
    dist_bottom = (h - 1) - yy

    dist = np.minimum(np.minimum(dist_left, dist_right),
                      np.minimum(dist_top, dist_bottom))

    t = np.clip(dist / fade, 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)  # smoothstep

    mask = (1.0 - strength) + strength * t
    out = (img.astype(np.float32) * mask[:, :, None]).astype(np.uint8)
    return out


def process_one(fp_in: Path, fp_out: Path, vignette: bool, vsize: float, vstrength: float):
    img = cv2.imread(str(fp_in))
    if img is None:
        return False

    mask = detect_logo_mask(img)
    out = remove_logo(img, mask)

    if vignette:
        out = apply_vignette(out, size=vsize, strength=vstrength)

    fp_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(fp_out), out)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root folder that contains trier/")
    parser.add_argument("--vignette", action="store_true", help="apply old subtle vignette")
    parser.add_argument("--vignette_size", type=float, default=0.05)
    parser.add_argument("--vignette_strength", type=float, default=0.85)

    # Speed / pipeline options
    parser.add_argument("--step", type=int, default=1, help="process every Nth frame (1=all). For test: 2")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 2), help="thread workers")
    parser.add_argument("--pattern", default="*.png", help="frame glob pattern inside each scene folder")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing outputs")

    args = parser.parse_args()

    root = Path(args.root)
    trier = root / "trier"
    if not trier.exists():
        raise RuntimeError("Folder 'trier' not found inside root.")

    out_base = root / "traiter"
    out_base.mkdir(exist_ok=True)

    scenes = sorted([d for d in trier.iterdir() if d.is_dir()])

    print("Output base:", out_base)
    print("Scenes found:", len(scenes))
    print(f"workers={args.workers} step={args.step} pattern={args.pattern}")

    for scene in tqdm(scenes, desc="Scenes"):
        # Si tu supprimes un dossier pendant l'exécution : on skip proprement
        if not scene.exists():
            continue

        out_scene = out_base / scene.name
        out_scene.mkdir(exist_ok=True)

        frames = sorted(scene.glob(args.pattern))
        if args.step > 1:
            frames = frames[::args.step]

        tasks = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for fp in frames:
                if not fp.exists():
                    continue
                fp_out = out_scene / fp.name
                if fp_out.exists() and not args.overwrite:
                    continue
                tasks.append(ex.submit(
                    process_one, fp, fp_out,
                    args.vignette, args.vignette_size, args.vignette_strength
                ))

            for _ in as_completed(tasks):
                pass

    print("DONE")


if __name__ == "__main__":
    main()
