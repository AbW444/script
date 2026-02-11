import argparse
import shutil
import subprocess
from pathlib import Path
import re

import cv2
import numpy as np
from tqdm import tqdm


def run(cmd):
    subprocess.run(cmd, check=True)


def extract_frames(video_path: Path, frames_dir: Path, fps: int):
    frames_dir.mkdir(parents=True, exist_ok=True)
    # Frames PNG (BGR) pour traitement
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        str(frames_dir / "frame_%06d.png"),
    ]
    run(cmd)


def create_alpha_abyss_translucent(
    img_bgr: np.ndarray,
    v_thresh: int,
    s_thresh: int,
    open_ksize: int,
    dilate_ksize: int,
    blur_sigma: float,
    gamma: float,
):
    """
    Matte doux pour sujets abyssaux translucides sur fond sombre.
    Combine Value (lumière) + Saturation, nettoie les particules, feather + gamma.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    _, mask_v = cv2.threshold(v, v_thresh, 255, cv2.THRESH_BINARY)
    _, mask_s = cv2.threshold(s, s_thresh, 255, cv2.THRESH_BINARY)

    alpha = cv2.bitwise_and(mask_v, mask_s)

    # Nettoyage particules (OPEN)
    if open_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, k)

    # Dilatation douce
    if dilate_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
        alpha = cv2.dilate(alpha, k, iterations=1)

    # Feather
    if blur_sigma > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), blur_sigma)

    # Gamma (densifie ou allège l'alpha)
    if gamma != 1.0:
        a = alpha.astype(np.float32) / 255.0
        a = np.power(a, gamma)
        alpha = np.clip(a * 255.0, 0, 255).astype(np.uint8)

    return alpha


def process_frames(
    frames_dir: Path,
    out_png_dir: Path,
    v_thresh: int,
    s_thresh: int,
    open_ksize: int,
    dilate_ksize: int,
    blur_sigma: float,
    gamma: float,
):
    out_png_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        raise RuntimeError(f"Aucune frame trouvée dans {frames_dir}")

    for fp in tqdm(frames, desc="Matting"):
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            continue

        alpha = create_alpha_abyss_translucent(
            img, v_thresh, s_thresh, open_ksize, dilate_ksize, blur_sigma, gamma
        )

        # BGR -> BGRA
        b, g, r = cv2.split(img)
        bgra = cv2.merge((b, g, r, alpha))

        cv2.imwrite(str(out_png_dir / fp.name), bgra)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Chemin vidéo source")
    p.add_argument("--outdir", required=True, help="Dossier output (traiter)")
    p.add_argument("--fps", type=int, default=25)

    # Paramètres matte (ajustables)
    p.add_argument("--v_thresh", type=int, default=25, help="Seuil luminance (Value)")
    p.add_argument("--s_thresh", type=int, default=15, help="Seuil saturation")
    p.add_argument("--open_ksize", type=int, default=7, help="Nettoyage bruit (odd)")
    p.add_argument("--dilate_ksize", type=int, default=9, help="Gonfle silhouette (odd)")
    p.add_argument("--blur_sigma", type=float, default=18.0, help="Feather (sigma)")
    p.add_argument("--gamma", type=float, default=0.75, help="Densité alpha (<1 plus dense)")

    # Optionnel: supprimer frames sources après
    p.add_argument("--cleanup", action="store_true", help="Supprime le dossier _frames")

    args = p.parse_args()

    video_path = Path(args.input)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Nom safe dossier
    raw_name = video_path.stem
    name = re.sub(r'[<>:"/\\|?*]', '_', raw_name)
    work_dir = out_root / name
    frames_dir = work_dir / "_frames"
    out_png_dir = work_dir / "png_alpha"

    # Rebuild propre
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Video: {video_path}")
    print(f"Output: {work_dir}")

    extract_frames(video_path, frames_dir, args.fps)
    process_frames(
        frames_dir,
        out_png_dir,
        args.v_thresh,
        args.s_thresh,
        args.open_ksize,
        args.dilate_ksize,
        args.blur_sigma,
        args.gamma,
    )

    if args.cleanup:
        shutil.rmtree(frames_dir, ignore_errors=True)

    print("\nOK ✅")
    print(f"PNG alpha sequence: {out_png_dir}")


if __name__ == "__main__":
    main()
