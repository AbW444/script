import cv2
import numpy as np
from pathlib import Path
import argparse
import re

def create_alpha(img_bgr, v_thresh, s_thresh, open_ksize, blur_sigma):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    _, mask_v = cv2.threshold(v, v_thresh, 255, cv2.THRESH_BINARY)
    _, mask_s = cv2.threshold(s, s_thresh, 255, cv2.THRESH_BINARY)
    alpha = cv2.bitwise_and(mask_v, mask_s)

    if open_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, k)

    if blur_sigma > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), blur_sigma)

    return alpha

def bbox_from_alpha(alpha, bin_thresh=20):
    a = (alpha > bin_thresh).astype(np.uint8)
    area = int(a.sum())
    if area <= 0:
        return None, 0
    ys, xs = np.where(a > 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), area

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", required=True)

    # Détection présence
    p.add_argument("--min_area", type=int, default=4500, help="aire minimale (alpha) pour considérer 'créature présente'")
    p.add_argument("--gap_frames", type=int, default=15, help="frames sans sujet avant fermeture clip")

    # Matte (abysses)
    p.add_argument("--v_thresh", type=int, default=25)
    p.add_argument("--s_thresh", type=int, default=15)
    p.add_argument("--open_ksize", type=int, default=7)
    p.add_argument("--blur_sigma", type=float, default=6.0)

    # Crop
    p.add_argument("--margin", type=int, default=80, help="marge autour bbox")
    p.add_argument("--min_w", type=int, default=480, help="largeur minimale crop")
    p.add_argument("--min_h", type=int, default=480, help="hauteur minimale crop")

    # Anti-jitter crop (lissage bbox)
    p.add_argument("--smooth", type=float, default=0.85, help="0..1 (plus haut = crop plus stable)")

    args = p.parse_args()

    video_path = Path(args.input)
    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    safe_name = re.sub(r'[<>:"/\\\\|?*]', '_', video_path.stem)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    clip_idx = 1
    no_subject = 0
    writer = None

    # bbox lissée
    bx = by = bw = bh = None  # store as x1,y1,x2,y2

    def open_writer(w, h):
        nonlocal writer
        out_path = out_root / f"{safe_name}_clip_{clip_idx:03d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError("Impossible d'ouvrir VideoWriter (codec mp4v).")

    def close_writer():
        nonlocal writer, bx, by, bw, bh
        if writer is not None:
            writer.release()
            writer = None
        bx = by = bw = bh = None

    def lerp(a, b, t):
        return a * t + b * (1 - t)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        alpha = create_alpha(frame, args.v_thresh, args.s_thresh, args.open_ksize, args.blur_sigma)
        bb, area = bbox_from_alpha(alpha, bin_thresh=20)

        if area >= args.min_area and bb is not None:
            no_subject = 0
            x1, y1, x2, y2 = bb

            # marge + taille mini
            x1 = x1 - args.margin
            y1 = y1 - args.margin
            x2 = x2 + args.margin
            y2 = y2 + args.margin

            # impose min size autour du centre
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = max(args.min_w, (x2 - x1))
            h = max(args.min_h, (y2 - y1))
            x1 = cx - w / 2
            x2 = cx + w / 2
            y1 = cy - h / 2
            y2 = cy + h / 2

            # clamp image
            x1 = clamp(int(round(x1)), 0, W - 2)
            y1 = clamp(int(round(y1)), 0, H - 2)
            x2 = clamp(int(round(x2)), x1 + 1, W - 1)
            y2 = clamp(int(round(y2)), y1 + 1, H - 1)

            # lissage bbox pour éviter crop qui tremble
            if bx is None:
                bx, by, bw, bh = x1, y1, x2, y2
            else:
                t = args.smooth
                bx = int(round(lerp(bx, x1, t)))
                by = int(round(lerp(by, y1, t)))
                bw = int(round(lerp(bw, x2, t)))
                bh = int(round(lerp(bh, y2, t)))

            crop = frame[by:bh, bx:bw]
            if crop.size == 0:
                continue

            if writer is None:
                open_writer(crop.shape[1], crop.shape[0])

            writer.write(crop)

        else:
            no_subject += 1
            if writer is not None and no_subject >= args.gap_frames:
                close_writer()
                clip_idx += 1
                no_subject = 0

    # fin
    close_writer()
    cap.release()
    print("OK - clips exported to:", out_root)

if __name__ == "__main__":
    main()
