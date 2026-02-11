import cv2
import numpy as np
from pathlib import Path
import argparse
import csv

def create_alpha(img_bgr, v_thresh, s_thresh, open_ksize, blur_sigma):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

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
    ys, xs = np.nonzero(a)
    if xs.size == 0:
        return None, 0
    area = int(xs.size)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), area

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out_csv", required=True)

    p.add_argument("--min_area", type=int, default=3500)
    p.add_argument("--gap_frames", type=int, default=15)

    p.add_argument("--v_thresh", type=int, default=22)
    p.add_argument("--s_thresh", type=int, default=12)
    p.add_argument("--open_ksize", type=int, default=5)
    p.add_argument("--blur_sigma", type=float, default=4.0)

    p.add_argument("--margin", type=int, default=80)
    p.add_argument("--min_w", type=int, default=480)
    p.add_argument("--min_h", type=int, default=480)

    p.add_argument("--step", type=int, default=2)
    p.add_argument("--detect_width", type=int, default=960)

    args = p.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    clips = []
    in_clip = False
    start_f = None
    no_subject = 0

    ux1 = uy1 = 10**9
    ux2 = uy2 = -1

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % args.step != 0:
            continue

        H0, W0 = frame.shape[:2]
        scale = args.detect_width / W0
        small = cv2.resize(frame, (args.detect_width, int(H0 * scale)))

        alpha = create_alpha(small, args.v_thresh, args.s_thresh, args.open_ksize, args.blur_sigma)
        bb, area = bbox_from_alpha(alpha)

        if bb is not None and area >= args.min_area:
            no_subject = 0
            x1, y1, x2, y2 = bb

            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)

            x1 -= args.margin
            y1 -= args.margin
            x2 += args.margin
            y2 += args.margin

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = max(args.min_w, (x2 - x1))
            h = max(args.min_h, (y2 - y1))
            x1 = int(cx - w/2)
            x2 = int(cx + w/2)
            y1 = int(cy - h/2)
            y2 = int(cy + h/2)

            x1 = clamp(x1, 0, W-2)
            y1 = clamp(y1, 0, H-2)
            x2 = clamp(x2, x1+1, W-1)
            y2 = clamp(y2, y1+1, H-1)

            if not in_clip:
                in_clip = True
                start_f = frame_idx
                ux1, uy1, ux2, uy2 = x1, y1, x2, y2
            else:
                ux1 = min(ux1, x1)
                uy1 = min(uy1, y1)
                ux2 = max(ux2, x2)
                uy2 = max(uy2, y2)

        else:
            if in_clip:
                no_subject += 1
                if no_subject >= args.gap_frames:
                    end_f = frame_idx - args.gap_frames
                    clips.append((start_f, end_f, ux1, uy1, ux2-ux1, uy2-uy1))
                    in_clip = False
                    no_subject = 0

    if in_clip:
        clips.append((start_f, frame_idx, ux1, uy1, ux2-ux1, uy2-uy1))

    cap.release()

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start_frame","end_frame","start_sec","end_sec","x","y","w","h","fps"])
        for (sf, ef, x, y, cw, ch) in clips:
            w.writerow([sf, ef, sf/fps, ef/fps, x, y, cw, ch, fps])

    print("OK -", len(clips), "clips detected")

if __name__ == "__main__":
    main()
