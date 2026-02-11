import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def list_scene_dirs(traiter: Path):
    return sorted([d for d in traiter.iterdir() if d.is_dir()])

def sorted_frames(scene: Path, pattern: str):
    return sorted(scene.glob(pattern))

def read_bgr(fp: Path):
    return cv2.imread(str(fp), cv2.IMREAD_COLOR)

def temporal_median_background(frames, idx, window=25, detect_width=960):
    """
    Compute temporal median background on a window around idx.
    Returns background in full res (by upscaling median small).
    """
    n = len(frames)
    half = window // 2
    lo = clamp(idx - half, 0, n - 1)
    hi = clamp(idx + half, 0, n - 1)

    sample = frames[lo:hi+1]
    imgs = []
    scale = None
    H0 = W0 = None

    for fp in sample:
        img = read_bgr(fp)
        if img is None:
            continue
        H0, W0 = img.shape[:2]
        if W0 > detect_width:
            s = detect_width / W0
            small = cv2.resize(img, (detect_width, int(H0*s)), interpolation=cv2.INTER_AREA)
            scale = s
        else:
            small = img
            scale = 1.0
        imgs.append(small)

    if not imgs:
        return None

    stack = np.stack(imgs, axis=0)  # [T,H,W,3]
    med = np.median(stack, axis=0).astype(np.uint8)

    # upscale to full res if needed
    if scale != 1.0 and H0 is not None and W0 is not None:
        med_full = cv2.resize(med, (W0, H0), interpolation=cv2.INTER_LINEAR)
    else:
        med_full = med
    return med_full

def diff_mask(frame, bg, diff_thresh=22):
    """
    Foreground by abs diff to background.
    """
    d = cv2.absdiff(frame, bg)
    gray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    # slight blur to stabilize noise
    gray = cv2.GaussianBlur(gray, (0,0), 1.2)
    _, m = cv2.threshold(gray, diff_thresh, 255, cv2.THRESH_BINARY)
    return m

def motion_mask(prev, frame, flow_weight=1.0, flow_thresh=1.2):
    """
    Motion detection using optical flow magnitude (Farneback).
    """
    g0 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g0 = cv2.GaussianBlur(g0, (0,0), 1.2)
    g1 = cv2.GaussianBlur(g1, (0,0), 1.2)

    flow = cv2.calcOpticalFlowFarneback(
        g0, g1, None,
        pyr_scale=0.5, levels=3, winsize=21,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
    mag = mag * float(flow_weight)

    # threshold in "pixels per frame"
    m = (mag > float(flow_thresh)).astype(np.uint8) * 255
    # clean a bit
    m = cv2.medianBlur(m, 5)
    return m

def cleanup_mask(mask, open_ksize=5, close_ksize=11, dilate_ksize=9, dilate_iter=1):
    out = mask.copy()
    if open_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    if close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    if dilate_ksize > 1 and dilate_iter > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
        out = cv2.dilate(out, k, iterations=int(dilate_iter))
    return out

def keep_largest_component(mask, min_area=2500):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    # pick largest (excluding background)
    best_i = -1
    best_area = 0
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > best_area:
            best_area = area
            best_i = i
    if best_area < min_area:
        return mask  # too small → keep as-is (could be multiple small critters)
    out = np.zeros_like(mask)
    out[labels == best_i] = 255
    return out

def feather(mask, blur_sigma=18.0):
    if blur_sigma <= 0:
        return mask
    return cv2.GaussianBlur(mask, (0,0), float(blur_sigma))

def apply_black_bg(frame, soft_mask):
    a = (soft_mask.astype(np.float32)/255.0)[:,:,None]
    out = (frame.astype(np.float32) * a).astype(np.uint8)
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to <video_root> OR <video_root>/traiter")
    p.add_argument("--pattern", default="frame_*.png")

    # speed
    p.add_argument("--step", type=int, default=1, help="process 1 frame every N (1=all)")
    p.add_argument("--detect_width", type=int, default=960, help="downscale width for background calc window")

    # temporal background window
    p.add_argument("--bg_window", type=int, default=25, help="temporal window for median background")

    # thresholds
    p.add_argument("--diff_thresh", type=int, default=22, help="absdiff threshold")
    p.add_argument("--flow_thresh", type=float, default=1.2, help="optical flow magnitude threshold (px/frame)")
    p.add_argument("--flow_weight", type=float, default=1.0)

    # mask cleanup
    p.add_argument("--open_ksize", type=int, default=5)
    p.add_argument("--close_ksize", type=int, default=11)
    p.add_argument("--dilate_ksize", type=int, default=9)
    p.add_argument("--dilate_iter", type=int, default=1)
    p.add_argument("--largest", action="store_true", help="keep only largest connected component")
    p.add_argument("--min_area", type=int, default=2500)

    # feather / halo
    p.add_argument("--feather_sigma", type=float, default=18.0)

    p.add_argument("--debug", action="store_true")

    args = p.parse_args()

    in_path = Path(args.input)
    traiter = in_path if in_path.name.lower() == "traiter" else (in_path / "traiter")
    if not traiter.exists():
        raise SystemExit(f"traiter folder not found: {traiter}")

    root = traiter.parent
    out_base = root / "fond-traiter"
    ensure_dir(out_base)

    scene_dirs = list_scene_dirs(traiter)
    step = max(1, int(args.step))

    print("Input traiter:", traiter)
    print("Output base:", out_base)
    print("Scenes:", len(scene_dirs))

    for scene in tqdm(scene_dirs, desc="Scenes"):
        frames_all = sorted_frames(scene, args.pattern)
        if not frames_all:
            continue

        out_scene = out_base / scene.name
        ensure_dir(out_scene)

        # process subsampled indices
        indices = list(range(0, len(frames_all), step))

        prev_frame = None

        for j, idx in enumerate(tqdm(indices, desc=scene.name, leave=False)):
            fp = frames_all[idx]
            frame = read_bgr(fp)
            if frame is None:
                continue

            # background median (temporal)
            bg = temporal_median_background(frames_all, idx, window=args.bg_window, detect_width=args.detect_width)
            if bg is None:
                continue

            m_diff = diff_mask(frame, bg, diff_thresh=args.diff_thresh)

            if prev_frame is None:
                m_flow = np.zeros_like(m_diff)
            else:
                m_flow = motion_mask(prev_frame, frame, flow_weight=args.flow_weight, flow_thresh=args.flow_thresh)

            # fuse
            m = cv2.bitwise_or(m_diff, m_flow)

            # clean
            m = cleanup_mask(
                m,
                open_ksize=args.open_ksize,
                close_ksize=args.close_ksize,
                dilate_ksize=args.dilate_ksize,
                dilate_iter=args.dilate_iter
            )

            if args.largest:
                m = keep_largest_component(m, min_area=args.min_area)

            soft = feather(m, blur_sigma=args.feather_sigma)
            out = apply_black_bg(frame, soft)

            cv2.imwrite(str(out_scene / fp.name), out)

            if args.debug and j == 0:
                cv2.imwrite(str(out_scene / "_dbg_diff.png"), m_diff)
                cv2.imwrite(str(out_scene / "_dbg_flow.png"), m_flow)
                cv2.imwrite(str(out_scene / "_dbg_mask.png"), m)
                cv2.imwrite(str(out_scene / "_dbg_soft.png"), soft)

            prev_frame = frame

    print("DONE")

if __name__ == "__main__":
    main()
