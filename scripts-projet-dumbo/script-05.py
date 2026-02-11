#!/usr/bin/env python3
"""
script-05.py - Creature Isolation on Pure Black Background
Pipeline: traiter/ -> fond-traiter/

Architecture (7 stages):
  1. Background model     - temporal median per scene
  2. Hysteresis threshold  - preserves faint translucent edges connected to body
  3. Particle filtering    - connected components, remove small blobs
  4. Temporal smoothing    - optional multi-frame mask voting
  5. Morphological close   - fills gaps (NEVER open, which destroys tentacles)
  6. Inward feathering     - distance-transform based, zero halo
  7. Lossless compositing  - np.where: creature pixels 100% untouched

Key difference from previous version:
  - OLD: frame * soft_mask  -> degrades ALL creature pixels
  - NEW: np.where(mask, original, black) -> creature pixels EXACT originals
"""

import argparse
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def list_scene_dirs(folder: Path):
    return sorted([d for d in folder.iterdir() if d.is_dir()])


def sorted_frames(scene: Path, pattern: str):
    return sorted(scene.glob(pattern))


def read_bgr(fp: Path):
    return cv2.imread(str(fp), cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Stage 1 - Background Model (temporal median)
# ---------------------------------------------------------------------------

def compute_scene_background(frame_paths, detect_width=960, max_samples=120):
    """
    Temporal median background from evenly-sampled frames.
    Computed at detect_width resolution for speed.
    Returns (bg_small, scale, (H_full, W_full)) or (None, None, None).
    """
    n = len(frame_paths)
    if n == 0:
        return None, None, None

    # Subsample if many frames
    if n > max_samples:
        step = n / max_samples
        indices = [int(i * step) for i in range(max_samples)]
    else:
        indices = list(range(n))

    imgs = []
    H0 = W0 = None
    scale = 1.0

    for i in indices:
        img = read_bgr(frame_paths[i])
        if img is None:
            continue
        if H0 is None:
            H0, W0 = img.shape[:2]
            if W0 > detect_width:
                scale = detect_width / W0

        if scale != 1.0:
            h_s = int(H0 * scale)
            small = cv2.resize(img, (detect_width, h_s),
                               interpolation=cv2.INTER_AREA)
            imgs.append(small)
        else:
            imgs.append(img)

    if not imgs:
        return None, None, None

    stack = np.stack(imgs, axis=0)                       # [T, H, W, 3]
    bg_small = np.median(stack, axis=0).astype(np.uint8)
    return bg_small, scale, (H0, W0)


# ---------------------------------------------------------------------------
# Stage 2 - Hysteresis Thresholding
# ---------------------------------------------------------------------------

def hysteresis_threshold(diff_gray, t_high=25, t_low=8):
    """
    Double threshold inspired by Canny's hysteresis.
    Weak pixels (t_low <= val < t_high) are kept ONLY if they belong to
    a connected component that also contains at least one strong pixel
    (val >= t_high).  This preserves faint translucent edges (tentacles,
    membranes) while rejecting isolated dim noise.
    """
    strong = (diff_gray >= t_high).astype(np.uint8) * 255
    weak   = (diff_gray >= t_low).astype(np.uint8) * 255

    num_labels, labels = cv2.connectedComponents(weak, connectivity=8)

    result = np.zeros_like(weak)
    for lbl in range(1, num_labels):
        component_mask = (labels == lbl)
        if np.any(strong[component_mask] > 0):
            result[component_mask] = 255

    return result


# ---------------------------------------------------------------------------
# Stage 3 - Connected-Component Particle Filtering
# ---------------------------------------------------------------------------

def filter_particles(mask, min_area=3000, max_components=5):
    """
    Remove small blobs (marine snow, particles, sensor noise).
    Keep only the N largest blobs above min_area.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)

    if num_labels <= 1:
        return mask

    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    areas.sort(key=lambda x: x[1], reverse=True)

    result = np.zeros_like(mask)
    kept = 0
    for label_id, area in areas:
        if area >= min_area and kept < max_components:
            result[labels == label_id] = 255
            kept += 1

    # Safety: if everything was too small, keep the biggest blob
    if kept == 0 and areas:
        result[labels == areas[0][0]] = 255

    return result


# ---------------------------------------------------------------------------
# Stage 5 - Morphological Cleanup
# ---------------------------------------------------------------------------

def morphological_cleanup(mask, close_ksize=7):
    """
    MORPH_CLOSE only: fills small internal gaps without eroding thin
    biological structures (tentacles, filaments, translucent membranes).
    NEVER use MORPH_OPEN here - it destroys fine detail.
    """
    if close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (close_ksize, close_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask


def fill_holes(mask):
    """Fill enclosed holes inside the creature using flood-fill from edges."""
    h, w = mask.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    filled = mask.copy()
    cv2.floodFill(filled, flood_mask, (0, 0), 255)
    filled_inv = cv2.bitwise_not(filled)
    return cv2.bitwise_or(mask, filled_inv)


# ---------------------------------------------------------------------------
# Stage 6 - Inward-Only Feathering (distance-transform based)
# ---------------------------------------------------------------------------

def feather_inward(mask, feather_px=2):
    """
    Create soft alpha mask with feathering ONLY on the inside edge.

    Uses cv2.distanceTransform:
      - pixels >= feather_px inside the boundary -> 1.0 (full creature, untouched)
      - pixels 0..feather_px from boundary       -> smooth 0..1 transition
      - pixels outside the mask                  -> 0.0 (pure black)

    Result: zero grey halo, 1-2px anti-aliased edge, interior is EXACT 1.0.
    """
    if feather_px <= 0:
        return mask.astype(np.float32) / 255.0

    # Distance from each foreground pixel to nearest background pixel
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # Linear ramp: 0 at boundary -> 1.0 at feather_px inside
    soft = np.clip(dist / float(feather_px), 0.0, 1.0)

    # Smoothstep for a more natural transition
    soft = soft * soft * (3.0 - 2.0 * soft)

    return soft


def feather_inward_guided(mask, frame_gray, radius=4, eps=1e-4,
                          feather_px=2):
    """
    Edge-aware feathering using guided filter (opencv-contrib).
    Falls back to distance-transform method if ximgproc unavailable.
    """
    try:
        soft = cv2.ximgproc.guidedFilter(
            guide=frame_gray,
            src=mask.astype(np.float32) / 255.0,
            radius=radius,
            eps=eps
        )
        # Kill outward expansion (halo)
        soft[mask == 0] = 0.0
        # Ensure deep interior is exactly 1.0
        interior = cv2.erode(mask,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                             iterations=1)
        soft[interior == 255] = 1.0
        return np.clip(soft, 0.0, 1.0)
    except AttributeError:
        # opencv-contrib not installed
        return feather_inward(mask, feather_px=feather_px)


# ---------------------------------------------------------------------------
# Stage 7 - Lossless Compositing
# ---------------------------------------------------------------------------

def composite_on_black(frame, mask_float):
    """
    Creature pixels = ORIGINAL values.  Background = pure black.

    Where mask_float == 1.0 -> output pixel = frame pixel  (exact, no math)
    Where mask_float == 0.0 -> output pixel = (0, 0, 0)    (pure black)
    Only the 1-2px feather ring (0 < mask < 1) gets any blending,
    and that ring is entirely INSIDE the creature boundary.
    """
    # Binary zone (vast majority of pixels) - zero-degradation path
    binary = (mask_float >= 1.0)
    black  = (mask_float <= 0.0)

    out = np.zeros_like(frame)
    out[binary] = frame[binary]                              # exact copy

    # Feather zone (tiny 1-2px ring at mask edge)
    feather_zone = ~binary & ~black
    if np.any(feather_zone):
        alpha = mask_float[feather_zone, np.newaxis].astype(np.float32)
        out[feather_zone] = (frame[feather_zone].astype(np.float32)
                             * alpha).astype(np.uint8)

    return out


# ---------------------------------------------------------------------------
# Scene Processing
# ---------------------------------------------------------------------------

def process_scene(scene_path, out_scene_path, args, show_progress=True):
    """Process all frames in a single scene directory."""
    frames_all = sorted_frames(scene_path, args.pattern)
    if not frames_all:
        return 0

    ensure_dir(out_scene_path)

    # ---- Stage 1: Build background model ----
    bg_small, scale, full_size = compute_scene_background(
        frames_all,
        detect_width=args.detect_width,
        max_samples=args.bg_samples
    )
    if bg_small is None:
        return 0

    H0, W0 = full_size
    h_det, w_det = bg_small.shape[:2]

    # Smooth background for stable diff
    bg_small_blur = cv2.GaussianBlur(bg_small, (0, 0), 1.0)

    step = max(1, args.step)
    indices = list(range(0, len(frames_all), step))
    processed = 0

    # Temporal buffer
    prev_masks = deque(maxlen=args.temporal_window)

    iterator = tqdm(indices, desc=scene_path.name, leave=False) \
               if show_progress else indices

    for idx in iterator:
        fp = frames_all[idx]
        out_fp = out_scene_path / fp.name

        if out_fp.exists() and not args.overwrite:
            continue

        frame = read_bgr(fp)
        if frame is None:
            continue

        # Downscale for mask computation
        if scale != 1.0:
            frame_small = cv2.resize(frame, (w_det, h_det),
                                     interpolation=cv2.INTER_AREA)
        else:
            frame_small = frame

        # ---- Stage 2: Hysteresis thresholding ----
        diff = cv2.absdiff(frame_small, bg_small_blur)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff_gray = cv2.GaussianBlur(diff_gray, (0, 0), 1.0)

        mask = hysteresis_threshold(diff_gray,
                                    t_high=args.t_high,
                                    t_low=args.t_low)

        # ---- Stage 3: Particle removal ----
        mask = filter_particles(mask,
                                min_area=args.min_area,
                                max_components=args.max_blobs)

        # ---- Stage 4: Temporal smoothing (optional) ----
        if args.temporal_window > 1:
            prev_masks.append(mask.copy())
            if len(prev_masks) >= 2:
                stack = np.stack(list(prev_masks), axis=0).astype(
                    np.float32) / 255.0
                vote = np.mean(stack, axis=0)
                mask = (vote >= args.temporal_thresh).astype(
                    np.uint8) * 255

        # ---- Stage 5: Morphological cleanup ----
        mask = morphological_cleanup(mask, close_ksize=args.close_ksize)
        mask = fill_holes(mask)

        # Safety dilation: ensure mask fully covers creature
        if args.dilate_px > 0:
            ks = args.dilate_px * 2 + 1
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
            mask = cv2.dilate(mask, k, iterations=1)

        # ---- Upscale mask to full resolution ----
        if scale != 1.0:
            mask_full = cv2.resize(mask, (W0, H0),
                                   interpolation=cv2.INTER_NEAREST)
        else:
            mask_full = mask

        # ---- Stage 6: Feathering ----
        if args.use_guided_filter:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask_float = feather_inward_guided(
                mask_full, frame_gray,
                radius=args.guided_radius,
                eps=args.guided_eps,
                feather_px=args.feather_px)
        else:
            mask_float = feather_inward(mask_full,
                                        feather_px=args.feather_px)

        # ---- Stage 7: Lossless compositing ----
        out = composite_on_black(frame, mask_float)

        cv2.imwrite(str(out_fp), out)
        processed += 1

        # ---- Debug output (first processed frame only) ----
        if args.debug and processed == 1:
            # Diff at detect resolution
            cv2.imwrite(str(out_scene_path / "_dbg_01_diff.png"),
                        diff_gray)
            # Mask after hysteresis
            cv2.imwrite(str(out_scene_path / "_dbg_02_hysteresis.png"),
                        mask)
            # Full-res mask
            cv2.imwrite(str(out_scene_path / "_dbg_03_mask_full.png"),
                        mask_full)
            # Feather visualization
            soft_vis = (mask_float * 255).astype(np.uint8)
            cv2.imwrite(str(out_scene_path / "_dbg_04_feather.png"),
                        soft_vis)

    return processed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Creature isolation on pure black background "
                    "(zero-degradation pipeline)")

    p.add_argument("--input", required=True,
                   help="Path to <video_root> or <video_root>/traiter")
    p.add_argument("--pattern", default="frame_*.png")

    # --- Speed / pipeline ---
    g = p.add_argument_group("Speed")
    g.add_argument("--step", type=int, default=1,
                   help="Process 1 frame every N (1=all, 2=fast x2)")
    g.add_argument("--workers", type=int,
                   default=max(1, (os.cpu_count() or 4) // 2),
                   help="Parallel scenes (default: half CPU cores)")
    g.add_argument("--detect_width", type=int, default=960,
                   help="Downscale width for mask computation")
    g.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output files")

    # --- Background model ---
    g = p.add_argument_group("Background model")
    g.add_argument("--bg_samples", type=int, default=120,
                   help="Max frames for temporal median computation")

    # --- Hysteresis thresholds ---
    g = p.add_argument_group("Hysteresis thresholds")
    g.add_argument("--t_high", type=int, default=25,
                   help="Strong threshold (definitely creature)")
    g.add_argument("--t_low", type=int, default=8,
                   help="Weak threshold (preserved if connected to "
                        "strong region)")

    # --- Particle filtering ---
    g = p.add_argument_group("Particle filtering")
    g.add_argument("--min_area", type=int, default=3000,
                   help="Min blob area at detect_width (smaller = particle)")
    g.add_argument("--max_blobs", type=int, default=5,
                   help="Max creature blobs to keep per frame")

    # --- Temporal smoothing ---
    g = p.add_argument_group("Temporal smoothing")
    g.add_argument("--temporal_window", type=int, default=1,
                   help="Frames for mask voting (1=off, 3-5=smooth)")
    g.add_argument("--temporal_thresh", type=float, default=0.5,
                   help="Vote fraction to keep a pixel (0.5=majority)")

    # --- Morphology ---
    g = p.add_argument_group("Morphology")
    g.add_argument("--close_ksize", type=int, default=7,
                   help="Closing kernel (fills gaps in creature mask)")
    g.add_argument("--dilate_px", type=int, default=3,
                   help="Safety dilation in pixels (margin around "
                        "creature)")

    # --- Feathering ---
    g = p.add_argument_group("Feathering")
    g.add_argument("--feather_px", type=int, default=3,
                   help="Inward feather width in pixels "
                        "(0=hard edge, 2-4=smooth)")
    g.add_argument("--use_guided_filter", action="store_true",
                   help="Edge-aware feathering (needs opencv-contrib)")
    g.add_argument("--guided_radius", type=int, default=4)
    g.add_argument("--guided_eps", type=float, default=1e-4)

    # --- Debug ---
    p.add_argument("--debug", action="store_true",
                   help="Save intermediate masks for first frame per scene")

    args = p.parse_args()

    # Resolve paths
    in_path = Path(args.input)
    traiter = (in_path if in_path.name.lower() == "traiter"
               else in_path / "traiter")
    if not traiter.exists():
        raise SystemExit(f"traiter folder not found: {traiter}")

    root = traiter.parent
    out_base = root / "fond-traiter"
    ensure_dir(out_base)

    scene_dirs = list_scene_dirs(traiter)

    print("=" * 64)
    print("  SCRIPT-05 | Creature Isolation (Zero-Degradation Pipeline)")
    print("=" * 64)
    print(f"  Input:         {traiter}")
    print(f"  Output:        {out_base}")
    print(f"  Scenes:        {len(scene_dirs)}")
    print(f"  Step:          {args.step}"
          f" ({'all frames' if args.step == 1 else f'fast x{args.step}'})")
    print(f"  Workers:       {args.workers}")
    print(f"  Detect width:  {args.detect_width}px")
    print(f"  Thresholds:    T_high={args.t_high}  T_low={args.t_low}")
    print(f"  Min blob area: {args.min_area}px")
    print(f"  Temporal:      window={args.temporal_window}")
    print(f"  Feather:       {args.feather_px}px inward"
          f"{'  (guided filter)' if args.use_guided_filter else ''}")
    print(f"  Dilate:        {args.dilate_px}px safety margin")
    print("=" * 64)

    if args.workers > 1 and len(scene_dirs) > 1:
        # ---- Parallel scene processing ----
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for scene in scene_dirs:
                out_scene = out_base / scene.name
                f = executor.submit(process_scene, scene, out_scene,
                                    args, show_progress=False)
                futures[f] = scene.name

            for f in tqdm(as_completed(futures), total=len(futures),
                          desc="Scenes"):
                name = futures[f]
                try:
                    count = f.result()
                    tqdm.write(f"  {name}: {count} frames")
                except Exception as e:
                    tqdm.write(f"  {name}: ERROR - {e}")
    else:
        # ---- Sequential with per-frame progress ----
        for scene in tqdm(scene_dirs, desc="Scenes"):
            out_scene = out_base / scene.name
            count = process_scene(scene, out_scene, args,
                                  show_progress=True)
            tqdm.write(f"  {scene.name}: {count} frames")

    print("\nDONE")


if __name__ == "__main__":
    main()
