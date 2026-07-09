"""
Overlay scheletri GREZZI (in pixel) sui frame reali del video
=============================================================
Verifica che i keypoint HRNet combacino col pedone nel video.

A DIFFERENZA di viz_skeleton_norm.py (che disegna le pose NORMALIZZATE,
in uno spazio astratto non sovrapponibile al video), qui si usano i
keypoint GREZZI in PIXEL, presi direttamente dal pkl, e si disegnano sui
frame veri estratti dal .mp4. Serve a controllare l'allineamento
posa<->pedone, non la normalizzazione.

Sorgente frame: i frame-number sono ASSOLUTI (indici di frame del video).
Vengono estratti dal .mp4 con OpenCV per posizione (CAP_PROP_POS_FRAMES).

Struttura video attesa (come indicato):
    <video_root>/<set>/<video_id>.mp4     es. .../pie_video/set05/video_0001.mp4

Uso:
    python viz_skeleton_on_video.py \
        --pose_dir data/poses --set set05 --video video_0001 \
        --pid 5_1_1740 --video_root ~/Desktop/pie_video \
        --obs_len 16 --start 0 --out out_viz_video

Se manca il .mp4, con --no_video disegna su sfondo neutro (solo per
controllare che il disegno in pixel sia coerente col bbox).
"""

import argparse
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pose_cache import PoseCache
from pose_preproc import derive_19_joints
from skeleton import JOINT_EDGES, NECK, CHIP, JOINT_NAMES

IMG_W, IMG_H = 1920, 1080

# colori per gruppi anatomici (BGR non serve: usiamo matplotlib RGB)
EDGE_COLOR = "lime"
JOINT_COLOR = "yellow"
DERIVED_COLOR = "red"   # Neck, CHip


def find_video_file(video_root, set_id, video_id):
    """Cerca <root>/<set>/<video_id>.mp4 (e qualche variante di nome)."""
    root = Path(video_root).expanduser()
    candidates = [
        root / set_id / f"{video_id}.mp4",
        root / set_id / f"{video_id}.MP4",
        root / set_id / f"{video_id}.mov",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def extract_frames(video_path, frame_numbers):
    """Estrae i frame ai frame-number assoluti dati. Ritorna dict fn->RGB img."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = {}
    for fn in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fn))
        ok, img = cap.read()
        if ok:
            frames[fn] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            frames[fn] = None
            print(f"[WARN] frame {fn} non leggibile (video ha {total} frame)")
    cap.release()
    return frames


def get_raw_window(cache, set_id, video_id, pid, frames):
    """Keypoint grezzi [T,19,3] in pixel + bbox [T,4], dal pkl (no norm)."""
    kp17, mask = cache.get_window(set_id, video_id, pid, frames, fill="nan")
    kp19 = derive_19_joints(kp17)                     # [T,19,3] pixel
    key = (str(set_id), video_id, str(pid))
    _, bb_all = cache._store[key]
    idx_map = cache._frame_index[key]
    T = len(frames)
    bbox = np.full((T, 4), np.nan, dtype=np.float32)
    for t, fn in enumerate(frames):
        row = idx_map.get(int(fn))
        if row is not None:
            bbox[t] = bb_all[row]
    return kp19, bbox, mask


def draw_on_ax(ax, img, kp19, bbox, crop_pad=60):
    """Disegna scheletro+bbox su un frame. Se img None, sfondo neutro."""
    xy = kp19[:, :2]
    if img is not None:
        ax.imshow(img)
    else:
        ax.add_patch(plt.Rectangle((0, 0), IMG_W, IMG_H,
                                   facecolor="0.15", edgecolor="none"))

    # edges
    for i, j in JOINT_EDGES:
        if np.all(np.isfinite(xy[[i, j]])):
            ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]],
                    "-", color=EDGE_COLOR, lw=2, zorder=2)
    # joints
    ax.scatter(xy[:, 0], xy[:, 1], s=14, color=JOINT_COLOR, zorder=3)
    ax.scatter(xy[[NECK, CHIP], 0], xy[[NECK, CHIP], 1], s=32,
               facecolors="none", edgecolors=DERIVED_COLOR, linewidths=1.6,
               zorder=4)
    # bbox
    if np.all(np.isfinite(bbox)):
        x1, y1, x2, y2 = bbox
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, edgecolor="cyan", lw=1.5,
                                   zorder=1))
        # zoom attorno al pedone
        ax.set_xlim(max(0, x1 - crop_pad), min(IMG_W, x2 + crop_pad))
        ax.set_ylim(min(IMG_H, y2 + crop_pad), max(0, y1 - crop_pad))
    else:
        ax.set_xlim(0, IMG_W)
        ax.set_ylim(IMG_H, 0)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_grid(frames_imgs, kp19, bbox, frame_numbers, out_path, title_prefix=""):
    T = len(frame_numbers)
    ncols = 4
    nrows = int(np.ceil(T / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    for t in range(T):
        fn = frame_numbers[t]
        img = frames_imgs.get(fn) if frames_imgs else None
        draw_on_ax(axes[t], img, kp19[t], bbox[t])
        axes[t].set_title(f"frame {fn}", fontsize=9)
    for t in range(T, len(axes)):
        axes[t].axis("off")
    fig.suptitle(f"{title_prefix}Scheletri GREZZI (pixel) sul video — {T} frame",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose_dir", default="data/poses")
    ap.add_argument("--set", dest="set_id", default="set05")
    ap.add_argument("--video", dest="video_id", default=None)
    ap.add_argument("--pid", default=None)
    ap.add_argument("--video_root", default="~/Desktop/pie_video")
    ap.add_argument("--obs_len", type=int, default=16)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default="out_viz_video")
    ap.add_argument("--no_video", action="store_true",
                    help="disegna su sfondo neutro (senza .mp4)")
    args = ap.parse_args()

    cache = PoseCache(args.pose_dir, set_ids=[args.set_id])

    if args.pid is None or args.video_id is None:
        for (s, v, p), (kp, bb) in cache._store.items():
            if s == args.set_id and kp.shape[0] >= args.obs_len:
                args.video_id, args.pid = v, p
                break
        print(f"[auto] track scelta: {args.set_id}/{args.video_id}/{args.pid}")

    key = (args.set_id, args.video_id, str(args.pid))
    kp_all, _ = cache._store[key]
    inv = {row: fn for fn, row in cache._frame_index[key].items()}
    all_frames = [inv[r] for r in range(kp_all.shape[0])]

    s, e = args.start, args.start + args.obs_len
    assert e <= len(all_frames), \
        f"start+obs_len={e} supera la track ({len(all_frames)} frame)"
    frames = all_frames[s:e]
    print(f"finestra: frame assoluti {frames[0]}..{frames[-1]} (obs_len={args.obs_len})")

    kp19, bbox, mask = get_raw_window(cache, args.set_id, args.video_id,
                                      args.pid, frames)
    print(f"coverage pose: {100*mask.mean():.0f}%")

    frames_imgs = None
    if not args.no_video:
        vpath = find_video_file(args.video_root, args.set_id, args.video_id)
        if vpath is None:
            print(f"[WARN] video non trovato in "
                  f"{args.video_root}/{args.set_id}/{args.video_id}.mp4 "
                  f"-> uso sfondo neutro (--no_video)")
        else:
            print(f"video: {vpath}")
            frames_imgs = extract_frames(vpath, frames)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.set_id}_{args.video_id}_{args.pid}_RAW"
    p = plot_grid(frames_imgs, kp19, bbox, frames, outdir / f"{tag}_onvideo.png",
                  title_prefix=f"[{args.set_id}/{args.video_id}/{args.pid}] ")
    print("salvato:", p)


if __name__ == "__main__":
    main()
