"""
Dump diagnostico per SEQUENZA — cosa arriva davvero al modello
==============================================================
Scopo (richiesta relatrice): prendere UN pedone, costruire le sue finestre
di osservazione esattamente come fa il dataset, e per OGNI finestra stampare
i dati che arrivano al modello: il tensore [T, 19, 5] dove ogni giunto e'
(x, y, conf, cx, cy) = coordinate keypoint normalizzate + centro bbox.

Per ogni sequenza stampa:
  - metadati (frame assoluti coperti, label, TTE, end-point)
  - statistiche per canale (min/max/mean/std) sui 5 canali
  - i valori frame-by-frame di alcuni giunti chiave
  - la traiettoria del centro bbox lungo la sequenza
Cosi' si vede in che range vanno i dati e se la traiettoria e' preservata.

Uso:
    python dump_keypoints.py --pose_dir data/poses --set set01 \\
        --video video_0001 --pid 1_1_1 --norm reference_point \\
        --obs_len 26 [--max_windows 3] [--full] [--save dump.npz]

  --full        : stampa TUTTI i 19 giunti per ogni frame (molto verboso)
  --max_windows : quante finestre stampare (default 3)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "data"))

from pose_cache import PoseCache
from pose_preproc import (derive_19_joints, concat_center,
                          normalize_pose, fill_missing)
from skeleton import JOINT_NAMES

CH = ["x", "y", "conf", "cx", "cy"]
KEY_JOINTS = ["nose", "left_shoulder", "right_shoulder",
              "left_hip", "right_hip", "left_ankle", "right_ankle",
              "Neck", "CHip"]


def chan_stats(arr2d, label):
    """arr2d: [T, 19] di un singolo canale -> stat globali sulla finestra."""
    a = arr2d[np.isfinite(arr2d)]
    if a.size == 0:
        print(f"    {label:6s}: (vuoto)")
        return
    print(f"    {label:6s}: min={a.min():8.3f}  max={a.max():8.3f}  "
          f"mean={a.mean():8.3f}  std={a.std():7.3f}")


def build_windows(frames, obs_len, tte_min, tte_max, endpoint_step, anchor):
    """Ricostruisce gli start delle finestre come build_samples (per indici)."""
    T = len(frames)
    if T < obs_len + tte_min:
        return []
    if anchor:
        end_hi = T - tte_min
        end_lo = T - tte_max
        w_ends = list(range(end_hi, end_lo - 1, -endpoint_step))
        w_ends = [we for we in w_ends if we - obs_len >= 0]
        starts = [we - obs_len for we in sorted(w_ends)]
    else:
        start_idx = max(0, T - obs_len - tte_max)
        end_idx = T - obs_len - tte_min
        step = max(1, round(obs_len * 0.4))
        starts = list(range(start_idx, end_idx + 1, step))
    return starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose_dir", default="data/poses")
    ap.add_argument("--set", default="set01")
    ap.add_argument("--video", default=None)
    ap.add_argument("--pid", default=None)
    ap.add_argument("--norm", default="reference_point",
                    choices=["reference_point", "reference_point_seq",
                             "reference_point_perframe", "hip_reference",
                             "minmax", "none"])
    ap.add_argument("--obs_len", type=int, default=26)
    ap.add_argument("--tte_min", type=int, default=30)
    ap.add_argument("--tte_max", type=int, default=60)
    ap.add_argument("--endpoint_step", type=int, default=6)
    ap.add_argument("--anchor", action="store_true",
                    help="finestre ancorate agli end-point (come val/test)")
    ap.add_argument("--max_windows", type=int, default=3)
    ap.add_argument("--full", action="store_true",
                    help="stampa tutti i 19 giunti per ogni frame")
    ap.add_argument("--img_w", type=float, default=1920.0)
    ap.add_argument("--img_h", type=float, default=1080.0)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    cache = PoseCache(args.pose_dir, set_ids=[args.set], verbose=True)

    import pickle
    f = next(Path(args.pose_dir).glob(f"*{args.set}*.pkl"))
    d = pickle.load(open(f, "rb"))
    set_key = list(d.keys())[0]
    video = args.video or list(d[set_key].keys())[0]
    pid = args.pid or list(d[set_key][video].keys())[0]

    frames_all = np.array(d[set_key][video][pid]["frames"])
    bbox_all = np.array(d[set_key][video][pid]["bbox"])
    T_track = len(frames_all)

    print("\n" + "=" * 70)
    print(f"PEDONE: set={set_key}  video={video}  pid={pid}")
    print(f"  track: {T_track} frame, da {frames_all[0]} a {frames_all[-1]}")
    print(f"  obs_len={args.obs_len}  TTE=[{args.tte_min},{args.tte_max}]  "
          f"norm={args.norm}  anchor={args.anchor}")
    print("=" * 70)

    starts = build_windows(frames_all, args.obs_len, args.tte_min,
                           args.tte_max, args.endpoint_step, args.anchor)
    if not starts:
        print("Track troppo corta per generare finestre.")
        return
    print(f"\nFinestre generate per questo pedone: {len(starts)}")
    print(f"(ne stampo al massimo {args.max_windows})\n")

    saved = []
    for wi, w_start in enumerate(starts[:args.max_windows]):
        w_end = w_start + args.obs_len
        win_frames = frames_all[w_start:w_end]
        kp17, mask = cache.get_window(set_key, video, pid,
                                      win_frames.tolist(), fill="nan")
        bb = bbox_all[w_start:w_end]
        center = np.stack([(bb[:, 0] + bb[:, 2]) / 2,
                           (bb[:, 1] + bb[:, 3]) / 2], axis=1)

        kp19 = derive_19_joints(kp17)
        feat_px = fill_missing(concat_center(kp19, center))
        feat = normalize_pose(feat_px.copy(), method=args.norm,
                              img_w=args.img_w, img_h=args.img_h)
        tte = T_track - w_end

        print("─" * 70)
        print(f"[FINESTRA {wi+1}/{min(len(starts),args.max_windows)}]")
        print(f"  frame assoluti: {win_frames[0]} … {win_frames[-1]} "
              f"(end-point = {win_frames[-1]})")
        print(f"  TTE = {tte} frame   coverage pose = {mask.mean():.0%}")
        print(f"  tensore al modello: shape {feat.shape} = [T={feat.shape[0]}, "
              f"giunti={feat.shape[1]}, canali={feat.shape[2]}]")

        print("\n  Statistiche per canale (sull'intera finestra, NORMALIZZATO):")
        for ci, name in enumerate(CH):
            chan_stats(feat[:, :, ci], name)

        # traiettoria del centro
        print("\n  Traiettoria centro bbox lungo la sequenza:")
        print(f"    cx norm: {np.round(feat[:,0,3],3).tolist()}")
        print(f"    cy norm: {np.round(feat[:,0,4],3).tolist()}")
        dcx = float(feat[:,0,3].max() - feat[:,0,3].min())
        print(f"    range cx = {dcx:.4f} "
              f"({'PRESERVATA' if dcx > 1e-3 else 'AZZERATA!'})")

        # valori frame-by-frame
        joints = list(range(19)) if args.full else \
                 [JOINT_NAMES.index(j) for j in KEY_JOINTS]
        print(f"\n  Valori (x,y) normalizzati frame-by-frame "
              f"({'tutti i 19 giunti' if args.full else 'giunti chiave'}):")
        # intestazione: alcuni frame campione
        show_t = range(feat.shape[0]) if feat.shape[0] <= 6 else \
                 [0, 1, 2, feat.shape[0]//2, feat.shape[0]-2, feat.shape[0]-1]
        for j in joints:
            vals = "  ".join(f"t{t}:({feat[t,j,0]:+.2f},{feat[t,j,1]:+.2f})"
                             for t in show_t)
            print(f"    {JOINT_NAMES[j]:14s} {vals}")
        print()

        saved.append(dict(window=wi, frames=win_frames, feat_px=feat_px,
                          feat_norm=feat, tte=tte))

    if args.save:
        np.savez(args.save, **{f"win{ s['window'] }_frames": s["frames"] for s in saved},
                 **{f"win{ s['window'] }_norm": s["feat_norm"] for s in saved},
                 **{f"win{ s['window'] }_pixel": s["feat_px"] for s in saved})
        print(f"Salvato dump completo in {args.save}")


if __name__ == "__main__":
    main()
