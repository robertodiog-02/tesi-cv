#!/usr/bin/env python3
"""
draw_skeletons_debug.py
=======================
Versione diagnostica: disegna lo stesso frame in PIU' VARIANTI di orientamento
dei keypoint, affiancate, per capire quale convenzione incolla lo scheletro al
corpo del pedone. Disegna anche la bbox UFFICIALE (dal pickle sequences) se la
trovi, per confermare l'identita' del pedone.

USO:
  python draw_skeletons_debug.py \
      --skel /Users/robertodioguardi/Downloads/jaadpie_pose/skeleton/pie/set01_video_0002.pkl \
      --video /Users/robertodioguardi/Desktop/pie_video/set01/video_0002.mp4 \
      --pid 1_2_23 --frame 5101 \
      --out /Users/robertodioguardi/Desktop/pie_video/debug_1_2_23 \
      [--seq /Users/robertodioguardi/Downloads/jaadpie_pose/sequences/pie/<split>/combined/set01.pkl]

Produce nella cartella --out un'immagine 'variants_<frame>.png' con 4 pannelli:
  (a) RAW         -> (x, y) cosi' come nel pickle
  (b) SWAP        -> (y, x)  (colonne invertite)
  (c) ROT+FLIP    -> la trasformazione esatta del repo (rotate_and_flip_points)
  (d) SWAP+FLIPH  -> (y,x) poi flip orizzontale attorno al centro

Guarda quale pannello sovrappone lo scheletro al corpo reale: quella e' la
convenzione corretta da usare in TUTTI gli script.
"""

import os
import argparse
import numpy as np

try:
    import pickle5 as pickle
except Exception:
    import pickle

import cv2

SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
]
KEPT = set(i for i in range(17) if i not in {1, 2, 3, 4})


def load_pkl(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def rotate_and_flip_points(points, flip_flag=False):
    """Replica ESATTA della funzione del repo (su tutti i punti passati)."""
    points = np.array(points, dtype=float)
    cx = np.mean(points[:, 0])
    cy = np.mean(points[:, 1])
    center = np.array([cx, cy])
    tp = points - center
    if not flip_flag:
        rot = np.array([[-y, x] for x, y in tp])
    else:
        rot = np.array([[y, -x] for x, y in tp])
    rot += center
    flipped = rot.copy()
    flipped[:, 0] = cx - (flipped[:, 0] - cx)
    return flipped


def variant_points(kps17, mode):
    p = np.array(kps17, dtype=float)[:, :2]
    if mode == 'RAW':
        return p
    if mode == 'SWAP':
        return p[:, ::-1].copy()
    if mode == 'ROT+FLIP':
        return rotate_and_flip_points(p, flip_flag=False)
    if mode == 'ROT-90':
        # rotazione oraria -90 attorno al baricentro, senza flip
        c = p.mean(0); t = p - c
        r = np.array([[y, -x] for x, y in t]) + c
        return r
    if mode == 'ROT+90':
        c = p.mean(0); t = p - c
        r = np.array([[-y, x] for x, y in t]) + c
        return r
    if mode == 'SWAP+FLIPV':
        q = p[:, ::-1].copy()
        cy = q[:, 1].mean()
        q[:, 1] = cy - (q[:, 1] - cy)
        return q
    return p
    if mode == 'SWAP':
        return p[:, ::-1].copy()
    if mode == 'ROT+FLIP':
        return rotate_and_flip_points(p, flip_flag=False)
    if mode == 'SWAP+FLIPH':
        q = p[:, ::-1].copy()
        cx = q[:, 0].mean()
        q[:, 0] = cx - (q[:, 0] - cx)
        return q
    return p


def draw_skeleton(img, pts, color_line=(0, 255, 0)):
    out = img.copy()
    for a, b in SKELETON_EDGES:
        pa, pb = pts[a], pts[b]
        if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
            cv2.line(out, tuple(pa.astype(int)), tuple(pb.astype(int)), color_line, 2)
    for i, (x, y) in enumerate(pts):
        if np.isfinite(x) and np.isfinite(y):
            col = (0, 180, 255) if i in KEPT else (170, 170, 170)
            cv2.circle(out, (int(x), int(y)), 4 if i in KEPT else 2, col, -1)
    return out


def find_bbox(seq_path, pid, frame):
    """Cerca la bbox ufficiale del pid al frame dato nel pickle sequences."""
    if not seq_path or not os.path.exists(seq_path):
        return None
    try:
        data = load_pkl(seq_path)
    except Exception:
        return None
    # struttura: lista di dict con 'ped_data'; ogni ped ha pid, bbox, path
    for top in data:
        for ped in top.get('ped_data', []):
            if str(ped.get('pid')) == str(pid):
                img_path = top.get('path', '')
                try:
                    fnum = int(os.path.splitext(os.path.basename(img_path))[0])
                except Exception:
                    fnum = None
                if fnum == frame:
                    return ped.get('bbox')
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skel', required=True)
    ap.add_argument('--video', required=True)
    ap.add_argument('--pid', required=True)
    ap.add_argument('--frame', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seq', default=None, help='(opzionale) pickle sequences per la bbox ufficiale')
    ap.add_argument('--frame_offset', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    skel = load_pkl(args.skel)
    if args.pid not in skel:
        print("pid non trovato. Disponibili:", list(skel.keys())[:15])
        return
    if args.frame not in skel[args.pid]:
        fk = sorted(int(x) for x in skel[args.pid].keys())
        print(f"frame {args.frame} non presente per {args.pid}. Range {fk[0]}..{fk[-1]}")
        print("Uso il frame valido piu' vicino.")
        args.frame = min(fk, key=lambda f: abs(f - args.frame))
        print("frame scelto:", args.frame)

    kps = skel[args.pid][args.frame]

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame + args.frame_offset)
    ok, base = cap.read()
    cap.release()
    if not ok:
        print("Impossibile leggere il frame dal video.")
        return

    bbox = find_bbox(args.seq, args.pid, args.frame)

    modes = ['RAW', 'ROT+FLIP', 'ROT-90', 'ROT+90', 'SWAP', 'SWAP+FLIPV']
    panels = []
    for mode in modes:
        pts = variant_points(kps, mode)
        img = draw_skeleton(base, pts)
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(img, mode, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        panels.append(img)

    h, w = panels[0].shape[:2]
    r1 = np.hstack([panels[0], panels[1], panels[2]])
    r2 = np.hstack([panels[3], panels[4], panels[5]])
    grid = np.vstack([r1, r2])
    grid = cv2.resize(grid, (w, h))  # rimpicciolisce per comodita'
    outp = os.path.join(args.out, f"variants_{args.frame}.png")
    cv2.imwrite(outp, grid)
    # salva anche full-res separati
    for mode, img in zip(modes, panels):
        cv2.imwrite(os.path.join(args.out, f"{mode.replace('+','_')}_{args.frame}.png"), img)
    print("Salvato:", outp)
    print("E i singoli pannelli full-res nella stessa cartella.")
    if bbox is None and args.seq:
        print("NB: bbox ufficiale non trovata nel pickle sequences fornito.")


if __name__ == "__main__":
    main()
