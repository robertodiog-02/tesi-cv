#!/usr/bin/env python3
"""
draw_skeletons.py
=================
Sovrappone gli scheletri ViTPose (dal pickle) ai frame originali del video PIE,
per un pedone e un range di frame, cosi' puoi vedere COSA ha prodotto ViTPose
anche nei frame occlusi.

USO tipico:
  python draw_skeletons.py \
      --skel /Users/robertodioguardi/Downloads/jaadpie_pose/skeleton/pie/set01_video_0002.pkl \
      --video /Users/robertodioguardi/Desktop/pie_video/set01/video_0002.mp4 \
      --pid 1_2_23 --f_start 5101 --f_end 5162 \
      --out /Users/robertodioguardi/Desktop/pie_video/out_1_2_23

Produce, nella cartella --out:
  - un PNG per ogni frame della finestra (con scheletro disegnato se presente,
    e una scritta "NO SKELETON" se il pid non ha quel frame nel pickle);
  - un montaggio 'montage.png' con tutte le miniature in griglia.

Richiede: opencv-python, numpy   (pip install opencv-python numpy)

NOTE:
 - Il pickle e' indicizzato: skel[pid][frame_int] = lista di K keypoint [x,y(,conf)].
   Il formato qui e' COCO-WholeBody (133 punti); disegniamo i primi 17 (il corpo).
 - I frame del video sono numerati a partire da 0; il pickle usa lo stesso indice
   del nome immagine PIE (es. 5101 = frame 5101 del video). Se trovi un offset di 1,
   usa --frame_offset.
"""

import os
import argparse
import numpy as np

try:
    import pickle5 as pickle
except Exception:
    import pickle

import cv2


# COCO-17 body order (primi 17 del whole-body)
COCO_NAMES = ['nose', 'eyeL', 'eyeR', 'earL', 'earR', 'shL', 'shR',
              'elbL', 'elbR', 'wrL', 'wrR', 'hipL', 'hipR',
              'kneeL', 'kneeR', 'ankL', 'ankR']

# edge del corpo (coppie di indici COCO-17)
SKELETON_EDGES = [
    (5, 6),            # spalle
    (5, 7), (7, 9),    # braccio sx
    (6, 8), (8, 10),   # braccio dx
    (5, 11), (6, 12),  # tronco
    (11, 12),          # bacino
    (11, 13), (13, 15),  # gamba sx
    (12, 14), (14, 16),  # gamba dx
    (0, 5), (0, 6),    # collo-naso
]

# keypoint scartati dal repo (occhi/orecchie): li disegniamo in grigio chiaro
KEPT = set(i for i in range(17) if i not in {1, 2, 3, 4})


def load_skeletons(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def draw_one(frame_img, kps, present):
    """Disegna lo scheletro (primi 17 kp) su una copia del frame."""
    img = frame_img.copy()
    h, w = img.shape[:2]
    if not present:
        cv2.putText(img, "NO SKELETON IN PICKLE", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return img

    pts = np.array(kps[:17], dtype=float)[:, :2]  # primi 17, solo x,y

    # edge
    for a, b in SKELETON_EDGES:
        pa, pb = pts[a], pts[b]
        if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
            cv2.line(img, tuple(pa.astype(int)), tuple(pb.astype(int)),
                     (0, 255, 0), 2)
    # punti
    for i, (x, y) in enumerate(pts):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        color = (0, 180, 255) if i in KEPT else (180, 180, 180)
        r = 4 if i in KEPT else 2
        cv2.circle(img, (int(x), int(y)), r, color, -1)

    # bounding box dei keypoint del corpo tenuti (utile per vedere se "esplode")
    kept_pts = pts[sorted(KEPT)]
    finite = kept_pts[np.all(np.isfinite(kept_pts), axis=1)]
    if len(finite):
        x0, y0 = finite.min(0).astype(int)
        x1, y1 = finite.max(0).astype(int)
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 255), 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skel', required=True, help='pickle scheletro del video (es. set01_video_0002.pkl)')
    ap.add_argument('--video', required=True, help='mp4 originale del video')
    ap.add_argument('--pid', required=True, help='es. 1_2_23')
    ap.add_argument('--f_start', type=int, required=True)
    ap.add_argument('--f_end', type=int, required=True)
    ap.add_argument('--out', required=True, help='cartella di output')
    ap.add_argument('--frame_offset', type=int, default=0,
                    help='offset tra numero frame del pickle e indice nel video (default 0)')
    ap.add_argument('--montage_cols', type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    skel = load_skeletons(args.skel)
    if args.pid not in skel:
        cands = [k for k in skel.keys() if args.pid.split('_')[-1] in str(k)]
        print(f"pid '{args.pid}' non trovato. Candidati simili: {cands[:10]}")
        return
    frames_dict = skel[args.pid]
    present_frames = set(int(k) for k in frames_dict.keys())
    print(f"pid {args.pid}: {len(present_frames)} frame con scheletro "
          f"(range {min(present_frames)}..{max(present_frames)})")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("ERRORE: non riesco ad aprire il video:", args.video)
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video: {total} frame totali")

    thumbs = []
    for fr in range(args.f_start, args.f_end + 1):
        video_idx = fr + args.frame_offset
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_idx)
        ok, img = cap.read()
        if not ok:
            print(f"  frame {fr}: impossibile leggere dal video (idx {video_idx})")
            continue

        present = fr in present_frames
        kps = frames_dict[fr] if present else None
        ann = draw_one(img, kps, present)

        tag = "OK" if present else "MISSING"
        cv2.putText(ann, f"frame {fr} [{tag}]", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0) if present else (0, 0, 255), 2)

        outp = os.path.join(args.out, f"frame_{fr}_{tag}.png")
        cv2.imwrite(outp, ann)

        # miniatura per il montaggio
        th = cv2.resize(ann, (320, 180))
        thumbs.append(th)

    cap.release()

    # montaggio in griglia
    if thumbs:
        cols = args.montage_cols
        rows = (len(thumbs) + cols - 1) // cols
        cell_h, cell_w = thumbs[0].shape[:2]
        canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
        for i, th in enumerate(thumbs):
            r, c = divmod(i, cols)
            canvas[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = th
        mpath = os.path.join(args.out, "montage.png")
        cv2.imwrite(mpath, canvas)
        print("Montaggio salvato in:", mpath)
    print("Frame annotati salvati in:", args.out)


if __name__ == "__main__":
    main()
