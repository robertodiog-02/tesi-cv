 #!/usr/bin/env python3
"""
analyze_pie.py
==============
Analizza la cartella jaadpie_pose per il dataset PIE, replicando la logica del
dataloader di SGNetPose (custom_data_layer.py) e producendo le statistiche che
servono per decidere come gestire i frame mancanti (occlusioni / ViTPose failure)
nel caso delle FINESTRE FISSE del benchmark di Kotseruba/Rasouli.

USO:
    python analyze_pie.py --root /users/robertodioguardi/downloads/jaadpie_pose

Cosa fa:
 1) Apre uno skeleton pickle e ne stampa la STRUTTURA reale (per non indovinare).
 2) Ispeziona il pedone richiesto (default 1_2_23 = set01, video_0002, pid contenente '23')
    sui frame [5101, 5162]: dice quali frame ci sono e quali MANCANO.
 3) Calcola, su tutto PIE-train (o split scelto), la distribuzione dei BUCHI:
    quante finestre fisse di lunghezza L avrebbero >=1 frame mancante, e di che
    dimensione sono i buchi (così sai se l'interpolazione breve basta).
 4) Replica calc_angles + norm_coords del repo, così puoi verificarli.

NOTA: lo script NON assume nulla sulla struttura interna oltre a ciò che il codice
del repo usa: skeleton_dict[pid][frame][k][0:2] con k in 0..16 (COCO), scartando 1,2,3,4.
Se la struttura reale differisce, lo step (1) te lo mostra e adatti le chiavi.
"""

import os
import sys
import glob
import argparse
import fnmatch
import numpy as np

try:
    import pickle5 as pickle
except Exception:
    import pickle


# ----------------------------------------------------------------------------
# Keypoint handling (identico al repo: 17 COCO, si scartano occhi/orecchie 1-4)
# ----------------------------------------------------------------------------
KEPT_INDICES = [k for k in range(17) if k not in {1, 2, 3, 4}]  # -> 13 punti
# ordine risultante (indice nel vettore 'only_coords' di 13 elementi):
# 0 naso, 1 spalla L, 2 spalla R, 3 gomito L, 4 gomito R, 5 polso L, 6 polso R,
# 7 anca L, 8 anca R, 9 ginocchio L, 10 ginocchio R, 11 caviglia L, 12 caviglia R


def compute_angle(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    cos_theta = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta)) / 180.0


def calc_angles(sk):
    """sk: array (13,2) di keypoint nell'ordine KEPT_INDICES. Replica il repo."""
    sk = np.asarray(sk, dtype=float)
    mid = [(sk[1][0] + sk[2][0]) / 2, (sk[1][1] + sk[2][1]) / 2]
    angles = [
        compute_angle(sk[1] - mid, sk[0] - mid),          # left_shoulder_nose
        compute_angle(sk[2] - mid, sk[0] - mid),          # right_shoulder_nose
        compute_angle(sk[3] - sk[1], sk[2] - sk[1]),      # left_armpit
        compute_angle(sk[4] - sk[2], sk[2] - sk[1]),      # right_armpit
        compute_angle(sk[5] - sk[3], sk[1] - sk[3]),      # left_elbow
        compute_angle(sk[6] - sk[4], sk[2] - sk[4]),      # right_elbow
        compute_angle(sk[9] - sk[7], sk[7] - sk[1]),      # left_hip
        compute_angle(sk[10] - sk[8], sk[8] - sk[2]),     # right_hip
        compute_angle(sk[9] - sk[7], sk[7] - sk[8]),      # left_thigh
        compute_angle(sk[10] - sk[8], sk[7] - sk[8]),     # right_thigh
        compute_angle(sk[11] - sk[9], sk[7] - sk[9]),     # left_knee
        compute_angle(sk[12] - sk[10], sk[8] - sk[10]),   # right_knee
    ]
    return np.nan_to_num(np.array(angles), nan=0.5)


def norm_coords(sk):
    """Min-max per-frame sui keypoint stessi -> [0,1]. Replica il repo."""
    arr = np.asarray(sk, dtype=float)
    x, y = arr[:, 0], arr[:, 1]
    rx = (x.max() - x.min()) or 1.0
    ry = (y.max() - y.min()) or 1.0
    nx = (x - x.min()) / rx
    ny = (y - y.min()) / ry
    return np.column_stack((nx, ny))


# ----------------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def find_skeleton_dir(root):
    for cand in [os.path.join(root, "skeleton", "pie"),
                 os.path.join(root, "skeleton", "PIE")]:
        if os.path.isdir(cand):
            return cand
    # fallback: cerca una dir 'skeleton'
    hits = glob.glob(os.path.join(root, "**", "skeleton", "**", ""), recursive=True)
    return hits[0] if hits else None


# ----------------------------------------------------------------------------
# (1) Ispezione struttura
# ----------------------------------------------------------------------------
def inspect_structure(skel_dir):
    print("=" * 70)
    print("STEP 1 — STRUTTURA DEI PICKLE SCHELETRO")
    print("=" * 70)
    files = sorted(glob.glob(os.path.join(skel_dir, "*.pkl")))
    print(f"Trovati {len(files)} file scheletro in {skel_dir}")
    if not files:
        return None
    f0 = files[0]
    print(f"Apro: {os.path.basename(f0)}")
    d = load_pickle(f0)
    print(f"  tipo top-level: {type(d)}")
    if isinstance(d, dict):
        pids = list(d.keys())
        print(f"  n. pid (chiavi): {len(pids)}  esempi: {pids[:5]}")
        p0 = pids[0]
        frames = d[p0]
        print(f"  pid '{p0}' -> tipo {type(frames)}")
        if isinstance(frames, dict):
            fk = sorted(frames.keys())
            print(f"    n. frame: {len(fk)}  range: {fk[0]}..{fk[-1]}  esempi: {fk[:5]}")
            kp = frames[fk[0]]
            arr = np.array(kp)
            print(f"    keypoint per frame: tipo {type(kp)}  shape {arr.shape}")
            print(f"    primo keypoint (k=0): {np.array(kp[0])}")
    return files


# ----------------------------------------------------------------------------
# (2) Ispezione del pedone specifico
# ----------------------------------------------------------------------------
def inspect_pedestrian(skel_dir, set_id, video_id, pid_substr, f_start, f_end):
    print("=" * 70)
    print(f"STEP 2 — PEDONE set{set_id:02d}_video_{video_id:04d}, pid~'{pid_substr}', "
          f"frame {f_start}..{f_end}")
    print("=" * 70)
    pattern = f"set{set_id:02d}_video_{video_id:04d}"
    matches = [p for p in glob.glob(os.path.join(skel_dir, "*.pkl"))
               if fnmatch.fnmatch(os.path.basename(p), f"{pattern}*")]
    if not matches:
        print(f"  Nessun file scheletro corrisponde a {pattern}*")
        return
    print(f"  File: {[os.path.basename(m) for m in matches]}")

    skel = {}
    for m in matches:
        skel.update(load_pickle(m))

    # trova il pid che contiene la substring richiesta
    cand_pids = [k for k in skel.keys() if pid_substr in str(k)]
    print(f"  pid candidati (contengono '{pid_substr}'): {cand_pids[:10]}")
    if not cand_pids:
        print("  pid non trovato; elenco primi pid disponibili:", list(skel.keys())[:10])
        return

    pid = cand_pids[0]
    frames = skel[pid]
    present = sorted(int(fr) for fr in frames.keys())
    want = list(range(f_start, f_end + 1))
    present_set = set(present)
    missing = [fr for fr in want if fr not in present_set]

    print(f"\n  pid usato: {pid}")
    print(f"  frame totali con scheletro per questo pid: {len(present)} "
          f"(range {present[0]}..{present[-1]})")
    print(f"  finestra richiesta: {f_start}..{f_end}  ({len(want)} frame)")
    print(f"  presenti nella finestra: {len(want) - len(missing)}/{len(want)}")
    if missing:
        print(f"  >>> MANCANTI: {missing}")
        # raggruppa i buchi consecutivi
        gaps = []
        s = missing[0]
        prev = missing[0]
        for fr in missing[1:]:
            if fr == prev + 1:
                prev = fr
            else:
                gaps.append((s, prev))
                s = prev = fr
        gaps.append((s, prev))
        print("  >>> buchi (start,end,len):",
              [(a, b, b - a + 1) for a, b in gaps])
    else:
        print("  >>> Nessun frame mancante: finestra completa.")

    # mostra angoli/norm per il primo frame valido della finestra
    first_valid = next((fr for fr in want if fr in present_set), None)
    if first_valid is not None:
        raw = []
        for k in KEPT_INDICES:
            raw.append(np.array(frames[first_valid][k][0:2], dtype=float))
        raw = np.array(raw)
        print(f"\n  Esempio frame {first_valid}:")
        print(f"    keypoint grezzi (13x2), primi 3:\n{raw[:3]}")
        print(f"    norm_coords primi 3:\n{np.round(norm_coords(raw)[:3], 3)}")
        print(f"    12 angoli (norm /180, NaN->0.5):\n{np.round(calc_angles(raw), 3)}")


# ----------------------------------------------------------------------------
# (3) Statistiche buchi su tutto lo split (per finestre fisse del benchmark)
# ----------------------------------------------------------------------------
def gap_statistics(skel_dir, win_len=16, max_gap_report=10):
    print("=" * 70)
    print(f"STEP 3 — STATISTICHE BUCHI (finestre fisse L={win_len})")
    print("=" * 70)
    files = sorted(glob.glob(os.path.join(skel_dir, "*.pkl")))
    total_pids = 0
    total_gap1 = 0   # pid con almeno un buco
    gap_size_hist = {}
    longest_run_lt_win = 0

    for fp in files:
        d = load_pickle(fp)
        if not isinstance(d, dict):
            continue
        for pid, frames in d.items():
            if not isinstance(frames, dict) or len(frames) == 0:
                continue
            total_pids += 1
            present = sorted(int(fr) for fr in frames.keys())
            span = present[-1] - present[0] + 1
            has_gap = span != len(present)
            if has_gap:
                total_gap1 += 1
                # misura i buchi
                for i in range(1, len(present)):
                    g = present[i] - present[i - 1] - 1
                    if g > 0:
                        gap_size_hist[g] = gap_size_hist.get(g, 0) + 1
            # run consecutivo piu' lungo
            longest = 1
            cur = 1
            for i in range(1, len(present)):
                if present[i] == present[i - 1] + 1:
                    cur += 1
                    longest = max(longest, cur)
                else:
                    cur = 1
            if longest < win_len:
                longest_run_lt_win += 1

    print(f"  pid totali analizzati: {total_pids}")
    print(f"  pid con >=1 buco (detection mancante): {total_gap1} "
          f"({100*total_gap1/max(total_pids,1):.1f}%)")
    print(f"  pid il cui run consecutivo piu' lungo e' < {win_len}: {longest_run_lt_win}")
    if gap_size_hist:
        print("  distribuzione dimensione buchi (size: count):")
        for g in sorted(gap_size_hist)[:max_gap_report]:
            print(f"    buco di {g:3d} frame: {gap_size_hist[g]}")
        small = sum(c for g, c in gap_size_hist.items() if g <= 2)
        big = sum(c for g, c in gap_size_hist.items() if g > 2)
        print(f"  buchi <=2 frame (interpolabili facilmente): {small}")
        print(f"  buchi  >2 frame (occlusioni lunghe):        {big}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="percorso a jaadpie_pose (contiene skeleton/, sequences/, input_data/)")
    ap.add_argument("--set_id", type=int, default=1)
    ap.add_argument("--video_id", type=int, default=2)
    ap.add_argument("--pid_substr", type=str, default="23")
    ap.add_argument("--f_start", type=int, default=5101)
    ap.add_argument("--f_end", type=int, default=5162)
    ap.add_argument("--win_len", type=int, default=16)
    ap.add_argument("--skip_stats", action="store_true",
                    help="salta lo step 3 (lento su tutti i file)")
    args = ap.parse_args()

    skel_dir = find_skeleton_dir(args.root)
    if not skel_dir:
        print("ERRORE: non trovo la cartella skeleton/pie sotto", args.root)
        sys.exit(1)
    print("Cartella scheletri:", skel_dir, "\n")

    inspect_structure(skel_dir)
    print()
    inspect_pedestrian(skel_dir, args.set_id, args.video_id,
                       args.pid_substr, args.f_start, args.f_end)
    print()
    if not args.skip_stats:
        gap_statistics(skel_dir, win_len=args.win_len)


if __name__ == "__main__":
    main()
