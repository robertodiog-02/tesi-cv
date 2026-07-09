"""
Finestre REALI del dataset: keypoint GREZZI vs NORMALIZZATI
==========================================================
Dato un PEDONE (ped_id), usa le finestre VERE prodotte da PIEDataset
(protocollo con TTE, le stesse che entrano nel modello) e per OGNI finestra
genera una figura a DUE COLONNE:

    | frame | GREZZI (pixel)        | NORMALIZZATI (hip_reference_seq) |
    |   0   | scheletro in pixel    | scheletro normalizzato           |
    |   1   |        ...            |            ...                   |
    ...

Cosi' vedi, finestra per finestra e frame per frame, come la normalizzazione
trasforma lo scheletro reale.

IMPORTANTE — questo script usa il VERO PIEDataset, quindi richiede:
  - pie_data.py + annotazioni PIE complete in --annotation_root
  - i pkl pose in --pose_dir
Le finestre NON sono tagli arbitrari: sono i campioni reali (con TTE 30-60,
windowing di build_samples). Filtriamo i sample per ped_id e disegniamo
esattamente quelli.

Uso:
     python data/viz_windows_raw_vs_norm.py \                                                                   ─╯
    --split val \
    --annotation_root ~/Desktop/PIE/annotations

Se --ped_id non e' dato, elenca i ped_id disponibili nello split e ne
sceglie uno con piu' finestre.
"""

import argparse
from pathlib import Path
from collections import defaultdict

# --- rendi lo script eseguibile da QUALUNQUE cartella --------------------
# Lo script vive in data/ insieme a pie_dataset.py, pose_cache.py, ecc.
# Aggiungiamo la sua cartella a sys.path così gli import funzionano sia se
# lo lanci da root (python data/viz_windows_raw_vs_norm.py ...) sia da
# dentro data/. Nessun bisogno di 'cd data' o di -m.
import sys
from pathlib import Path
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
# -------------------------------------------------------------------------

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# la pipeline e gli indici scheletro
from pose_preproc import derive_19_joints, concat_center, fill_missing, normalize_pose
from skeleton import JOINT_EDGES, NECK, CHIP

IMG_W, IMG_H = 1920.0, 1080.0


def _draw(ax, xy, color="tab:blue", lw=1.6, ms=14, invert=False):
    for i, j in JOINT_EDGES:
        if np.all(np.isfinite(xy[[i, j]])):
            ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]],
                    "-", color=color, lw=lw, zorder=1)
    ax.scatter(xy[:, 0], xy[:, 1], s=ms, color=color, zorder=2)
    ax.scatter(xy[[NECK, CHIP], 0], xy[[NECK, CHIP], 1], s=ms * 2.2,
               facecolors="none", edgecolors="red", linewidths=1.3, zorder=3)
    ax.set_aspect("equal")
    # NB: l'inversione dell'asse y (per la convenzione immagine, y verso il
    # basso) e' gestita dal CHIAMANTE tramite set_ylim(max, min). Qui NON si
    # inverte, per evitare la doppia inversione (che capovolge lo scheletro).
    if invert:
        ax.invert_yaxis()
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.2)


def raw_keypoints_window(pose_cache, s):
    """Keypoint GREZZI [T,19,2] in pixel per un sample (no normalizzazione)."""
    kp17, mask = pose_cache.get_window(s["set_id"], s["video_id"], s["ped_id"],
                                       s["frames"], fill="nan")
    kp19 = derive_19_joints(kp17)                 # [T,19,3] pixel
    return kp19[:, :, :2], mask


def norm_keypoints_window(dataset, s):
    """Keypoint NORMALIZZATI [T,19,2] chiamando la pipeline del dataset."""
    feat = dataset._get_pose(s).numpy()           # [T,19,C] (C=5 o 3)
    return feat[:, :, :2]


def plot_window(raw_xy, norm_xy, frames, tte, out_path, ped_id, norm_name):
    T = raw_xy.shape[0]
    fig, axes = plt.subplots(T, 2, figsize=(6.4, 2.7 * T))
    if T == 1:
        axes = axes.reshape(1, 2)

    # limiti comuni per colonna (scala coerente lungo la finestra)
    def lims(xy):
        f = xy[np.isfinite(xy).all(axis=2)]
        if len(f) == 0:
            return (0, 1, 0, 1)
        pad_x = 0.12 * (f[:, 0].max() - f[:, 0].min() + 1e-6)
        pad_y = 0.12 * (f[:, 1].max() - f[:, 1].min() + 1e-6)
        return (f[:, 0].min() - pad_x, f[:, 0].max() + pad_x,
                f[:, 1].min() - pad_y, f[:, 1].max() + pad_y)

    rx0, rx1, ry0, ry1 = lims(raw_xy)
    nx0, nx1, ny0, ny1 = lims(norm_xy)

    for t in range(T):
        axL, axR = axes[t]
        _draw(axL, raw_xy[t], color="tab:gray")
        _draw(axR, norm_xy[t], color="tab:blue")
        # y in convenzione IMMAGINE: cresce verso il basso -> set_ylim(max, min).
        # Una sola inversione qui (dentro _draw non si inverte piu').
        axL.set_xlim(rx0, rx1); axL.set_ylim(ry1, ry0)
        axR.set_xlim(nx0, nx1); axR.set_ylim(ny1, ny0)
        axL.set_ylabel(f"frame {t}\n(abs {frames[t]})", fontsize=8)
        if t == 0:
            axL.set_title("GREZZI (pixel)", fontsize=11)
            axR.set_title(f"NORMALIZZATI ({norm_name})", fontsize=11)

    fig.suptitle(f"Pedone {ped_id} — finestra (TTE={tte}, {T} frame)",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(out_path, dpi=105, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation_root", required=True,
                    help="root PIE con pie_data.py + annotazioni")
    ap.add_argument("--pose_dir", default="data/poses")
    ap.add_argument("--split", default=None, choices=["train", "val", "test"],
                    help="se omesso e --ped_id e' dato, viene dedotto dal "
                         "set (prima cifra del ped_id): 1/2/4->train, "
                         "5/6->val, 3->test")
    ap.add_argument("--ped_id", default=None)
    ap.add_argument("--norm", default="hip_reference_seq")
    ap.add_argument("--obs_len", type=int, default=16)
    ap.add_argument("--overlap", type=float, default=0.6)
    ap.add_argument("--anchor_eval", action="store_true", default=True)
    ap.add_argument("--endpoint_step", type=int, default=6)
    ap.add_argument("--max_windows", type=int, default=8,
                    help="limite di finestre da disegnare per pedone")
    ap.add_argument("--out", default="out_windows")
    args = ap.parse_args()

    # --- deduci lo split dal ped_id se non specificato -------------------
    # ped_id PIE = '<set>_<video>_<obj>' -> la prima cifra e' il set.
    # Mappa set -> split (come SPLIT_SETS in pie_dataset.py).
    SET_TO_SPLIT = {1: "train", 2: "train", 4: "train",
                    5: "val", 6: "val", 3: "test"}
    if args.split is None:
        if args.ped_id is None:
            raise SystemExit(
                "Serve --split oppure --ped_id (per dedurre lo split dal "
                "set). Es: --ped_id 5_1_1740  (set05 -> val).")
        try:
            set_num = int(str(args.ped_id).split("_")[0])
        except ValueError:
            raise SystemExit(f"ped_id '{args.ped_id}' non nel formato "
                             f"'<set>_<video>_<obj>'; passa --split a mano.")
        if set_num not in SET_TO_SPLIT:
            raise SystemExit(f"set {set_num} (da ped_id {args.ped_id}) non "
                             f"riconosciuto; passa --split a mano.")
        args.split = SET_TO_SPLIT[set_num]
        print(f"[auto] ped_id {args.ped_id} -> set{set_num:02d} -> "
              f"split '{args.split}'")

    # importa qui cosi' lo script parte anche solo per --help senza PIE
    # (il path e' gia' stato sistemato in cima al file)
    from pie_dataset import PIEDataset

    ds = PIEDataset(
        args.annotation_root, split=args.split,
        obs_len=args.obs_len, overlap=args.overlap,
        pose_dir=args.pose_dir, pose_norm=args.norm,
        use_center_channels=True,
        anchor_endpoints=args.anchor_eval, endpoint_step=args.endpoint_step,
    )

    # normalizza un ped_id eventualmente annidato ([[pid]]) a stringa
    def _norm_pid(pid):
        while isinstance(pid, (list, tuple)) and len(pid) > 0:
            pid = pid[0]
        return str(pid)

    # raggruppa i sample per ped_id (normalizzato)
    by_ped = defaultdict(list)
    for s in ds.samples:
        by_ped[_norm_pid(s["ped_id"])].append(s)

    if args.ped_id is None:
        # scegli il pedone con piu' finestre (e stampa la lista)
        ranked = sorted(by_ped.items(), key=lambda kv: -len(kv[1]))
        print("Pedoni disponibili (prime 15 per n. finestre):")
        for pid, samps in ranked[:15]:
            print(f"  {pid}: {len(samps)} finestre")
        args.ped_id = ranked[0][0]
        print(f"[auto] scelto: {args.ped_id}")

    # match del ped_id: prima esatto, poi per prefisso (PIE aggiunge spesso
    # un suffisso, es. '5_1_1740' -> '5_1_1740b'). Se ambiguo, elenca.
    target = str(args.ped_id)
    samples = by_ped.get(target, [])
    if not samples:
        matches = [pid for pid in by_ped if pid.startswith(target)]
        if len(matches) == 1:
            print(f"[match] '{target}' -> '{matches[0]}' (prefisso)")
            target = matches[0]
            samples = by_ped[target]
        elif len(matches) > 1:
            print(f"[ambiguo] '{target}' matcha piu' pedoni: {matches}")
            print("Rilancia con il ped_id esatto tra questi.")
            raise SystemExit(1)

    if not samples:
        avail = sorted(by_ped.keys())
        raise SystemExit(
            f"Nessuna finestra per ped_id={args.ped_id} nello split "
            f"{args.split}.\nPedoni disponibili ({len(avail)}): "
            f"{avail[:20]}{' ...' if len(avail) > 20 else ''}")
    args.ped_id = target
    samples = sorted(samples, key=lambda s: s["w_start"])[:args.max_windows]
    print(f"Pedone {args.ped_id}: {len(by_ped[target])} finestre "
          f"totali, disegno le prime {len(samples)}.")

    outdir = Path(args.out) / str(args.ped_id)
    outdir.mkdir(parents=True, exist_ok=True)

    for wi, s in enumerate(samples):
        raw_xy, mask = raw_keypoints_window(ds._pose_cache, s)
        norm_xy = norm_keypoints_window(ds, s)
        out = outdir / f"win{wi:02d}_start{s['w_start']}_tte{s['tte']}.png"
        plot_window(raw_xy, norm_xy, s["frames"], s["tte"], out,
                    args.ped_id, args.norm)
        print(f"  finestra {wi}: start={s['w_start']} tte={s['tte']} "
              f"cov={100*mask.mean():.0f}%  -> {out}")

    print(f"\nFatto. Figure in: {outdir}")


if __name__ == "__main__":
    main()
