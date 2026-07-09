"""
Verifica dell'ordine dei giunti e della struttura del grafo scheletro
=====================================================================
Produce tre viste per controllare che JOINT_EDGES / JOINT_NAMES / indici
siano coerenti (nessun arco "assurdo" tipo polso-orecchio):

  1. SCHELETRO SCHEMATICO: i 19 giunti disposti in una posa canonica "a T"
     (coordinate fisse plausibili), con archi disegnati e nomi accanto.
     Se un arco collega giunti sbagliati, si vede a colpo d'occhio.
  2. MATRICE DI ADIACENZA come heatmap, con etichette dei giunti sui due assi.
  3. STAMPA testuale di ogni arco con i NOMI (non solo gli indici), per il
     controllo puntuale.

Non richiede dati: usa solo skeleton.py.

Uso:
    python verify_skeleton_graph.py --out out_skel_check
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from data.skeleton import (JOINT_NAMES, JOINT_EDGES, NUM_JOINTS, build_adjacency,
                      NOSE, LEYE, REYE, LEAR, REAR, LSHO, RSHO, LELB, RELB,
                      LWRI, RWRI, LHIP, RHIP, LKNE, RKNE, LANK, RANK, NECK, CHIP)


# Coordinate canoniche (x, y) per una posa frontale "a T", solo per il disegno.
# y cresce verso l'ALTO qui (poi invertiamo l'asse per la convenzione immagine).
# Valori scelti a mano per una figura leggibile; NON influenzano il modello.
CANONICAL_POSE = {
    NOSE:  (0.00,  9.0),
    LEYE:  (0.30,  9.3),  REYE: (-0.30,  9.3),
    LEAR:  (0.60,  9.1),  REAR: (-0.60,  9.1),
    NECK:  (0.00,  8.0),
    LSHO:  (1.20,  8.0),  RSHO: (-1.20,  8.0),
    LELB:  (1.80,  6.5),  RELB: (-1.80,  6.5),
    LWRI:  (2.20,  5.0),  RWRI: (-2.20,  5.0),
    CHIP:  (0.00,  5.0),
    LHIP:  (0.70,  5.0),  RHIP: (-0.70,  5.0),
    LKNE:  (0.80,  2.7),  RKNE: (-0.80,  2.7),
    LANK:  (0.90,  0.3),  RANK: (-0.90,  0.3),
}


def plot_schematic(out_path):
    fig, ax = plt.subplots(figsize=(7, 9))
    xy = np.array([CANONICAL_POSE[i] for i in range(NUM_JOINTS)])

    # archi
    for i, j in JOINT_EDGES:
        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]],
                "-", color="tab:blue", lw=2, zorder=1)
    # giunti
    ax.scatter(xy[:, 0], xy[:, 1], s=90, color="tab:orange",
               edgecolors="black", zorder=2)
    # snodi derivati evidenziati
    ax.scatter(xy[[NECK, CHIP], 0], xy[[NECK, CHIP], 1], s=180,
               facecolors="none", edgecolors="red", linewidths=2, zorder=3)
    # etichette: indice + nome
    for i, (x, y) in enumerate(xy):
        ax.annotate(f"{i}:{JOINT_NAMES[i]}", (x, y),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_aspect("equal")
    ax.set_title("Scheletro schematico — verifica archi e ordine giunti\n"
                 "(rosso = giunti derivati Neck/CHip)", fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.margins(0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_adjacency(out_path, normalize=False):
    A = build_adjacency(self_loops=True, normalize=normalize)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(A, cmap="viridis")
    ax.set_xticks(range(NUM_JOINTS)); ax.set_yticks(range(NUM_JOINTS))
    ax.set_xticklabels(JOINT_NAMES, rotation=90, fontsize=7)
    ax.set_yticklabels(JOINT_NAMES, fontsize=7)
    ax.set_title(f"Matrice di adiacenza (self-loops, "
                 f"{'normalizzata' if normalize else 'binaria'})", fontsize=11)
    # segna i valori non-zero
    for i in range(NUM_JOINTS):
        for j in range(NUM_JOINTS):
            if A[i, j] > 0:
                ax.text(j, i, "•", ha="center", va="center",
                        color="white", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def print_edges():
    print("=" * 60)
    print("ARCHI DEL GRAFO (indice -> nome), per controllo manuale")
    print("=" * 60)
    for i, j in JOINT_EDGES:
        print(f"  {i:2d} {JOINT_NAMES[i]:15s} -- {j:2d} {JOINT_NAMES[j]}")
    print(f"\n  Totale archi (non orientati): {len(JOINT_EDGES)}")

    # sanity check: gradi e simmetria
    A = build_adjacency(self_loops=False, normalize=False)
    deg = A.sum(1).astype(int)
    print("\nGRADO DI OGNI GIUNTO (senza self-loop):")
    for i in range(NUM_JOINTS):
        flag = "  <-- ISOLATO!" if deg[i] == 0 else ""
        print(f"  {i:2d} {JOINT_NAMES[i]:15s} grado={deg[i]}{flag}")
    print(f"\n  simmetrica? {np.allclose(A, A.T)}")
    print(f"  nodi isolati: {int((deg == 0).sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out_skel_check")
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    print_edges()
    p1 = plot_schematic(outdir / "skeleton_schematic.png")
    p2 = plot_adjacency(outdir / "adjacency_binary.png", normalize=False)
    p3 = plot_adjacency(outdir / "adjacency_normalized.png", normalize=True)
    print("\nFigure salvate:")
    for p in (p1, p2, p3):
        print("  ", p)


if __name__ == "__main__":
    main()
