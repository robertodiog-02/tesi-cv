"""
Plot della norma dei gradienti (e metriche correlate) da history.json
=====================================================================
Legge uno o piu' history.json prodotti da train.py e plotta l'andamento
per epoca di:
  - grad_norm_mean  (la norma media dei gradienti, cio' che ti interessa)
  - grad_clip_frac  (frazione di batch in cui e' scattato il clipping)
  - loss e F1 (train vs val) come contesto: la grad-norm che sale mentre il
    val peggiora e' la firma dell'overfitting, non di instabilita'.

Struttura attesa di history.json (come la salva train.py): lista di dict,
uno per epoca, con chiavi 'epoch', 'train_grad_norm_mean',
'train_grad_clip_frac', 'train_loss', 'val_loss', 'train_f1', 'val_f1', ...

Uso:
    # un solo run
    python plot_grad_history.py --history checkpoints/<run>/history.json

    # confronto di piu' run (es. baseline vs input_proj)
    python plot_grad_history.py \
        --history checkpoints/baseline/history.json \
        --history checkpoints/layerDense/history.json \
        --labels baseline dense_before_gcn \
        --out grad_history.png

    # solo la grad-norm, senza i pannelli di contesto
    python plot_grad_history.py --history .../history.json --only_grad
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_history(path):
    with open(path) as f:
        data = json.load(f)
    # lista di dict -> dict di liste, robusto a chiavi mancanti
    keys = set()
    for row in data:
        keys.update(row.keys())
    cols = {k: [row.get(k, None) for row in data] for k in keys}
    # epoch: se manca, usa l'indice
    if "epoch" not in cols or any(e is None for e in cols["epoch"]):
        cols["epoch"] = list(range(1, len(data) + 1))
    return cols


def _series(cols, key):
    """Estrae una serie numerica, mettendo nan dove manca."""
    if key not in cols:
        return None
    vals = [np.nan if v is None else float(v) for v in cols[key]]
    if all(np.isnan(v) for v in vals):
        return None
    return np.array(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", action="append", required=True,
                    help="path a history.json (ripetibile per piu' run)")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="etichette per i run (stesso ordine di --history)")
    ap.add_argument("--out", default="grad_history.png")
    ap.add_argument("--only_grad", action="store_true",
                    help="plotta SOLO la grad-norm (un pannello)")
    args = ap.parse_args()

    runs = []
    for i, h in enumerate(args.history):
        cols = load_history(h)
        label = (args.labels[i] if args.labels and i < len(args.labels)
                 else Path(h).parent.name)
        runs.append((label, cols))

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(runs), 1)))

    if args.only_grad:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax2 = ax.twinx()   # asse destro per il clip_frac
        for (label, cols), c in zip(runs, colors):
            ep = np.array(cols["epoch"])
            gn = _series(cols, "train_grad_norm_mean")
            cf = _series(cols, "train_grad_clip_frac")
            if gn is not None:
                ax.plot(ep, gn, "-o", color=c, ms=3,
                        label=f"{label} grad-norm")
            if cf is not None:
                # barre semitrasparenti sul secondo asse per l'attivita' clip
                ax2.bar(ep, 100 * cf, width=0.6, color=c, alpha=0.20,
                        label=f"{label} clip %")
        ax.set_title("Norma dei gradienti (—) e attività del clipping (barre)")
        ax.set_xlabel("epoca")
        ax.set_ylabel("grad_norm_mean")
        ax2.set_ylabel("clip_frac (%)")
        ax2.set_ylim(bottom=0)
        # legenda combinata dei due assi
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out, dpi=130, bbox_inches="tight")
        print("salvato:", args.out)
        return

    # layout completo: 4 pannelli
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.reshape(-1)

    # --- pannello 1: grad_norm_mean ---
    ax = axes[0]
    for (label, cols), c in zip(runs, colors):
        ep = np.array(cols["epoch"])
        gn = _series(cols, "train_grad_norm_mean")
        if gn is not None:
            ax.plot(ep, gn, "-o", color=c, ms=3, label=label)
    ax.set_title("Norma media dei gradienti per epoca")
    ax.set_xlabel("epoca"); ax.set_ylabel("grad_norm_mean")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    # --- pannello 2: grad_clip_frac ---
    ax = axes[1]
    for (label, cols), c in zip(runs, colors):
        ep = np.array(cols["epoch"])
        cf = _series(cols, "train_grad_clip_frac")
        if cf is not None:
            ax.plot(ep, 100 * cf, "-o", color=c, ms=3, label=label)
    ax.set_title("Frazione di batch clippati (%)")
    ax.set_xlabel("epoca"); ax.set_ylabel("clip_frac (%)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    # --- pannello 3: loss train vs val ---
    ax = axes[2]
    for (label, cols), c in zip(runs, colors):
        ep = np.array(cols["epoch"])
        tr = _series(cols, "train_loss")
        vl = _series(cols, "val_loss")
        if tr is not None:
            ax.plot(ep, tr, "-", color=c, label=f"{label} train")
        if vl is not None:
            ax.plot(ep, vl, "--", color=c, label=f"{label} val")
    ax.set_title("Loss: train (—) vs val (- -)")
    ax.set_xlabel("epoca"); ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # --- pannello 4: F1 train vs val ---
    ax = axes[3]
    for (label, cols), c in zip(runs, colors):
        ep = np.array(cols["epoch"])
        tr = _series(cols, "train_f1")
        vl = _series(cols, "val_f1")
        if tr is not None:
            ax.plot(ep, tr, "-", color=c, label=f"{label} train")
        if vl is not None:
            ax.plot(ep, vl, "--", color=c, label=f"{label} val")
    ax.set_title("F1: train (—) vs val (- -)")
    ax.set_xlabel("epoca"); ax.set_ylabel("F1")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle("History del training — gradienti e metriche", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print("salvato:", args.out)


if __name__ == "__main__":
    main()
