"""
Plot della norma dei gradienti LAYER-PER-LAYER dentro ogni RAMO
==============================================================
Legge history.json (con la chiave 'train_grad_per_layer' salvata da train.py)
e produce un pannello per ogni ramo (pose, kinematics, crop, temporal, head),
dove ogni curva e' un layer di quel ramo, piu' la curva TOTALE del ramo
(= sqrt(somma dei quadrati dei layer), tratteggiata e in nero).

Serve a diagnosticare lo squilibrio tra rami nel multi-stream: se attivi
pose+kinematics ma la kinematics riceve gradienti ~0, i suoi pesi non si
aggiornano e il ramo e' "morto" (spiega perche' le metriche non cambiano).

Uso:
    python plot_grad_per_layer.py --history checkpoints/<run>/history.json
    python plot_grad_per_layer.py --history .../history.json --out gpl.png
    # confronto totali-per-ramo di piu' run:
    python plot_grad_per_layer.py --history a/history.json --history b/history.json \
        --labels runA runB --totals_only
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    with open(path) as f:
        return json.load(f)


def branch_layer_series(history, grad_key="train_grad_per_layer"):
    """
    Da history (lista di epoche) a struttura:
      {ramo: {layer: np.array per epoca}}, piu' epochs.
    Gestisce layer che appaiono/spariscono (riempie con nan).
    """
    epochs = [row.get("epoch", i + 1) for i, row in enumerate(history)]
    # raccogli tutti i (ramo, layer) esistenti
    branches = {}
    for row in history:
        gpl = row.get(grad_key, {}) or {}
        for b, layers in gpl.items():
            for l in layers:
                branches.setdefault(b, set()).add(l)
    # costruisci le serie
    data = {}
    for b, layers in branches.items():
        data[b] = {}
        for l in sorted(layers):
            serie = []
            for row in history:
                v = (row.get(grad_key, {}) or {}).get(b, {}).get(l, None)
                serie.append(np.nan if v is None else float(v))
            data[b][l] = np.array(serie)
    return np.array(epochs), data


def branch_total(layers_dict):
    """Totale ramo per epoca = sqrt(sum quadrati layer), nan-safe."""
    mats = np.vstack(list(layers_dict.values()))       # [n_layer, n_epoch]
    return np.sqrt(np.nansum(mats ** 2, axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", action="append", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default="grad_per_layer.png")
    ap.add_argument("--totals_only", action="store_true",
                    help="un solo pannello: solo i TOTALI per ramo (confronto)")
    args = ap.parse_args()

    runs = []
    for i, h in enumerate(args.history):
        hist = load(h)
        label = (args.labels[i] if args.labels and i < len(args.labels)
                 else Path(h).parent.name)
        ep, data = branch_layer_series(hist)
        runs.append((label, ep, data))

    # ---------- modalita' 1: solo i totali per ramo (confronto tra run) ----
    if args.totals_only:
        fig, ax = plt.subplots(figsize=(9, 6))
        for (label, ep, data) in runs:
            for b, layers in data.items():
                tot = branch_total(layers)
                ax.plot(ep, tot, "-o", ms=3, label=f"{label}:{b}")
        ax.set_title("Norma gradiente TOTALE per ramo")
        ax.set_xlabel("epoca"); ax.set_ylabel("||grad|| del ramo")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(args.out, dpi=130, bbox_inches="tight")
        print("salvato:", args.out)
        return

    # ---------- modalita' 2: un pannello per ramo, layer-per-layer ---------
    # usa il PRIMO run (per confronto multi-run usa --totals_only)
    label, ep, data = runs[0]
    branches = list(data.keys())
    n = len(branches)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.2 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, b in zip(axes, branches):
        layers = data[b]
        for l, serie in sorted(layers.items()):
            ax.plot(ep, serie, "-o", ms=2.5, label=l)
        # totale del ramo (tratteggiato nero)
        tot = branch_total(layers)
        ax.plot(ep, tot, "--", color="black", lw=2, label="TOTALE ramo")
        ax.set_title(f"Ramo: {b}")
        ax.set_xlabel("epoca"); ax.set_ylabel("||grad||")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Norma gradienti layer-per-layer per ramo — {label}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print("salvato:", args.out)


if __name__ == "__main__":
    main()
