"""
Training + Evaluation — Baseline GRU geometry-only (bbox + delta)
=================================================================
File unico: training, validation ed evaluation finale.

Uso:
    python train.py --config configs/baseline_base.yaml

Per ogni epoca calcola su TRAIN e VAL:
    loss, accuracy, f1, auc, precision, recall

Al termine valuta il best model su train/val/test e salva:
    - history.json   : tutte le metriche per epoca (train + val)
    - test_results.json
    - predictions.json : label/pred/prob per train, val, test (per le
                         confusion matrix e i plot)
Poi basta lanciare:
    python plot_metrics.py --exp_dir checkpoints/<nome_esperimento>
"""

import os
import json
import time
import random
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    recall_score, precision_score,
)

from data.pie_dataset import PIEDataset
from models.models import BaselineGRU
from models.pedgt import PedGT as PedGT_cls
from plot_metrics import plot_metric_curves, plot_confusion_matrices


# ─── Utility ──────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: Apple MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: CUDA — {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Device: CPU")
    return device


def load_config(config_path: str) -> Dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def collate_fn(batch):
    """Collate: bbox, bbox_delta, ego_speed, label (+ stream tensors, meta)."""
    keys_tensor = ["bbox", "bbox_delta", "ego_speed", "label"]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys_tensor}
    # stream opzionali: presenti solo se lo stream e' attivo nel dataset
    for opt in ("keypoints", "kinematics", "crop_feat"):
        if opt in batch[0]:
            out[opt] = torch.stack([b[opt] for b in batch])
    out["ped_id"] = [b["ped_id"] for b in batch]
    return out


def build_model(cfg: Dict) -> nn.Module:
    name = cfg["model"]["name"]
    if name == "BaselineGRU":
        return BaselineGRU(
            hidden_dim=cfg["model"]["hidden_dim"],
            num_layers=cfg["model"]["num_layers"],
            dropout=cfg["model"]["dropout"],
            use_bbox_delta=cfg["model"].get("use_bbox_delta", True),
            use_ego_speed=cfg["model"].get("use_ego_speed", False),
        )
    if name == "PedGT":
        from models.pedgt import PedGT
        m = cfg["model"]
        streams_cfg = _build_streams_cfg(cfg)
        return PedGT(
            in_channels=m.get("in_channels", 5),
            d_model=m.get("d_model", 128),
            gcn_hidden=m.get("gcn_hidden", 64),
            spatial_out=m.get("spatial_out", 64),
            n_heads=m.get("n_heads", 4),
            n_layers=m.get("n_layers", 2),
            dim_feedforward=m.get("dim_feedforward", 256),
            dropout=m.get("dropout", 0.7),
            encoder_dropout=m.get("encoder_dropout", None),
            obs_len=cfg["data"]["obs_len"],
            causal=m.get("causal", False),
            streams_cfg=streams_cfg,
        )
    raise ValueError(f"Modello non supportato: {name}.")


def _build_streams_cfg(cfg: Dict) -> Dict:
    """
    Traduce la sezione `streams` del config nel dict atteso da PedGT.

    Default (se `streams` assente): solo pose, center channels ON,
    fusion=concat_tokens -> comportamento identico al PedGT originale.
    Le dimensioni di input (kinematics/crop) sono ricavate dal config data:
      - kinematics in_dim = 4 (bbox) + 1 (ego) = 5
      - crop       in_dim = 768 (ConvNeXt)
    """
    s = cfg.get("streams", None)
    if s is None:
        return {"pose": {"enabled": True, "use_center_channels": True},
                "fusion": "concat_tokens"}

    pose_s = s.get("pose", {}) or {}
    kin_s  = s.get("kinematics", {}) or {}
    crop_s = s.get("crop", {}) or {}

    out = {
        "pose": {
            "enabled": pose_s.get("enabled", True),
            "use_center_channels": pose_s.get("use_center_channels", True),
        },
        "kinematics": {
            "enabled": kin_s.get("enabled", False),
            "in_dim": 5,   # 4 bbox + ego
            "hidden": kin_s.get("hidden", cfg["model"].get("d_model", 128)),
            "dropout": kin_s.get("dropout", 0.1),
        },
        "crop": {
            "enabled": crop_s.get("enabled", False),
            "in_dim": crop_s.get("in_dim", 768),
            "hidden": crop_s.get("hidden", cfg["model"].get("d_model", 128)),
            "dropout": crop_s.get("dropout", 0.1),
        },
        "fusion": s.get("fusion", "concat_tokens"),
    }
    return out


# ─── Metriche ─────────────────────────────────────────────────────────────────

def compute_metrics(labels: np.ndarray, preds: np.ndarray,
                    probs: np.ndarray) -> Dict[str, float]:
    """Calcola tutte le metriche a partire da label/pred/prob."""
    acc       = accuracy_score(labels, preds)
    f1        = f1_score(labels, preds, pos_label=1, zero_division=0)
    recall    = recall_score(labels, preds, pos_label=1, zero_division=0)
    precision = precision_score(labels, preds, pos_label=1, zero_division=0)
    if len(np.unique(labels)) > 1:
        auc = roc_auc_score(labels, probs)
    else:
        auc = 0.0
    return {
        "acc":       float(acc),
        "f1":        float(f1),
        "auc":       float(auc),
        "precision": float(precision),
        "recall":    float(recall),
    }


# ─── Train / Eval di una epoca ────────────────────────────────────────────────

def run_epoch(model, loader, criterion, device, optimizer=None,
              desc="train", grad_clip=1.0) -> Tuple[Dict[str, float], Dict[str, list]]:
    """
    Esegue una passata completa sul loader.
    Se optimizer è passato -> training (backward + step), altrimenti eval.

    Returns:
        metrics : dict con loss, acc, f1, auc, precision, recall,
                  grad_norm_mean, grad_clip_frac (solo in training)
        outputs : dict con liste labels/preds/probs (per confusion matrix)
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total = 0.0, 0
    all_labels, all_preds, all_probs = [], [], []
    grad_norm_sum = 0.0
    grad_norm_count = 0
    grad_clipped_count = 0

    pbar = tqdm(loader, desc=desc, leave=False, ncols=100)
    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()

    with grad_ctx:
        for batch in pbar:
            bbox       = batch["bbox"].to(device)
            bbox_delta = batch["bbox_delta"].to(device)
            ego_speed  = batch["ego_speed"].to(device)
            labels     = batch["label"].to(device)
            keypoints  = batch["keypoints"].to(device) if "keypoints" in batch else None
            kinematics = batch["kinematics"].to(device) if "kinematics" in batch else None
            crop_feat  = batch["crop_feat"].to(device) if "crop_feat" in batch else None

            if is_train:
                optimizer.zero_grad()

            if isinstance(model, PedGT_cls):
                # modello multi-stream: passa tutti gli stream disponibili
                # per keyword; il modello usa solo quelli attivi.
                logits = model(keypoints=keypoints, kinematics=kinematics,
                               crop_feat=crop_feat, ego_speed=ego_speed)
            elif keypoints is not None:
                logits = model(keypoints, ego_speed)
            else:
                logits = model(bbox, bbox_delta, ego_speed)
            loss   = criterion(logits, labels)

            if is_train:
                loss.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip)
                optimizer.step()
                grad_norm_sum += float(total_norm)
                grad_norm_count += 1
                if float(total_norm) > grad_clip:
                    grad_clipped_count += 1

            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = logits.argmax(dim=-1)

            total_loss += loss.item() * len(labels)
            total      += len(labels)
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_probs.extend(probs.detach().cpu().numpy().tolist())

            running_acc = np.mean(np.array(all_preds) == np.array(all_labels))
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{running_acc:.3f}")

    labels_np = np.array(all_labels)
    preds_np  = np.array(all_preds)
    probs_np  = np.array(all_probs)

    metrics = compute_metrics(labels_np, preds_np, probs_np)
    metrics["loss"] = total_loss / total
    if is_train and grad_norm_count > 0:
        metrics["grad_norm_mean"] = grad_norm_sum / grad_norm_count
        metrics["grad_clip_frac"] = grad_clipped_count / grad_norm_count

    outputs = {
        "labels": all_labels,
        "preds":  all_preds,
        "probs":  all_probs,
    }
    return metrics, outputs


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--test_as_val", action="store_true",
                        help="DIAGNOSTICO: usa il TEST set come validation per "
                             "plottare le curve epoca-per-epoca sul test. "
                             "NON usare per risultati ufficiali (data leakage).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    device = get_device()

    out_name = cfg["experiment"]["name"]
    if args.test_as_val:
        out_name += "_TESTASVAL"
    out_dir = Path(cfg["output"]["checkpoint_dir"]) / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    print(f"\nEsperimento: {cfg['experiment']['name']}")
    print(f"Config: {args.config}")
    print(f"Output: {out_dir}\n")

    # -- Dataset --------------------------------------------------------------
    data_cfg  = cfg["data"]
    train_cfg = cfg["training"]

    pose_dir  = data_cfg.get("pose_dir", None)
    pose_norm = data_cfg.get("pose_norm", "reference_point")

    # -- Stream multi-input (sezione `streams` del config) -------------------
    streams_cfg   = cfg.get("streams", {}) or {}
    pose_s        = streams_cfg.get("pose", {}) or {}
    kin_s         = streams_cfg.get("kinematics", {}) or {}
    crop_s        = streams_cfg.get("crop", {}) or {}
    # pose attiva -> serve pose_dir; se lo stream pose e' disabilitato,
    # non carichiamo le pose (pose_dir_eff = None).
    pose_enabled  = pose_s.get("enabled", True)
    pose_dir_eff  = pose_dir if pose_enabled else None
    use_center_ch = pose_s.get("use_center_channels", True)
    use_kinematics = kin_s.get("enabled", False)
    bbox_format    = kin_s.get("bbox_format", "xyxy")
    use_crop       = crop_s.get("enabled", False)
    crop_dir       = data_cfg.get("crop_dir", None) if use_crop else None
    if use_crop and crop_dir is None:
        raise ValueError("streams.crop.enabled=true ma data.crop_dir non "
                         "impostato nel config.")

    # kwargs comuni ai 3 dataset per gli stream
    stream_kwargs = dict(
        pose_dir=pose_dir_eff, pose_norm=pose_norm,
        use_center_channels=use_center_ch,
        use_kinematics=use_kinematics, bbox_format=bbox_format,
        crop_dir=crop_dir,
    )

    overlap        = data_cfg.get("overlap", 0.6)           # train (augmentation)
    overlap_eval   = data_cfg.get("overlap_eval", 0.6)      # val/test (se non anchored)
    anchor_eval    = data_cfg.get("anchor_eval", True)      # val/test: end-point ancorati
    endpoint_step  = data_cfg.get("endpoint_step", 6)       # step assoluto del benchmark

    train_ds = PIEDataset(data_cfg["annotation_root"], split="train",
                          obs_len=data_cfg["obs_len"], overlap=overlap,
                          **stream_kwargs)
    val_split = "test" if args.test_as_val else "val"
    if args.test_as_val:
        print("\n" + "!" * 60)
        print("!! MODALITÀ DIAGNOSTICA: test_as_val ATTIVA")
        print("!! Il VALIDATION usato per le curve è il TEST set.")
        print("!! Risultati NON validi metodologicamente (solo ispezione).")
        print("!" * 60)
    val_ds   = PIEDataset(data_cfg["annotation_root"], split=val_split,
                          obs_len=data_cfg["obs_len"], overlap=overlap_eval,
                          anchor_endpoints=anchor_eval, endpoint_step=endpoint_step,
                          **stream_kwargs)

    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"],
                              shuffle=True, num_workers=0, collate_fn=collate_fn)
    # train_loader per la valutazione (no shuffle, per metriche stabili)
    train_eval_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"],
                                   shuffle=False, num_workers=0, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=train_cfg["batch_size"],
                              shuffle=False, num_workers=0, collate_fn=collate_fn)

    # -- Modello --------------------------------------------------------------
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Feature attive: bbox=True, "
          f"bbox_delta={getattr(model, 'use_bbox_delta', '?')}, "
          f"ego_speed={getattr(model, 'use_ego_speed', '?')} "
          f"-> input_dim={getattr(model, 'input_dim', '?')}")
    print(f"Parametri: {n_params:,}\n")

    # -- Loss / optim / scheduler --------------------------------------------
    class_weights = train_ds.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"],
                                  weight_decay=train_cfg["weight_decay"])

    sched_name = train_cfg.get("scheduler", "cosine").lower()
    if sched_name in ("plateau", "reduce_on_plateau", "reducelronplateau"):
        # riduce lr quando val_f1 smette di migliorare (mode='max')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=train_cfg.get("plateau_factor", 0.5),
            patience=train_cfg.get("plateau_patience", 3),
            min_lr=1e-6)
        sched_step_on_metric = True
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg["epochs"], eta_min=1e-5)
        sched_step_on_metric = False
    print(f"Scheduler: {sched_name}  |  grad_clip: {train_cfg.get('grad_clip', 1.0)}")
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    # -- Resume ---------------------------------------------------------------
    start_epoch = 1
    best_val_f1 = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt["val_metrics"]["f1"]
        print(f"Ripreso da epoch {start_epoch} (best_f1={best_val_f1:.4f})")

    # -- Training loop --------------------------------------------------------
    patience_counter = 0
    history = []
    epochs = train_cfg["epochs"]

    print("=== Training ===")
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        train_m, _ = run_epoch(model, train_loader, criterion, device,
                               optimizer=optimizer, grad_clip=grad_clip,
                               desc=f"Epoch {epoch}/{epochs} [train]")
        val_m, _   = run_epoch(model, val_loader, criterion, device,
                               optimizer=None,
                               desc=f"Epoch {epoch}/{epochs} [val]  ")
        if sched_step_on_metric:
            scheduler.step(val_m["f1"])        # plateau: si basa su val_f1
        else:
            scheduler.step()                   # cosine: avanza ogni epoca

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        gn = train_m.get("grad_norm_mean", float("nan"))
        gc = train_m.get("grad_clip_frac", float("nan"))
        print(
            f"Epoch {epoch:3d}/{epochs} ({elapsed:.0f}s) lr={lr_now:.2e} "
            f"grad_norm={gn:.3f} clip_frac={gc:.0%}\n"
            f"  TRAIN loss={train_m['loss']:.4f} acc={train_m['acc']:.4f} "
            f"f1={train_m['f1']:.4f} auc={train_m['auc']:.4f} "
            f"P={train_m['precision']:.4f} R={train_m['recall']:.4f}\n"
            f"  VAL   loss={val_m['loss']:.4f} acc={val_m['acc']:.4f} "
            f"f1={val_m['f1']:.4f} auc={val_m['auc']:.4f} "
            f"P={val_m['precision']:.4f} R={val_m['recall']:.4f}"
        )

        history.append({
            "epoch": epoch,
            "val_split": val_split,   # 'val' o 'test' (se test_as_val)
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"val_{k}": v for k, v in val_m.items()},
        })
        # salva history a ogni epoca (per poter plottare anche durante il run)
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Best model (su val f1)
        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_metrics": val_m,
                "config": cfg,
            }, out_dir / "best_model.pt")
            print(f"  ✓ Best model salvato (val_f1={best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= train_cfg["patience"]:
                print(f"\nEarly stopping (patience={train_cfg['patience']})")
                break

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_metrics": val_m,
                "config": cfg,
            }, out_dir / f"epoch_{epoch:03d}.pt")

    # -- Evaluation finale del best model ------------------------------------
    print("\n=== Evaluation finale (best model) ===")
    test_ds = PIEDataset(data_cfg["annotation_root"], split="test",
                         obs_len=data_cfg["obs_len"], overlap=overlap_eval,
                         anchor_endpoints=anchor_eval, endpoint_step=endpoint_step,
                         **stream_kwargs)
    test_loader = DataLoader(test_ds, batch_size=train_cfg["batch_size"],
                             shuffle=False, num_workers=0, collate_fn=collate_fn)

    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    final_metrics = {}
    predictions   = {}
    for split, loader in [("train", train_eval_loader),
                          ("val",   val_loader),
                          ("test",  test_loader)]:
        m, out = run_epoch(model, loader, criterion, device,
                           optimizer=None, desc=f"eval [{split}]")
        final_metrics[split] = m
        predictions[split]   = out
        print(f"\n[{split.upper()}] "
              f"acc={m['acc']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f} "
              f"P={m['precision']:.4f} R={m['recall']:.4f}")

    with open(out_dir / "test_results.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    with open(out_dir / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"\nRisultati salvati in: {out_dir}")

    # -- Plot automatici ------------------------------------------------------
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print("\n=== Generazione grafici ===")
    plot_metric_curves(history, plots_dir)
    plot_confusion_matrices(predictions, plots_dir)
    print(f"Grafici salvati in: {plots_dir}")


if __name__ == "__main__":
    main()