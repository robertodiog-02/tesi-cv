"""
Diagnostica exploding gradients nella GCN di PedGT
==================================================
Verifica EMPIRICAMENTE le due cause citate in letteratura per l'esplosione
del gradiente nelle GCN, applicate al TUO modello:

  (1) SPETTRO dell'adiacenza normalizzata  =  D^{-1/2} A D^{-1/2}
      -> se l'autovalore massimo |lambda|_max <= 1, la propagazione nel
         grafo NON amplifica il segnale. Con self-loop + normalizzazione
         simmetrica (build_adjacency) ci si aspetta |lambda|_max = 1.

  (2) NORMA dei JACOBIANI layer-per-layer della parte spaziale (le 3 GCNConv)
      -> misura ||d output / d input|| di ogni layer sul modello REALE.
         Se il prodotto cresce esponenzialmente > 1, c'e' amplificazione;
         se resta O(1) o si contrae, non e' quello il problema.
      Calcolata come norma spettrale (max singular value) del Jacobiano
      completo, via autograd, su un input reale o casuale.

Puo' girare:
  - sul modello INIZIALIZZATO (pesi random) -> struttura pura;
  - su un CHECKPOINT allenato (--ckpt) -> stato reale a fine training,
    quando la grad-norm era alta.

Uso:
    # solo struttura + init
    python diagnose_gcn_gradients.py

    # con checkpoint allenato e un batch vero dal dataset
    python diagnose_gcn_gradients.py \
        --ckpt checkpoints/<run>/best_model.pt \
        --annotation_root ~/Desktop/PIE/annotations --pose_dir data/poses

Nota: se non passi il dataset, usa un input casuale con la stessa forma
([B,T,19,5]); lo spettro di A e i Jacobiani per-layer non dipendono molto
dallo specifico input, ma con dati veri il numero e' piu' rappresentativo.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

# path robusto (lo script puo' stare in data/ o in root)
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


# ----------------------------------------------------------------- spettro A
def analyze_adjacency():
    """Autovalori di A normalizzata (con e senza self-loop)."""
    from data.skeleton import build_adjacency
    print("=" * 64)
    print("(1) SPETTRO DELL'ADIACENZA NORMALIZZATA")
    print("=" * 64)
    for self_loops in (True, False):
        A = build_adjacency(self_loops=self_loops, normalize=True)
        # A e' simmetrica normalizzata -> autovalori reali
        eig = np.linalg.eigvalsh(A)
        lam_max = np.abs(eig).max()
        lam_min = eig.min()
        print(f"\nself_loops={self_loops}:")
        print(f"  shape A: {A.shape}")
        print(f"  |lambda|_max = {lam_max:.4f}   (>1 => amplifica)")
        print(f"  lambda_min   = {lam_min:.4f}")
        print(f"  primi 5 |lambda| ord.: "
              f"{np.sort(np.abs(eig))[::-1][:5].round(4).tolist()}")
        if lam_max <= 1.0 + 1e-4:
            print(f"  -> OK: |lambda|_max <= 1, il grafo NON amplifica il segnale.")
        else:
            print(f"  -> ATTENZIONE: |lambda|_max > 1, possibile amplificazione.")
    print()


# ------------------------------------------------------- jacobiani per-layer
def _spectral_norm_jacobian(func, x, n_iter=None):
    """
    Norma spettrale (max singular value) del Jacobiano di func in x.
    Usa il Jacobiano esplicito via autograd (dimensioni piccole: 19*C nodi).
    """
    x = x.clone().detach().requires_grad_(True)
    y = func(x)
    y_flat = y.reshape(-1)
    n_out = y_flat.numel()
    n_in = x.numel()
    J = torch.zeros(n_out, n_in)
    for i in range(n_out):
        g = torch.autograd.grad(y_flat[i], x, retain_graph=True,
                                create_graph=False)[0]
        J[i] = g.reshape(-1)
    # max singular value
    sv = torch.linalg.svdvals(J)
    return float(sv[0]), J.shape


def analyze_gcn_jacobians(model, sample_nodes):
    """
    Misura la norma spettrale del Jacobiano di ogni GCNConv (gcn1,gcn2,gcn3)
    e il prodotto cumulato, sul modulo spaziale reale del modello.
    sample_nodes: [N, C] feature di UN grafo (un frame), N=19.
    """
    print("=" * 64)
    print("(2) NORMA SPETTRALE DEI JACOBIANI (parte spaziale, per-layer)")
    print("=" * 64)
    spatial = model.spatial if hasattr(model, "spatial") else \
        model.encoders["pose"].spatial
    ei = spatial.edge_index

    import torch.nn.functional as F

    # CRUCIALE: BatchNorm in eval -> usa running_mean/var (statistiche fisse),
    # non ricalcola sul singolo grafo. In train mode, misurare il Jacobiano su
    # UN grafo solo dà norme gonfiate/instabili (il famoso "126" spurio):
    # BN normalizza per la varianza del mini-batch, che con 1 campione esplode.
    spatial.eval()
    print("  [BatchNorm in eval(): usa running stats, misura affidabile]")

    # --- input projection (se presente): trasforma l'input PRIMA di gcn1 ---
    # Con input_proj attiva, il "vero" input di gcn1 non e' il keypoint grezzo
    # ma l'output della projection. La applichiamo qui, cosi' i Jacobiani dei
    # GCN sono misurati sullo stesso input che ricevono nel forward reale.
    x0 = sample_nodes.clone()
    has_ip = getattr(spatial, "input_proj", None) is not None
    if has_ip:
        with torch.no_grad():
            x0 = spatial.input_proj(x0)          # [19, in_ch] -> [19, proj_dim]
        # misura anche il Jacobiano della projection stessa
        s_ip, sh_ip = _spectral_norm_jacobian(spatial.input_proj,
                                              sample_nodes.clone())
        print(f"  [input_proj attiva: {sample_nodes.shape[1]} -> "
              f"{x0.shape[1]} canali; ||J_proj|| = {s_ip:.4f}]")

    N, C = x0.shape

    # funzioni per-layer (replicano il forward di PedGTSpatial, un grafo solo)
    def f1(x):
        return F.relu(spatial.bn1(spatial.gcn1(x, ei)))

    def f2(h):
        return F.relu(spatial.bn2(spatial.gcn2(h, ei)))

    def f3(h):
        return F.relu(spatial.gcn3(h, ei))

    model.eval()
    with torch.no_grad():
        h1 = f1(x0)
        h2 = f2(h1)

    s1, sh1 = _spectral_norm_jacobian(f1, x0)
    s2, sh2 = _spectral_norm_jacobian(f2, h1)
    s3, sh3 = _spectral_norm_jacobian(f3, h2)

    print(f"\n  gcn1: ||J|| = {s1:.4f}   J shape {sh1}")
    print(f"  gcn2: ||J|| = {s2:.4f}   J shape {sh2}")
    print(f"  gcn3: ||J|| = {s3:.4f}   J shape {sh3}")
    prod = s1 * s2 * s3
    print(f"\n  prodotto ||J1||*||J2||*||J3|| = {prod:.4f}")
    if prod <= 1.5:
        print(f"  -> La parte spaziale NON amplifica (prodotto ~<=1).")
    elif prod <= 5:
        print(f"  -> Amplificazione moderata (attesa in reti sane).")
    else:
        print(f"  -> Amplificazione forte (Jacobiano locale; NON implica "
              f"gradienti esplosivi: vedi grad-norm reale sotto).")

    # --- norma dei PESI per layer (per collegare Jacobiani a overfitting) ---
    print("\n  Norma dei pesi (||W||_2 spettrale) per layer:")
    def wnorm(mod, attr="lin"):
        w = getattr(getattr(mod, attr), "weight", None) if attr else mod.weight
        return float(torch.linalg.matrix_norm(w, ord=2)) if w is not None else float("nan")
    try:
        print(f"    gcn1.lin: {wnorm(spatial.gcn1):.4f}")
        print(f"    gcn2.lin: {wnorm(spatial.gcn2):.4f}")
        print(f"    gcn3.lin: {wnorm(spatial.gcn3):.4f}")
        if has_ip:
            # prima Linear della input_proj
            for m in spatial.input_proj:
                if isinstance(m, torch.nn.Linear):
                    print(f"    input_proj[0]: {float(torch.linalg.matrix_norm(m.weight, ord=2)):.4f}")
                    break
    except Exception as e:
        print(f"    (norma pesi non calcolata: {e})")
    print()
    return prod


# ----------------------------------------------- gradiente end-to-end (bonus)
def analyze_end_to_end_grad(model, batch, device="cpu"):
    """
    Grad-norm totale su un batch reale (come nel training) e ripartizione
    per gruppo di parametri (spatial vs transformer vs classifier).
    """
    print("=" * 64)
    print("(3) GRAD-NORM END-TO-END SU UN BATCH (ripartizione per modulo)")
    print("=" * 64)
    model.train()
    crit = torch.nn.CrossEntropyLoss()
    logits = model(keypoints=batch["keypoints"].to(device),
                   kinematics=batch.get("kinematics"),
                   crop_feat=batch.get("crop_feat"),
                   ego_speed=batch.get("ego_speed"))
    loss = crit(logits, batch["label"].to(device))
    model.zero_grad()
    loss.backward()

    groups = {"spatial(GCN)": [], "transformer": [], "classifier": [], "altro": []}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.norm().item()
        if "spatial" in name:
            groups["spatial(GCN)"].append(g)
        elif "transformer" in name or "temporal" in name:
            groups["transformer"].append(g)
        elif "classifier" in name:
            groups["classifier"].append(g)
        else:
            groups["altro"].append(g)

    total = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            total += p.grad.norm().item() ** 2
    total = total ** 0.5
    print(f"\n  loss = {loss.item():.4f}")
    print(f"  grad-norm TOTALE = {total:.4f}")
    for k, v in groups.items():
        if v:
            gk = (sum(x ** 2 for x in v)) ** 0.5
            print(f"    {k:16s}: ||grad|| = {gk:.4f}  ({len(v)} tensori)")
    print()


def _remap_old_checkpoint(state):
    """
    Rimappa i nomi dei parametri dal vecchio pedgt.py (struttura piatta) al
    nuovo (multi-stream, annidato). Serve per caricare checkpoint allenati
    PRIMA del refactoring multi-stream.

    Vecchio -> Nuovo:
      spatial.*         -> encoders.pose.spatial.*
      transformer.*     -> temporal.transformer.*
      pos_embedding     -> temporal.pos_embedding
      speed_proj.*      -> (rimosso: use_ego_speed non esiste piu') -> scartato
      classifier.*      -> classifier.*   (invariato)

    Ritorna (nuovo_state, report).
    """
    new_state = {}
    dropped = []
    remapped = 0
    for k, v in state.items():
        if k.startswith("speed_proj"):
            dropped.append(k)                       # ego injection rimossa
            continue
        nk = k
        if k.startswith("spatial."):
            nk = "encoders.pose." + k
            remapped += 1
        elif k.startswith("transformer."):
            nk = "temporal." + k
            remapped += 1
        elif k == "pos_embedding":
            nk = "temporal.pos_embedding"
            remapped += 1
        new_state[nk] = v
    return new_state, {"remapped": remapped, "dropped": dropped}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="checkpoint .pt allenato")
    ap.add_argument("--annotation_root", default=None)
    ap.add_argument("--pose_dir", default="data/poses")
    ap.add_argument("--obs_len", type=int, default=16)
    ap.add_argument("--in_channels", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    # (1) struttura del grafo: sempre disponibile, non serve nulla
    analyze_adjacency()

    from models.pedgt import PedGT

    # --- carica il checkpoint PRIMA, per leggerne la config -------------
    ck = None
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu")

    # --- costruisci il modello con la STESSA config del checkpoint -------
    # train.py salva "config": cfg dentro il checkpoint. Se c'e', ricostruiamo
    # il modello IDENTICO a quello allenato (stessi stream/fusione) -> i nomi
    # dei parametri combaciano e il load e' pulito, senza remap.
    def _streams_from_cfg(cfg):
        """Replica la logica di train._build_streams_cfg (versione ridotta)."""
        s = cfg.get("streams", None)
        m = cfg.get("model", {})
        if s is None:
            return ({"pose": {"enabled": True, "use_center_channels": True},
                     "fusion": "concat_tokens"}, m)
        pose_s = s.get("pose", {}) or {}
        kin_s = s.get("kinematics", {}) or {}
        crop_s = s.get("crop", {}) or {}
        out = {
            "pose": {"enabled": pose_s.get("enabled", True),
                     "use_center_channels": pose_s.get("use_center_channels", True),
                     "input_proj": pose_s.get("input_proj", None),
                     "input_proj_dropout": pose_s.get("input_proj_dropout", 0.0)},
            "kinematics": {"enabled": kin_s.get("enabled", False), "in_dim": 5,
                           "hidden": kin_s.get("hidden", m.get("d_model", 128)),
                           "dropout": kin_s.get("dropout", 0.1)},
            "crop": {"enabled": crop_s.get("enabled", False),
                     "in_dim": crop_s.get("in_dim", 768),
                     "hidden": crop_s.get("hidden", m.get("d_model", 128)),
                     "dropout": crop_s.get("dropout", 0.1)},
            "fusion": s.get("fusion", "concat_tokens"),
        }
        return out, m

    if ck is not None and isinstance(ck, dict) and "config" in ck:
        cfg = ck["config"]
        streams_cfg, mcfg = _streams_from_cfg(cfg)
        obs_len = cfg.get("data", {}).get("obs_len", args.obs_len)
        model = PedGT(
            in_channels=mcfg.get("in_channels", 5),
            d_model=mcfg.get("d_model", 128),
            gcn_hidden=mcfg.get("gcn_hidden", 64),
            spatial_out=mcfg.get("spatial_out", 64),
            n_heads=mcfg.get("n_heads", 4),
            n_layers=mcfg.get("n_layers", 2),
            dim_feedforward=mcfg.get("dim_feedforward", 256),
            dropout=mcfg.get("dropout", 0.7),
            encoder_dropout=mcfg.get("encoder_dropout", None),
            obs_len=obs_len, causal=mcfg.get("causal", False),
            streams_cfg=streams_cfg,
        )
        print(f"[info] modello ricostruito dalla config del checkpoint: "
              f"stream attivi={model.active}, fusion={model.fusion}")
        args.obs_len = obs_len
    else:
        # nessuna config nel checkpoint: modello sola-pose di default
        streams_cfg = {"pose": {"enabled": True,
                                "use_center_channels": args.in_channels == 5},
                       "fusion": "concat_tokens"}
        model = PedGT(in_channels=args.in_channels, obs_len=args.obs_len,
                      streams_cfg=streams_cfg)

    tag = "INIZIALIZZATO (pesi random)"
    if ck is not None:
        state = ck.get("model_state", ck)
        # tenta il load diretto; se molti mismatch, prova il remap old->new
        try:
            missing, unexpected = model.load_state_dict(state, strict=False)
        except RuntimeError as e:
            # size mismatch: la config ricostruita non combacia col checkpoint.
            # Causa tipica: una chiave (es. input_proj) non e' stata letta dalla
            # config del checkpoint. Messaggio chiaro invece di crash secco.
            print("\n[ERRORE] size mismatch nel caricamento del checkpoint.")
            print("  Significa che il modello ricostruito NON ha la stessa")
            print("  architettura del checkpoint. Controlla che la config nel")
            print("  checkpoint (streams.pose.input_proj, use_center_channels,")
            print("  d_model, ...) sia letta correttamente.\n  Dettaglio:")
            print(f"  {e}")
            raise SystemExit(1)
        if len(missing) > 2 or len(unexpected) > 2:
            print(f"[info] load diretto: missing={len(missing)} "
                  f"unexpected={len(unexpected)} -> provo il remap old->new")
            remapped_state, rep = _remap_old_checkpoint(state)
            missing, unexpected = model.load_state_dict(remapped_state,
                                                        strict=False)
            print(f"[info] remap: {rep['remapped']} chiavi rinominate, "
                  f"{len(rep['dropped'])} scartate (speed_proj)")
        tag = f"CHECKPOINT {args.ckpt}"
        # cosa resta non caricato (ignorando i buffer non-peso)
        real_missing = [m for m in missing
                        if not m.endswith("num_batches_tracked")
                        and "edge_index" not in m and m != "stream_emb"]
        if real_missing or unexpected:
            print(f"[warn] dopo il load: missing={len(real_missing)} "
                  f"unexpected={len(unexpected)}")
            if real_missing:
                print(f"        missing es.: {real_missing[:4]}")
            if unexpected:
                print(f"        unexpected es.: {list(unexpected)[:4]}")
        else:
            print(f"[OK] checkpoint caricato COMPLETAMENTE "
                  f"(pesi allenati attivi).")
    print(f"\n>>> Modello analizzato: {tag}\n")

    # input: reale se ho il dataset, altrimenti casuale
    batch = None
    if args.annotation_root:
        try:
            from data.pie_dataset import PIEDataset
            from torch.utils.data import DataLoader
            ds = PIEDataset(args.annotation_root, split="val",
                            obs_len=args.obs_len, overlap=0.6,
                            pose_dir=args.pose_dir, pose_norm="hip_reference_seq",
                            use_center_channels=(args.in_channels == 5),
                            anchor_endpoints=True, endpoint_step=6)

            def coll(b):
                return {"keypoints": torch.stack([x["keypoints"] for x in b]),
                        "ego_speed": torch.stack([x["ego_speed"] for x in b]),
                        "label": torch.stack([x["label"] for x in b])}
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            collate_fn=coll)
            batch = next(iter(dl))
            print(f"[info] batch reale dal dataset: "
                  f"keypoints {tuple(batch['keypoints'].shape)}")
        except Exception as e:
            print(f"[warn] dataset non disponibile ({e}); uso input casuale.")

    if batch is None:
        B, T = args.batch_size, args.obs_len
        batch = {"keypoints": torch.randn(B, T, 19, args.in_channels),
                 "ego_speed": torch.randn(B, T, 1),
                 "label": torch.randint(0, 2, (B,))}
        print(f"[info] input CASUALE: keypoints {tuple(batch['keypoints'].shape)}")

    # (2) jacobiani per-layer su UN grafo (un frame del primo campione)
    sample_nodes = batch["keypoints"][0, 0]      # [19, C]
    analyze_gcn_jacobians(model, sample_nodes)

    # (3) grad-norm end-to-end ripartita
    analyze_end_to_end_grad(model, batch)

    print("Interpretazione rapida:")
    print("  - Se |lambda|_max(A)=1 e prodotto Jacobiani ~O(1): NON e'")
    print("    exploding gradient strutturale. Una grad-norm che sale piano")
    print("    (3->8) senza NaN e' overfitting/confidenza crescente, non")
    print("    instabilita' numerica.")


if __name__ == "__main__":
    main()
