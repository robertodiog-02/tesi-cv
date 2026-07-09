"""
PedGT multi-stream — Skeleton GCN + Kinematics + Crop, fusione configurabile
============================================================================
Estende il PedGT originale (Riaz et al., IEEE IV 2025) a piu' stream di input.

STREAM (attivabili indipendentemente dal config, sezione `streams`):
  - pose       : GCN a 19 giunti (come il paper). in_channels = 5 (x,y,conf,
                 cx,cy) oppure 3 (x,y,conf) se use_center_channels=False.
                 -> sequenza [B,T,d_model].
  - kinematics : MLP per-frame su 5 feature (4 bbox + ego) -> [B,T,d_model].
  - crop       : MLP per-frame su 768 feature ConvNeXt -> [B,T,d_model].

Ogni stream produce una sequenza di token [B,T,d_model]. La FUSIONE combina
gli stream attivi (config: streams.fusion):

  - concat_tokens : somma dei token per-frame degli stream attivi (con
                    embedding di stream additivo per distinguerli) ->
                    UN transformer temporale -> ultimo step -> classifier.
                    (equivale al PedGT originale quando c'e' solo la pose.)
  - cross_attention : uno stream 'primario' (il primo attivo, per default la
                    pose) fa da query; gli altri stream, impilati come token
                    extra per-frame, fanno da key/value in una cross-attention
                    per-frame. Output arricchito -> transformer temporale.
  - late_fusion   : un transformer temporale SEPARATO per ogni stream ->
                    ultimo step di ciascuno -> concat dei vettori finali ->
                    classifier.

Compatibilita': con solo la pose attiva e fusion=concat_tokens il modello
coincide col PedGT originale (stesso GCN, stesso transformer, ultimo step).
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

from data.skeleton import build_edge_index, NUM_JOINTS


# ============================================================ STREAM ENCODERS
class PedGTSpatial(nn.Module):
    """Triplet di GCN -> maxpool sui nodi -> FC verso d_model (pose stream)."""

    def __init__(self, in_channels: int = 5, gcn_hidden: int = 64,
                 spatial_out: int = 64, d_model: int = 128):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)
        self.gcn3 = GCNConv(gcn_hidden, spatial_out)
        self.bn1 = nn.BatchNorm1d(gcn_hidden)
        self.bn2 = nn.BatchNorm1d(gcn_hidden)
        self.proj = nn.Linear(spatial_out, d_model)

        edge_index = build_edge_index(self_loops=True)
        self.register_buffer("edge_index", edge_index)

    def forward(self, x_nodes: torch.Tensor) -> torch.Tensor:
        """x_nodes : [M, 19, in_channels]  (M = B*T). return [M, d_model]."""
        M, N, C = x_nodes.shape
        x = x_nodes.reshape(M * N, C)
        ei = self._batched_edge_index(M, N, x.device)

        h = F.relu(self.bn1(self.gcn1(x, ei)))
        h = F.relu(self.bn2(self.gcn2(h, ei)))
        h = F.relu(self.gcn3(h, ei))

        h = h.view(M, N, -1)
        h_sm = h.max(dim=1).values
        return self.proj(h_sm)

    def _batched_edge_index(self, M: int, N: int, device) -> torch.Tensor:
        ei = self.edge_index.to(device)
        E = ei.shape[1]
        offsets = (torch.arange(M, device=device) * N).repeat_interleave(E)
        ei_rep = ei.repeat(1, M) + offsets.unsqueeze(0)
        return ei_rep


class PoseStream(nn.Module):
    """Wrapper: [B,T,19,C] -> [B,T,d_model]."""

    def __init__(self, in_channels, gcn_hidden, spatial_out, d_model):
        super().__init__()
        self.spatial = PedGTSpatial(in_channels, gcn_hidden, spatial_out, d_model)

    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        B, T, N, C = keypoints.shape
        x = keypoints.reshape(B * T, N, C)
        h = self.spatial(x)              # [B*T, d_model]
        return h.view(B, T, -1)


class MLPStream(nn.Module):
    """Encoder per-frame per feature vettoriali (kinematics, crop).
    [B,T,in_dim] -> [B,T,d_model]. MLP a 2 layer con LayerNorm+ReLU."""

    def __init__(self, in_dim: int, d_model: int, hidden: Optional[int] = None,
                 dropout: float = 0.1):
        super().__init__()
        hidden = hidden or d_model
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================ TEMPORAL / HEAD
def _make_transformer(d_model, n_heads, dim_ff, dropout, n_layers):
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
        dropout=dropout, activation="relu", batch_first=True)
    return nn.TransformerEncoder(layer, num_layers=n_layers)


class TemporalEncoder(nn.Module):
    """Positional embedding + Transformer encoder + (opz.) maschera causale.
    Restituisce l'intera sequenza [B,T,d_model]."""

    def __init__(self, d_model, n_heads, n_layers, dim_ff, dropout,
                 obs_len, causal=False):
        super().__init__()
        self.causal = causal
        self.pos_embedding = nn.Parameter(torch.zeros(1, obs_len, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.transformer = _make_transformer(
            d_model, n_heads, dim_ff, dropout, n_layers)
        if causal:
            mask = torch.triu(torch.ones(obs_len, obs_len, dtype=torch.bool),
                              diagonal=1)
            self.register_buffer("causal_mask", mask, persistent=False)
        else:
            self.causal_mask = None

    def forward(self, h_seq: torch.Tensor) -> torch.Tensor:
        B, T, D = h_seq.shape
        h_seq = h_seq + self.pos_embedding[:, :T, :]
        attn_mask = None
        if self.causal:
            if self.causal_mask is not None and self.causal_mask.size(0) == T:
                attn_mask = self.causal_mask
            else:
                attn_mask = torch.triu(
                    torch.ones(T, T, dtype=torch.bool, device=h_seq.device),
                    diagonal=1)
        return self.transformer(h_seq, mask=attn_mask)


# ==================================================================== MODEL
STREAM_KEYS = ("pose", "kinematics", "crop")


class PedGT(nn.Module):
    """
    PedGT multi-stream.

    Firma forward compatibile con train.py: accetta keyword `keypoints`,
    `kinematics`, `crop_feat`, `ego_speed`, ... . Usa solo gli stream
    attivati alla costruzione (streams_cfg).

    streams_cfg (dict), esempio:
        {
          "pose":       {"enabled": True, "use_center_channels": True},
          "kinematics": {"enabled": False, "in_dim": 5},
          "crop":       {"enabled": False, "in_dim": 768},
          "fusion":     "concat_tokens",   # concat_tokens|cross_attention|late_fusion
        }
    """

    def __init__(
        self,
        in_channels: int = 5,          # canali pose (5 o 3), retro-compat
        d_model: int = 128,
        gcn_hidden: int = 64,
        spatial_out: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.7,
        encoder_dropout: float = None,
        num_classes: int = 2,
        obs_len: int = 26,
        causal: bool = False,
        streams_cfg: Optional[Dict] = None,
        # legacy, ignorato (ego ora vive nel ramo kinematics):
        use_ego_speed: bool = False,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.d_model = d_model
        if encoder_dropout is None:
            encoder_dropout = dropout

        # -- config stream: default = solo pose, concat_tokens (baseline) ----
        if streams_cfg is None:
            streams_cfg = {"pose": {"enabled": True,
                                    "use_center_channels": True}}
        self.fusion = streams_cfg.get("fusion", "concat_tokens")
        assert self.fusion in ("concat_tokens", "cross_attention", "late_fusion"), \
            f"fusione non valida: {self.fusion}"

        # quali stream sono attivi (ordine fisso e deterministico)
        self.active: List[str] = [
            k for k in STREAM_KEYS
            if streams_cfg.get(k, {}).get("enabled", False)
        ]
        if not self.active:
            self.active = ["pose"]   # fallback di sicurezza
        assert len(self.active) >= 1

        # -- encoder per stream ---------------------------------------------
        self.encoders = nn.ModuleDict()
        for k in self.active:
            cfg = streams_cfg.get(k, {})
            if k == "pose":
                # use_center_channels decide 5 (x,y,conf,cx,cy) vs 3 (x,y,conf).
                # 'in_channels' nel cfg dello stream e' un override esplicito
                # (ha la precedenza) per casi speciali.
                pose_ch = 5 if cfg.get("use_center_channels", True) else 3
                pose_ch = cfg.get("in_channels", pose_ch)
                self.encoders["pose"] = PoseStream(
                    pose_ch, gcn_hidden, spatial_out, d_model)
            elif k == "kinematics":
                self.encoders["kinematics"] = MLPStream(
                    cfg.get("in_dim", 5), d_model,
                    hidden=cfg.get("hidden", d_model),
                    dropout=cfg.get("dropout", 0.1))
            elif k == "crop":
                self.encoders["crop"] = MLPStream(
                    cfg.get("in_dim", 768), d_model,
                    hidden=cfg.get("hidden", d_model),
                    dropout=cfg.get("dropout", 0.1))

        # -- moduli di fusione ----------------------------------------------
        if self.fusion == "concat_tokens":
            # embedding di stream additivo (distingue i token sommati)
            self.stream_emb = nn.Parameter(
                torch.zeros(len(self.active), 1, 1, d_model))
            nn.init.trunc_normal_(self.stream_emb, std=0.02)
            self.temporal = TemporalEncoder(
                d_model, n_heads, n_layers, dim_feedforward,
                encoder_dropout, obs_len, causal)
            head_in = d_model

        elif self.fusion == "cross_attention":
            # primario = primo stream attivo (pose se presente)
            self.primary = self.active[0]
            self.others = self.active[1:]
            if self.others:
                self.cross_attn = nn.MultiheadAttention(
                    d_model, n_heads, dropout=encoder_dropout,
                    batch_first=True)
                self.cross_norm = nn.LayerNorm(d_model)
            self.temporal = TemporalEncoder(
                d_model, n_heads, n_layers, dim_feedforward,
                encoder_dropout, obs_len, causal)
            head_in = d_model

        elif self.fusion == "late_fusion":
            # un transformer temporale per stream
            self.temporals = nn.ModuleDict({
                k: TemporalEncoder(d_model, n_heads, n_layers,
                                   dim_feedforward, encoder_dropout,
                                   obs_len, causal)
                for k in self.active
            })
            head_in = d_model * len(self.active)

        # -- classificatore --------------------------------------------------
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(head_in, num_classes)

        # log helper per train.py
        self.input_dim = in_channels
        self.use_bbox_delta = False

    # ------------------------------------------------------------ encode
    def _encode_streams(self, inputs: Dict[str, torch.Tensor]
                        ) -> Dict[str, torch.Tensor]:
        """Ritorna {stream: [B,T,d_model]} per gli stream attivi."""
        out = {}
        for k in self.active:
            if k == "pose":
                out["pose"] = self.encoders["pose"](inputs["keypoints"])
            elif k == "kinematics":
                out["kinematics"] = self.encoders["kinematics"](inputs["kinematics"])
            elif k == "crop":
                out["crop"] = self.encoders["crop"](inputs["crop_feat"])
        return out

    # ------------------------------------------------------------ forward
    def forward(self, keypoints: torch.Tensor = None,
                kinematics: torch.Tensor = None,
                crop_feat: torch.Tensor = None,
                ego_speed: torch.Tensor = None,
                *args, **kwargs) -> torch.Tensor:
        inputs = {"keypoints": keypoints, "kinematics": kinematics,
                  "crop_feat": crop_feat}
        enc = self._encode_streams(inputs)     # {stream: [B,T,d]}

        if self.fusion == "concat_tokens":
            # somma dei token per-frame + embedding di stream
            h = None
            for i, k in enumerate(self.active):
                tok = enc[k] + self.stream_emb[i]
                h = tok if h is None else h + tok
            h_enc = self.temporal(h)                 # [B,T,d]
            feat = h_enc[:, -1, :]

        elif self.fusion == "cross_attention":
            q = enc[self.primary]                    # [B,T,d]
            if self.others:
                B, T, d = q.shape
                # key/value: token degli altri stream impilati per-frame
                # -> [B, T, S, d] -> reshape a cross-attn per-frame
                kv = torch.stack([enc[k] for k in self.others], dim=2)  # [B,T,S,d]
                S = kv.shape[2]
                qf = q.reshape(B * T, 1, d)
                kvf = kv.reshape(B * T, S, d)
                attn_out, _ = self.cross_attn(qf, kvf, kvf)   # [B*T,1,d]
                attn_out = attn_out.reshape(B, T, d)
                q = self.cross_norm(q + attn_out)             # residuo
            h_enc = self.temporal(q)
            feat = h_enc[:, -1, :]

        elif self.fusion == "late_fusion":
            finals = []
            for k in self.active:
                h_enc = self.temporals[k](enc[k])
                finals.append(h_enc[:, -1, :])
            feat = torch.cat(finals, dim=-1)         # [B, d*S]

        logits = self.classifier(self.dropout(feat))
        return logits
