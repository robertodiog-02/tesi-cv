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

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv

from data.skeleton import build_edge_index, NUM_JOINTS


# Mappa nome -> costruttore di un layer GNN message-passing.
# Ogni entry e' una lambda (in_dim, out_dim) -> nn.Module conv.
def _make_gnn_conv(gnn_type: str, in_dim: int, out_dim: int) -> nn.Module:
    """Costruisce un layer di convoluzione su grafo del tipo richiesto.

    gcn       : GCNConv (baseline, come il paper PedGT).
    graphsage : SAGEConv (aggregazione mean dei vicini + skip del nodo).
    gat       : GATConv (attenzione sugli edge, 4 head concatenate).
    gin       : GINConv (MLP su somma dei vicini, molto espressivo).
    """
    g = gnn_type.lower()
    if g in ("gcn", "gcnconv"):
        return GCNConv(in_dim, out_dim)
    if g in ("graphsage", "sage", "sageconv"):
        return SAGEConv(in_dim, out_dim)
    if g in ("gat", "gatconv"):
        # heads=4 concatenate -> out_dim deve essere divisibile per 4
        heads = 4
        assert out_dim % heads == 0, \
            f"GAT: out_dim ({out_dim}) deve essere divisibile per heads ({heads})"
        return GATConv(in_dim, out_dim // heads, heads=heads, concat=True)
    if g in ("gin", "ginconv"):
        mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(),
                            nn.Linear(out_dim, out_dim))
        return GINConv(mlp)
    raise ValueError(f"gnn_type non supportato: {gnn_type} "
                     f"(usa gcn|graphsage|gat|gin)")


# ============================================================ STREAM ENCODERS
class PedGTSpatial(nn.Module):
    """Triplet di GCN -> maxpool sui nodi -> FC verso d_model (pose stream).

    input_proj (opzionale): lista di dimensioni per un MLP per-nodo applicato
    PRIMA di gcn1. Es. [32] -> un dense 5->32; [32,64] -> due dense 5->32->64.
    Ogni giunto e' proiettato indipendentemente (stesso peso per tutti i nodi,
    come un embedding per-nodo). Se None, l'input va diretto a gcn1 (baseline).

    Razionale: alleggerisce gcn1 dal salto in_channels->gcn_hidden (che nel
    baseline ha il Jacobiano piu' grande) proiettando l'input in uno spazio
    piu' ricco a monte. ATTENZIONE: aggiunge parametri -> puo' accentuare
    l'overfitting su dataset piccoli. Tenerlo stretto (es. [16] o [32]).
    """

    def __init__(self, in_channels: int = 5, gcn_hidden: int = 64,
                 spatial_out: int = 64, d_model: int = 128,
                 input_proj=None, input_proj_dropout: float = 0.0,
                 gnn_type: str = "gcn",
                 exclude_head: bool = False, add_cross_limb: bool = False):
        super().__init__()
        self.gnn_type = gnn_type
        # -- input projection per-nodo (opzionale) --------------------------
        gcn_in = in_channels
        self.input_proj = None
        if input_proj:
            layers = []
            prev = in_channels
            for dim in input_proj:
                layers += [nn.Linear(prev, dim), nn.ReLU()]
                if input_proj_dropout > 0:
                    layers.append(nn.Dropout(input_proj_dropout))
                prev = dim
            self.input_proj = nn.Sequential(*layers)
            gcn_in = prev                       # gcn1 riceve l'ultima dim proiettata

        # Triplet di conv su grafo del tipo scelto (gcn/graphsage/gat/gin).
        # I nomi restano gcn1/2/3 per compatibilita' col logging per-layer.
        self.bn0 = nn.BatchNorm1d(in_channels)
        self.gcn1 = _make_gnn_conv(gnn_type, gcn_in, gcn_hidden)
        self.gcn2 = _make_gnn_conv(gnn_type, gcn_hidden, gcn_hidden)
        self.gcn3 = _make_gnn_conv(gnn_type, gcn_hidden, spatial_out)
        self.bn1 = nn.BatchNorm1d(gcn_hidden)
        self.bn2 = nn.BatchNorm1d(gcn_hidden)
        self.proj = nn.Linear(spatial_out, d_model)

        edge_index = build_edge_index(self_loops=True,
                                      exclude_head=exclude_head,
                                      add_cross_limb=add_cross_limb)
        self.register_buffer("edge_index", edge_index)

    def forward(self, x_nodes: torch.Tensor) -> torch.Tensor:
        """x_nodes : [M, 19, in_channels]  (M = B*T). return [M, d_model]."""
        M, N, C = x_nodes.shape
        x = x_nodes.reshape(M * N, C)
        # proiezione per-nodo prima della GCN (se attiva)
        x = self.bn0(x)
        if self.input_proj is not None:
            x = self.input_proj(x)
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


class PoseTransformerSpatial(nn.Module):
    """Encoder spaziale attention-only sui giunti (stile PoseFormer).

    Nessuna GCN: i J giunti di un frame diventano una sequenza di J token;
    un Transformer encoder li mette in relazione via self-attention (l'
    attenzione impara da sola quali giunti interagiscono, senza scheletro
    fisso). Un embedding per-giunto apprendibile fa da 'posizione' spaziale.

    Aggregazione finale sui giunti: mean-pool -> proiezione a d_model.
    Firma I/O identica a PedGTSpatial: [M, J, C] -> [M, d_model], cosi'
    e' intercambiabile dentro PoseStream.
    """

    def __init__(self, in_channels: int = 5, d_model: int = 128,
                 num_joints: int = NUM_JOINTS, n_heads: int = 4,
                 n_layers: int = 2, dim_ff: Optional[int] = None,
                 dropout: float = 0.1, embed_dim: Optional[int] = None):
        super().__init__()
        embed_dim = embed_dim or d_model
        dim_ff = dim_ff or (2 * embed_dim)
        self.in_proj = nn.Linear(in_channels, embed_dim)     # per-giunto
        # embedding posizionale per giunto (una riga per giunto)
        self.joint_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        nn.init.trunc_normal_(self.joint_embed, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation="relu", batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, d_model)

    def forward(self, x_nodes: torch.Tensor) -> torch.Tensor:
        """x_nodes : [M, J, C] (M = B*T). return [M, d_model]."""
        M, J, C = x_nodes.shape
        h = self.in_proj(x_nodes)                    # [M, J, embed]
        h = h + self.joint_embed[:, :J, :]
        h = self.encoder(h)                          # [M, J, embed]
        h = self.norm(h)
        h = h.mean(dim=1)                            # pool sui giunti
        return self.proj(h)                          # [M, d_model]


class PoseStream(nn.Module):
    """Wrapper: [B,T,J,C] -> [B,T,d_model].

    L'encoder spaziale (per-frame) e' selezionabile:
      encoder='gcn'|'graphsage'|'gat'|'gin' -> PedGTSpatial (message passing);
      encoder='transformer'                 -> PoseTransformerSpatial (attn).
    """

    def __init__(self, in_channels, gcn_hidden, spatial_out, d_model,
                 input_proj=None, input_proj_dropout: float = 0.0,
                 encoder: str = "gcn", gnn_type: Optional[str] = None,
                 exclude_head: bool = False, add_cross_limb: bool = False,
                 num_joints: int = NUM_JOINTS,
                 tf_heads: int = 4, tf_layers: int = 2,
                 tf_dim_ff: Optional[int] = None, tf_dropout: float = 0.1):
        super().__init__()
        self.encoder_kind = encoder.lower()
        if self.encoder_kind == "transformer":
            self.spatial = PoseTransformerSpatial(
                in_channels=in_channels, d_model=d_model,
                num_joints=num_joints, n_heads=tf_heads, n_layers=tf_layers,
                dim_ff=tf_dim_ff, dropout=tf_dropout)
        else:
            # encoder message-passing; 'encoder' stesso nomina il conv se
            # gnn_type non e' dato esplicitamente (es. encoder='graphsage').
            gt = gnn_type or self.encoder_kind
            if gt == "gcn" or gt in ("graphsage", "gat", "gin"):
                pass
            else:
                gt = "gcn"
            self.spatial = PedGTSpatial(
                in_channels, gcn_hidden, spatial_out, d_model,
                input_proj=input_proj, input_proj_dropout=input_proj_dropout,
                gnn_type=gt, exclude_head=exclude_head,
                add_cross_limb=add_cross_limb)

    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        B, T, N, C = keypoints.shape
        x = keypoints.reshape(B * T, N, C)
        h = self.spatial(x)              # [B*T, d_model]
        return h.view(B, T, -1)


class MLPStream(nn.Module):
    """Encoder per-frame per feature vettoriali (kinematics, crop).
    [B,T,in_dim] -> [B,T,d_model]. MLP a 2 layer con LayerNorm+ReLU.

    input_dropout: dropout applicato PRIMA della prima Linear, cioe' sulle
    feature grezze (utile per il crop 768-dim: spegne a caso una frazione
    dei canali visivi per ridurre la memorizzazione dell'aspetto)."""

    def __init__(self, in_dim: int, d_model: int, hidden: Optional[int] = None,
                 dropout: float = 0.1, input_dropout: float = 0.0):
        super().__init__()
        hidden = hidden or d_model
        self.in_drop = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.in_drop(x))


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
                # Canali pose: x,y (2) + conf (se use_confidence) + cx,cy
                # (se use_center_channels). 'in_channels' resta override esplicito.
                pose_ch = 2
                if cfg.get("use_confidence", True):
                    pose_ch += 1
                if cfg.get("use_center_channels", True):
                    pose_ch += 2
                pose_ch = cfg.get("in_channels", pose_ch)
                exclude_head   = cfg.get("exclude_head", False)
                add_cross_limb = cfg.get("add_cross_limb", False)
                # num giunti attivi (14 se testa esclusa, altrimenti 19)
                n_joints = 14 if exclude_head else NUM_JOINTS
                self.encoders["pose"] = PoseStream(
                    pose_ch, gcn_hidden, spatial_out, d_model,
                    input_proj=cfg.get("input_proj", None),
                    input_proj_dropout=cfg.get("input_proj_dropout", 0.0),
                    encoder=cfg.get("encoder", "gcn"),
                    gnn_type=cfg.get("gnn_type", None),
                    exclude_head=exclude_head, add_cross_limb=add_cross_limb,
                    num_joints=n_joints,
                    tf_heads=cfg.get("tf_heads", 4),
                    tf_layers=cfg.get("tf_layers", 2),
                    tf_dim_ff=cfg.get("tf_dim_ff", None),
                    tf_dropout=cfg.get("tf_dropout", 0.1))
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
