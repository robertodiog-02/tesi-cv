"""
PedGT — Skeleton-based Graph-Transformer (replica del paper)
============================================================
Riaz et al., IEEE IV 2025.

Architettura (Sec. III-C, Fig. 2):
  Spatial module : 3 x GCNConv (PyG) con ReLU + BatchNorm dopo i primi 2,
                   max-pool sui nodi -> vettore 64-dim per frame,
                   FC di proiezione -> F=128 per il transformer.
  Temporal module: Transformer encoder, 2 layer, 4 head, d_model=128.
                   Si usa SOLO l'output dell'ultimo time step (paper eq. 7).
  Classification : Dropout(0.7) + FC -> 2 classi (C / NC).

Input atteso dal forward:
  keypoints : [B, T, 17, 5]  (x, y, conf, cx, cy) gia' normalizzati
              T = 26 (obs_len di PedGT).

Iperparametri dal paper (Sec. IV-B):
  GCN 5->64, transformer 2 layer x 4 head, d_model=128, dropout 0.7.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

from data.skeleton import build_edge_index, NUM_JOINTS


class PedGTSpatial(nn.Module):
    """Triplet di GCN -> maxpool sui nodi -> FC verso d_model."""

    def __init__(self, in_channels: int = 5, gcn_hidden: int = 64,
                 spatial_out: int = 64, d_model: int = 128):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)
        self.gcn3 = GCNConv(gcn_hidden, spatial_out)
        self.bn1 = nn.BatchNorm1d(gcn_hidden)
        self.bn2 = nn.BatchNorm1d(gcn_hidden)
        # proiezione H_SM (64) -> H_TI (d_model = 128)  [paper eq. 4]
        self.proj = nn.Linear(spatial_out, d_model)

        edge_index = build_edge_index(self_loops=True)
        self.register_buffer("edge_index", edge_index)

    def forward(self, x_nodes: torch.Tensor) -> torch.Tensor:
        """
        x_nodes : [M, 17, in_channels]  con M = B*T (grafi indipendenti).
        return  : [M, d_model]
        """
        M, N, C = x_nodes.shape
        # batch di M grafi identici (stesso edge_index, offset sui nodi)
        x = x_nodes.reshape(M * N, C)                       # [M*N, C]
        ei = self._batched_edge_index(M, N, x.device)       # [2, M*E]

        h = F.relu(self.bn1(self.gcn1(x, ei)))
        h = F.relu(self.bn2(self.gcn2(h, ei)))
        h = F.relu(self.gcn3(h, ei))                        # [M*N, spatial_out]

        h = h.view(M, N, -1)
        h_sm = h.max(dim=1).values                          # maxpool sui nodi -> [M, spatial_out]
        h_ti = self.proj(h_sm)                              # [M, d_model]
        return h_ti

    def _batched_edge_index(self, M: int, N: int, device) -> torch.Tensor:
        ei = self.edge_index.to(device)                     # [2, E]
        E = ei.shape[1]
        offsets = (torch.arange(M, device=device) * N).repeat_interleave(E)
        ei_rep = ei.repeat(1, M)                            # [2, M*E]
        ei_rep = ei_rep + offsets.unsqueeze(0)
        return ei_rep


class PedGT(nn.Module):
    """
    Modello completo PedGT.

    Firma forward compatibile con train.py: accetta keyword `keypoints`.
    Gli altri input geometry (bbox/delta/ego_speed) sono ignorati: PedGT
    usa solo pose + center (gia' incluso nei 5 canali di keypoints).
    """

    def __init__(
        self,
        in_channels: int = 5,
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
        use_ego_speed: bool = False,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.causal = causal
        # Dropout interno all'encoder (attenzione + FFN). Se None, ricade su
        # `dropout` (comportamento storico). Il paper descrive 0.7 solo PRIMA
        # del classificatore; valori 0.1-0.2 qui sono tipici per un encoder.
        if encoder_dropout is None:
            encoder_dropout = dropout
        self.spatial = PedGTSpatial(in_channels, gcn_hidden, spatial_out, d_model)

        # Proiezione ego-speed [B,T,1] -> [B,T,d_model], sommata al vettore-frame
        # prima del transformer (in parallelo al positional embedding). Il
        # bias e' inizializzato a zero cosi' all'inizio l'iniezione e' quasi
        # nulla e non destabilizza il training.
        self.use_ego_speed = use_ego_speed
        if self.use_ego_speed:
            self.speed_proj = nn.Linear(1, d_model)
            nn.init.zeros_(self.speed_proj.bias)

        self.pos_embedding = nn.Parameter(torch.zeros(1, obs_len, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=encoder_dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.dropout = nn.Dropout(dropout)   # dropout PRIMA del classificatore (paper)
        self.classifier = nn.Linear(d_model, num_classes)

        # Maschera causale [T,T]: True = posizione MASCHERATA (non visibile).
        # Ogni step t attende solo a {0..t}. Registrata come buffer cosi'
        # segue il device del modello; rigenerata se T cambia.
        if self.causal:
            mask = torch.triu(torch.ones(obs_len, obs_len, dtype=torch.bool),
                              diagonal=1)
            self.register_buffer("causal_mask", mask, persistent=False)
        else:
            self.causal_mask = None

        self.input_dim = in_channels  # per i log di train.py
        self.use_bbox_delta = False

    def forward(self, keypoints: torch.Tensor, ego_speed: torch.Tensor = None,
                *args, **kwargs) -> torch.Tensor:
        """
        keypoints : [B, T, 17, 5]
        ego_speed : [B, T, 1]  (usata solo se use_ego_speed=True)
        return    : [B, num_classes]
        """
        B, T, N, C = keypoints.shape
        x = keypoints.reshape(B * T, N, C)
        h_ti = self.spatial(x)                      # [B*T, d_model]
        h_seq = h_ti.view(B, T, -1)                 # [B, T, d_model]

        h_seq = h_seq + self.pos_embedding[:, :T, :]

        # Iniezione ego-speed: proiettata a d_model e sommata al vettore-frame
        # (come un secondo embedding additivo, per-frame). Se manca, ignorata.
        if self.use_ego_speed and ego_speed is not None:
            h_seq = h_seq + self.speed_proj(ego_speed[:, :T, :])

        # Maschera causale opzionale (deviazione dal PedGT originale, che e'
        # bidirezionale). src_mask [T,T] con True = posizione non visibile.
        attn_mask = None
        if self.causal:
            if self.causal_mask is not None and self.causal_mask.size(0) == T:
                attn_mask = self.causal_mask
            else:
                attn_mask = torch.triu(
                    torch.ones(T, T, dtype=torch.bool, device=h_seq.device),
                    diagonal=1)
        h_enc = self.transformer(h_seq, mask=attn_mask)   # [B, T, d_model]

        h_last = h_enc[:, -1, :]                    # solo ultimo step [paper eq. 7]
        logits = self.classifier(self.dropout(h_last))
        return logits