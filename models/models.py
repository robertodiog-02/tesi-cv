"""
Modelli per PIE PCIP
====================

Gerarchia:
  BaselineGRU     - lower bound: solo bbox + ego_speed → GRU → FC
  BranchA_GCN     - Branch A: skeleton GCN (richiede pose cache)
  BranchB_GAT     - Branch B: scene GAT (richiede detection objects)
  CrossAttnDualGraph - architettura completa con CGAM
"""

from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


# ─── Baseline GRU ────────────────────────────────────────────────────────────

class BaselineGRU(nn.Module):
    """
    Baseline GRU.

    Feature di input modulari, decise tramite flag:
        bbox        : 4 dim   (SEMPRE presente)
        bbox_delta  : +4 dim  se use_bbox_delta=True
        ego_speed   : +1 dim  se use_ego_speed=True

    input_dim effettivo viene calcolato automaticamente in base ai flag.
    """

    def __init__(
        self,
        hidden_dim:     int   = 256,
        num_layers:     int   = 2,
        dropout:        float = 0.3,
        use_bbox_delta: bool  = True,
        use_ego_speed:  bool  = False,
    ):
        super().__init__()
        self.use_bbox_delta = use_bbox_delta
        self.use_ego_speed  = use_ego_speed
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # bbox(4) sempre, +4 delta, +1 ego_speed
        input_dim = 4 + (4 if use_bbox_delta else 0) + (1 if use_ego_speed else 0)
        self.input_dim = input_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.temporal_attn = nn.Linear(hidden_dim, 1)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(
        self,
        bbox:       torch.Tensor,            # [B, T, 4]
        bbox_delta: torch.Tensor = None,     # [B, T, 4]
        ego_speed:  torch.Tensor = None,     # [B, T, 1]
    ) -> torch.Tensor:                       # [B, 2]

        features = [bbox]
        if self.use_bbox_delta:
            features.append(bbox_delta)
        if self.use_ego_speed:
            features.append(ego_speed)
        x = torch.cat(features, dim=-1)      # [B, T, input_dim]

        x = self.input_proj(x)
        gru_out, _ = self.gru(x)

        attn_weights = torch.softmax(self.temporal_attn(gru_out).squeeze(-1), dim=-1)
        context = torch.sum(gru_out * attn_weights.unsqueeze(-1), dim=1)

        logits = self.decoder(context)
        return logits


# ─── Branch A: Skeleton GCN (stub, richiede pose cache) ──────────────────────

class SkeletonGCNLayer(nn.Module):
    """
    Single GCN layer per grafo scheletro.
    Input: feature nodi [B, T, N_joints, C_in]
    Output: feature nodi [B, T, N_joints, C_out]
    """

    def __init__(self, in_channels: int, out_channels: int, n_joints: int = 17):
        super().__init__()
        self.n_joints = n_joints
        # Adjacency matrix apprendibile (come PedGraph+)
        self.adj_learnable = nn.Parameter(
            torch.eye(n_joints) + 0.01 * torch.randn(n_joints, n_joints)
        )
        self.weight = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, C]
        B, T, N, C = x.shape

        # Normalizza adjacency
        adj = self.adj_learnable + self.adj_learnable.t()
        d = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        adj_norm = adj / d

        # Message passing: x_new = adj_norm @ x
        x_flat = x.view(B * T, N, C)
        x_agg = torch.bmm(
            adj_norm.unsqueeze(0).expand(B * T, -1, -1), x_flat
        )  # [BT, N, C]

        # Linear transform
        x_out = self.weight(x_agg)            # [BT, N, C_out]
        x_out = x_out.view(B, T, N, -1)

        # BatchNorm su canali
        x_out = self.bn(x_out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self.relu(x_out)


class BranchA_GCN(nn.Module):
    """
    Branch A: skeleton GCN per processing dei keypoints.
    
    Input: keypoints [B, T, 17, 3] (x, y, confidence)
    Output: joint embeddings [B, T, 17, d_model]
    
    Architettura: 3 GCN layers con hidden_dim crescente
    Nota: richiede pose cache AlphaPose — stub per adesso
    """

    def __init__(
        self,
        in_channels: int = 3,    # x, y, confidence
        hidden_channels: int = 64,
        out_channels: int = 128,
        n_joints: int = 17,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_joints = n_joints

        self.gcn_layers = nn.ModuleList([
            SkeletonGCNLayer(in_channels,      hidden_channels, n_joints),
            SkeletonGCNLayer(hidden_channels,  hidden_channels, n_joints),
            SkeletonGCNLayer(hidden_channels,  out_channels,    n_joints),
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        # keypoints: [B, T, 17, 3]
        x = keypoints
        for layer in self.gcn_layers:
            x = self.dropout(layer(x))
        return x  # [B, T, 17, 128]


# ─── Branch B: Scene GAT (stub) ───────────────────────────────────────────────

class SceneGATLayer(nn.Module):
    """
    Single GAT layer per scene graph eterogeneo.
    Input: node features [B, T, N_nodes, C_in]
    Output: node features [B, T, N_nodes, C_out]
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        n_heads:      int = 4,
        dropout:      float = 0.2,
    ):
        super().__init__()
        assert out_channels % n_heads == 0
        self.n_heads = n_heads
        self.d_head = out_channels // n_heads

        self.W_q = nn.Linear(in_channels, out_channels, bias=False)
        self.W_k = nn.Linear(in_channels, out_channels, bias=False)
        self.W_v = nn.Linear(in_channels, out_channels, bias=False)
        self.out_proj = nn.Linear(out_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_channels)

    def forward(
        self,
        x:    torch.Tensor,           # [B, T, N, C]
        mask: Optional[torch.Tensor], # [N, N] adjacency mask (0=no edge)
    ) -> torch.Tensor:
        B, T, N, C = x.shape
        x_flat = x.view(B * T, N, C)

        Q = self.W_q(x_flat).view(B * T, N, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x_flat).view(B * T, N, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x_flat).view(B * T, N, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)

        # Applica mask: archi non presenti → -inf
        if mask is not None:
            # mask: [N, N], 0 = no edge
            attn_mask = (mask == 0).unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
            scores = scores.masked_fill(attn_mask, float("-inf"))

        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, V)  # [BT, heads, N, d_head]
        out = out.transpose(1, 2).contiguous().view(B * T, N, -1)
        out = self.out_proj(out)
        out = self.norm(out + x_flat)  # residual connection
        return out.view(B, T, N, -1)


class BranchB_GAT(nn.Module):
    """
    Branch B: scene GAT per processing del grafo scena eterogeneo.
    
    Nodi: pedone (1) + veicoli + semaforo + crosswalk + ego-vehicle
    
    Input:
        node_features: [B, T, N_nodes, C_in]
        adjacency:     [N_nodes, N_nodes] — topologia sparsa semantica
    
    Output: node embeddings [B, T, N_nodes, d_model]
    """

    def __init__(
        self,
        in_channels:  int = 16,   # bbox(4) + tl_onehot(5) + crosswalk(1) + ...
        hidden_dim:   int = 128,
        out_channels: int = 128,
        n_layers:     int = 2,
        n_heads:      int = 4,
        dropout:      float = 0.2,
    ):
        super().__init__()

        self.input_proj = nn.Linear(in_channels, hidden_dim)

        self.gat_layers = nn.ModuleList([
            SceneGATLayer(hidden_dim, hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, out_channels)

    def forward(
        self,
        node_features: torch.Tensor,          # [B, T, N, C_in]
        adjacency:     Optional[torch.Tensor], # [N, N]
    ) -> torch.Tensor:                         # [B, T, N, C_out]
        x = self.input_proj(node_features)
        for layer in self.gat_layers:
            x = layer(x, adjacency)
        return self.output_proj(x)


# ─── CGAM: Cross-Graph Attention Module ──────────────────────────────────────

class CGAM(nn.Module):
    """
    Cross-Graph Attention Module — il contributo principale.
    
    Fa cross-attention bidirezionale node-level tra:
    - Branch A: 17 joint embeddings [B, T, 17, d]
    - Branch B: N_scene node embeddings [B, T, N_scene, d]
    
    Ogni joint può attendere selettivamente ai nodi scena e viceversa.
    Operazione per-frame, prima del GRU temporale.
    
    Returns:
        h_A_updated: [B, T, 17, d]      - joint embeddings aggiornati
        h_B_updated: [B, T, N_scene, d] - scene embeddings aggiornati
    """

    def __init__(
        self,
        d_model:  int = 128,
        n_heads:  int = 4,
        dropout:  float = 0.2,
    ):
        super().__init__()
        # A→B: joint embeddings come Query, scene come K/V
        self.cross_attn_A2B = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        # B→A: scene embeddings come Query, joint come K/V
        self.cross_attn_B2A = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.norm_A = nn.LayerNorm(d_model)
        self.norm_B = nn.LayerNorm(d_model)

    def forward(
        self,
        h_A: torch.Tensor,  # [B, T, 17, d]
        h_B: torch.Tensor,  # [B, T, N_scene, d]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, N_A, d = h_A.shape
        N_B = h_B.shape[2]

        # Flatten temporal dimension per MultiheadAttention batch
        h_A_flat = h_A.view(B * T, N_A, d)
        h_B_flat = h_B.view(B * T, N_B, d)

        # A→B: ogni joint guarda tutti i nodi scena
        h_A_cross, _ = self.cross_attn_A2B(
            query=h_A_flat, key=h_B_flat, value=h_B_flat
        )
        h_A_updated = self.norm_A(h_A_flat + h_A_cross)  # residual

        # B→A: ogni nodo scena guarda tutti i joint
        h_B_cross, _ = self.cross_attn_B2A(
            query=h_B_flat, key=h_A_flat, value=h_A_flat
        )
        h_B_updated = self.norm_B(h_B_flat + h_B_cross)  # residual

        return (
            h_A_updated.view(B, T, N_A, d),
            h_B_updated.view(B, T, N_B, d),
        )


# ─── Modello completo (stub — richiede pose cache) ────────────────────────────

class CrossAttentiveDualGraph(nn.Module):
    """
    Cross-Attentive Dual-Graph Network — architettura completa.
    
    Richiede:
    - Pose cache AlphaPose per Branch A
    - Object detection per Branch B (o annotation PIE)
    
    Adesso è uno STUB — da implementare dopo la pose cache.
    
    Pipeline:
    1. Branch A: keypoints → SkeletonGCN → {h_j1,...,h_j17} per frame
    2. Branch B: scene nodes → SceneGAT  → {h_ped,...,h_cross} per frame
    3. CGAM:    cross-attention bidirezionale per frame
    4. Pool:    attention-weighted pooling dei joint e nodi aggiornati
    5. GRU:     sequenza temporale di [h_A_t, h_B_t]
    6. FC:      → sigmoid → P(crossing)
    """

    def __init__(
        self,
        d_model:  int = 128,
        n_joints: int = 17,
        n_scene:  int = 5,    # nodi scena: ped + veicoli + sem + cross + ego
        n_heads:  int = 4,
        dropout:  float = 0.2,
    ):
        super().__init__()
        self.branch_A = BranchA_GCN(
            in_channels=3, hidden_channels=64,
            out_channels=d_model, n_joints=n_joints
        )
        self.branch_B = BranchB_GAT(
            in_channels=16, hidden_dim=d_model,
            out_channels=d_model
        )
        self.cgam = CGAM(d_model=d_model, n_heads=n_heads, dropout=dropout)

        # Pooling attention-weighted
        self.pool_A = nn.Linear(d_model, 1)
        self.pool_B = nn.Linear(d_model, 1)

        # GRU temporale
        self.gru = nn.GRU(
            input_size=d_model * 2,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 2),  # +1 per ego_speed
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

    def forward(
        self,
        keypoints:     torch.Tensor,           # [B, T, 17, 3]
        scene_nodes:   torch.Tensor,           # [B, T, N_scene, C]
        ego_speed:     torch.Tensor,           # [B, 1]
        adjacency:     Optional[torch.Tensor], # [N_scene, N_scene]
    ) -> torch.Tensor:                         # [B, 2]
        # Branch A e B
        h_A = self.branch_A(keypoints)   # [B, T, 17, d]
        h_B = self.branch_B(scene_nodes, adjacency)  # [B, T, N, d]

        # CGAM — per ogni frame
        h_A, h_B = self.cgam(h_A, h_B)

        # Attention-weighted pooling
        w_A = torch.softmax(self.pool_A(h_A).squeeze(-1), dim=-1)  # [B, T, 17]
        pooled_A = (h_A * w_A.unsqueeze(-1)).sum(dim=2)            # [B, T, d]

        w_B = torch.softmax(self.pool_B(h_B).squeeze(-1), dim=-1)
        pooled_B = (h_B * w_B.unsqueeze(-1)).sum(dim=2)            # [B, T, d]

        # Concatena e passa al GRU
        seq = torch.cat([pooled_A, pooled_B], dim=-1)  # [B, T, 2d]
        gru_out, _ = self.gru(seq)
        context = gru_out[:, -1, :]  # ultimo hidden state

        # Decode
        context = torch.cat([context, ego_speed], dim=-1)
        return self.decoder(context)

