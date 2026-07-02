"""
Scheletro a 19 giunti per la GCN di PedGT
=========================================
PedGT eredita "the graph structure outlined in [5]" = PedGNN/PedSynth
(Riaz et al., 2024), che pubblica lo scheletro esatto nella sua Fig. 4:
19 giunti connessi come grafo non orientato.

I 19 giunti = 17 di COCO/AlphaPose + 2 derivati:
    17 Neck  = media(LShoulder, RShoulder)
    18 CHip  = media(LHip, RHip)
come specificato dal paper PedGNN (Sec. IV-A): "AlphaPose does not provide
the Neck and CHip joints. To compute the Neck coordinates we average
LShoulder and RShoulder [...] CHip [...] average LHip and RHip."

Indici (0..16 = COCO-17, poi Neck, CHip):
    0  nose
    1  left_eye      2  right_eye
    3  left_ear      4  right_ear
    5  left_shoulder 6  right_shoulder
    7  left_elbow    8  right_elbow
    9  left_wrist    10 right_wrist
    11 left_hip      12 right_hip
    13 left_knee     14 right_knee
    15 left_ankle    16 right_ankle
    17 Neck          18 CHip

Edge list da Fig. 4: Neck e CHip fanno da snodi del tronco.

Reference-point normalization (PedGT, Tab. II, best su PIE):
    d_s = || K_LShoulder - K_RShoulder ||  -> indici 5 e 6.
"""

import numpy as np
import torch

NUM_JOINTS = 19

JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "Neck", "CHip",
]

# Indici di comodo
(NOSE, LEYE, REYE, LEAR, REAR, LSHO, RSHO, LELB, RELB,
 LWRI, RWRI, LHIP, RHIP, LKNE, RKNE, LANK, RANK, NECK, CHIP) = range(19)

# Indici per la reference-point normalization (spalle)
LEFT_SHOULDER = LSHO    # 5
RIGHT_SHOULDER = RSHO   # 6

# Edge list (non orientata) da PedGNN Fig. 4.
JOINT_EDGES = [
    # testa
    (REAR, REYE), (REYE, NOSE), (NOSE, LEYE), (LEYE, LEAR),
    (NOSE, NECK),
    # tronco superiore: Neck snodo tra le spalle
    (RSHO, NECK), (NECK, LSHO),
    # braccia
    (RSHO, RELB), (RELB, RWRI),
    (LSHO, LELB), (LELB, LWRI),
    # colonna: Neck <-> CHip
    (NECK, CHIP),
    # bacino: CHip snodo tra le anche
    (RHIP, CHIP), (CHIP, LHIP),
    # gambe
    (RHIP, RKNE), (RKNE, RANK),
    (LHIP, LKNE), (LKNE, LANK),
]


def build_adjacency(num_joints: int = NUM_JOINTS,
                    self_loops: bool = True,
                    normalize: bool = True) -> np.ndarray:
    """Matrice di adiacenza dello scheletro a 19 giunti."""
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in JOINT_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    if self_loops:
        A += np.eye(num_joints, dtype=np.float32)
    if normalize:
        deg = A.sum(axis=1)
        d_inv_sqrt = np.zeros_like(deg)
        nz = deg > 0
        d_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
        A = np.diag(d_inv_sqrt) @ A @ np.diag(d_inv_sqrt)
    return A.astype(np.float32)


def build_edge_index(self_loops: bool = True) -> torch.Tensor:
    """edge_index [2, E] per PyG (grafo non orientato -> doppia direzione)."""
    edges = []
    for i, j in JOINT_EDGES:
        edges.append((i, j))
        edges.append((j, i))
    if self_loops:
        for i in range(NUM_JOINTS):
            edges.append((i, i))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


if __name__ == "__main__":
    A = build_adjacency()
    print("Adjacency:", A.shape, "sum:", round(float(A.sum()), 2))
    ei = build_edge_index()
    print("edge_index:", ei.shape, f"({ei.shape[1]} directed edges)")
    deg = build_adjacency(normalize=False, self_loops=False).sum(1)
    print("degrees:", deg.astype(int).tolist())
    assert (deg > 0).all(), "Nodo isolato!"
    print("OK")
