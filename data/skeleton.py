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

# --------------------------------------------------------------------------
# Gruppi di nodi e edge opzionali (per le ablation configurabili)
# --------------------------------------------------------------------------
# Nodi "testa": naso, occhi, orecchie. Il Neck NON e' testa (e' snodo del
# tronco) e resta sempre. Escludendo questi 5 nodi il giunto piu' alto
# diventa il Neck (indice 17).
HEAD_JOINTS = [NOSE, LEYE, REYE, LEAR, REAR]   # 0,1,2,3,4

# Edge extra "cross-limb": gomito <-> ginocchio controlaterale.
# Collega la catena braccio a quella della gamba opposta, dando alla GCN
# un percorso diretto tra arti che nello scheletro anatomico standard
# comunicano solo passando per Neck->CHip.
CROSS_LIMB_EDGES = [
    (RELB, LKNE),   # gomito destro  <-> ginocchio sinistro
    (LELB, RKNE),   # gomito sinistro <-> ginocchio destro
]


def get_active_joints(exclude_head: bool = False):
    """
    Ritorna (active_indices, old2new) per il sotto-grafo richiesto.

    active_indices : lista degli indici (nel sistema 0..18 originale) dei nodi
                     tenuti, in ordine crescente.
    old2new        : dict indice_originale -> nuovo indice compatto [0..K-1].

    Con exclude_head=False ritorna tutti i 19 giunti (identita').
    Con exclude_head=True rimuove i 5 nodi testa -> 14 giunti, Neck in cima.
    """
    if exclude_head:
        drop = set(HEAD_JOINTS)
        active = [j for j in range(NUM_JOINTS) if j not in drop]
    else:
        active = list(range(NUM_JOINTS))
    old2new = {old: new for new, old in enumerate(active)}
    return active, old2new


def build_joint_edges(exclude_head: bool = False,
                      add_cross_limb: bool = False):
    """
    Costruisce la edge-list (in indici COMPATTI del sotto-grafo) applicando
    le opzioni di ablation:
      - exclude_head   : rimuove i 5 nodi testa (edge che li toccano scartati)
      - add_cross_limb : aggiunge gli edge gomito<->ginocchio controlaterale

    Ritorna: (edges_compatti, num_nodi_attivi).
    """
    active, old2new = get_active_joints(exclude_head)
    active_set = set(active)

    edges = list(JOINT_EDGES)
    if add_cross_limb:
        edges = edges + list(CROSS_LIMB_EDGES)

    remapped = []
    for i, j in edges:
        if i in active_set and j in active_set:
            remapped.append((old2new[i], old2new[j]))
    return remapped, len(active)


def build_adjacency(num_joints: int = NUM_JOINTS,
                    self_loops: bool = True,
                    normalize: bool = True,
                    exclude_head: bool = False,
                    add_cross_limb: bool = False) -> np.ndarray:
    """Matrice di adiacenza dello scheletro (default: 19 giunti).

    exclude_head / add_cross_limb: vedi build_joint_edges. Quando una di
    queste e' attiva, num_joints viene ricalcolato dal sotto-grafo e il
    valore passato come argomento e' ignorato.
    """
    if exclude_head or add_cross_limb:
        edges, num_joints = build_joint_edges(exclude_head, add_cross_limb)
    else:
        edges = JOINT_EDGES
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    if self_loops:
        A += np.eye(num_joints, dtype=np.float32)
    if normalize:
        deg = A.sum(axis=1)
        # Protezione contro nodi isolati (grado 0 -> 1/sqrt(0) = inf).
        if (deg <= 0).any():
            isolated = np.where(deg <= 0)[0].tolist()
            raise ValueError(
                f"Nodi isolati (grado 0) in build_adjacency: {isolated}. "
                f"Controlla JOINT_EDGES o abilita self_loops=True.")
        d_inv_sqrt = np.zeros_like(deg)
        nz = deg > 0
        d_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
        # D^{-1/2} A D^{-1/2} con D diagonale == A * outer(d_inv_sqrt, d_inv_sqrt).
        # Usiamo il prodotto esterno (broadcasting) invece di due np.diag(...) @ A:
        # e' matematicamente identico ma NON attraversa il percorso 'matmul' di
        # numpy, che in versioni recenti (>=2.2) emette RuntimeWarning spuri
        # 'divide by zero / overflow in matmul' su matrici diagonali, anche
        # quando il risultato e' corretto. Cosi' l'output resta pulito.
        A = A * np.outer(d_inv_sqrt, d_inv_sqrt)
    return A.astype(np.float32)


def build_edge_index(self_loops: bool = True,
                     exclude_head: bool = False,
                     add_cross_limb: bool = False) -> torch.Tensor:
    """edge_index [2, E] per PyG (grafo non orientato -> doppia direzione).

    exclude_head / add_cross_limb: vedi build_joint_edges. Gli indici
    restituiti sono nel sistema COMPATTO del sotto-grafo (0..K-1).
    """
    if exclude_head or add_cross_limb:
        base_edges, n_nodes = build_joint_edges(exclude_head, add_cross_limb)
    else:
        base_edges, n_nodes = JOINT_EDGES, NUM_JOINTS
    edges = []
    for i, j in base_edges:
        edges.append((i, j))
        edges.append((j, i))
    if self_loops:
        for i in range(n_nodes):
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
