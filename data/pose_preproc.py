"""
Preprocessing PedGT — derivazione 19 giunti + bbox-center + normalizzazione
===========================================================================
Replica la pipeline di PedGT (Sec. III-B, Tab. II), con lo scheletro a 19
giunti ereditato da PedGNN/PedSynth [5].

Pipeline (decisa e documentata):
  0. Da 17 keypoint COCO/HRNet [T,17,3] si derivano i 2 giunti mancanti
     (Neck, CHip) come da PedGNN Sec. IV-A:
        Neck = media(LShoulder, RShoulder)
        CHip = media(LHip, RHip)
     -> tensore [T,19,3] = (x, y, conf).
  1. bbox-center [T,2] = (cx, cy) in PIXEL.
  2. Replica del center su tutti i 19 giunti e CONCAT -> [T,19,5]
     canali = (x, y, conf, cx, cy).  (PedGT: R^{T x N x 5})
  3. Normalizzazione sull'INTERO tensore [T,19,5] (vedi nota sotto).

NOTA sull'ordine concat/normalizzazione (paper ambiguo):
  Si concatena in PIXEL e poi si normalizza l'intero [T,19,5], cosi'
  coordinate-giunto e centro restano nello stesso spazio. Normalizzare le
  pose lasciando il centro in pixel darebbe canali a scale incompatibili.

Reference-point normalization (best su PIE, Tab. II):
  - si centra ogni coordinata (x,y e cx,cy) sul bbox-center del frame;
  - si divide per la distanza tra le spalle d_s = ||K_lsho - K_rsho||.
  Il canale 'conf' NON viene normalizzato.

Min-max (best su JAAD, PedGNN) supportata per completezza.
"""

import numpy as np

from skeleton import (LEFT_SHOULDER, RIGHT_SHOULDER,
                      LSHO, RSHO, LHIP, RHIP, NECK, CHIP, NUM_JOINTS,
                      get_active_joints)

CONF_CHANNEL = 2  # indice del canale confidence in (x, y, conf, cx, cy)
EPS = 1e-6


def derive_19_joints(kp17: np.ndarray) -> np.ndarray:
    """
    Da [T,17,3] (COCO/HRNet) a [T,19,3] aggiungendo Neck e CHip.

    Neck = media(LShoulder, RShoulder), CHip = media(LHip, RHip).
    La confidence dei giunti derivati = media delle confidence sorgenti
    (cosi' un giunto derivato da giunti incerti resta incerto).
    NaN-safe: se una sorgente e' NaN, il derivato e' NaN (gestito dal fill).
    """
    T = kp17.shape[0]
    out = np.full((T, NUM_JOINTS, 3), np.nan, dtype=np.float32)
    out[:, :17, :] = kp17
    # Neck
    out[:, NECK, :2] = (kp17[:, LSHO, :2] + kp17[:, RSHO, :2]) / 2.0
    out[:, NECK, 2]  = (kp17[:, LSHO, 2]  + kp17[:, RSHO, 2])  / 2.0
    # CHip
    out[:, CHIP, :2] = (kp17[:, LHIP, :2] + kp17[:, RHIP, :2]) / 2.0
    out[:, CHIP, 2]  = (kp17[:, LHIP, 2]  + kp17[:, RHIP, 2])  / 2.0
    return out


def concat_center(keypoints: np.ndarray, center: np.ndarray) -> np.ndarray:
    """
    keypoints : [T, 19, 3]  (x, y, conf) in pixel
    center    : [T, 2]      (cx, cy) in pixel
    return    : [T, 19, 5]  (x, y, conf, cx, cy) in pixel
    """
    T, N, _ = keypoints.shape
    center_rep = np.repeat(center[:, None, :], N, axis=1)   # [T, 19, 2]
    return np.concatenate([keypoints, center_rep], axis=-1)  # [T, 19, 5]


def _shoulder_distance(x, y):
    """d_s = ||K_lsho - K_rsho|| con fallback robusto (estensione verticale)."""
    d_s = np.sqrt((x[LEFT_SHOULDER]-x[RIGHT_SHOULDER])**2 +
                  (y[LEFT_SHOULDER]-y[RIGHT_SHOULDER])**2)
    if not np.isfinite(d_s) or d_s < EPS:
        valid = np.isfinite(y)
        if valid.sum() >= 2:
            d_s = max(y[valid].max() - y[valid].min(), EPS)
        else:
            d_s = 1.0
    return d_s


def reference_point_perframe(feat, img_w=1920.0, img_h=1080.0):
    """
    VARIANTE A0 (ORIGINALE, "rotta") — solo per ablation/confronto.
    Centra OGNI frame sul proprio bbox-center e scala per spalle.
    PROBLEMA: azzera il canale center e cancella la traiettoria temporale.
    Tenuta solo per documentare il fallimento nell'ablation.
    """
    feat = feat.astype(np.float32).copy()
    for t in range(feat.shape[0]):
        x, y = feat[t,:,0], feat[t,:,1]
        cx, cy = feat[t,0,3], feat[t,0,4]
        d_s = _shoulder_distance(x, y)
        feat[t,:,0] = (x - cx) / d_s
        feat[t,:,1] = (y - cy) / d_s
        feat[t,:,3] = (feat[t,:,3] - cx) / d_s   # -> 0
        feat[t,:,4] = (feat[t,:,4] - cy) / d_s   # -> 0
    return feat


def reference_point(feat, img_w=1920.0, img_h=1080.0):
    """
    VARIANTE A (raccomandata) — posa locale scale-invariant + traiettoria.
      keypoints (canali x,y): (coord - bbox_center_frame) / d_s
                              -> posa intra-frame, invariante a scala/posizione
      center    (canali cx,cy): min-max su immagine
                              -> traiettoria ASSOLUTA preservata nel tempo
    Risolve il bug: il movimento globale del pedone resta nel canale center.
    """
    feat = feat.astype(np.float32).copy()
    for t in range(feat.shape[0]):
        x, y = feat[t,:,0], feat[t,:,1]
        cx, cy = feat[t,0,3], feat[t,0,4]
        d_s = _shoulder_distance(x, y)
        feat[t,:,0] = (x - cx) / d_s
        feat[t,:,1] = (y - cy) / d_s
    # traiettoria: center normalizzato min-max (preserva il movimento)
    feat[:,:,3] /= img_w
    feat[:,:,4] /= img_h
    return feat


def reference_point_seq(feat, img_w=1920.0, img_h=1080.0):
    """
    VARIANTE B — riferimento = PRIMO frame della sequenza (non il frame corrente).
      Tutto (keypoint e center) e' espresso relativamente al bbox-center del
      primo frame osservato, scalato per la distanza-spalle media della
      sequenza. Sia posa che traiettoria mantengono il movimento relativo
      all'inizio dell'osservazione.
    """
    feat = feat.astype(np.float32).copy()
    # riferimento: centro del primo frame
    ref_x, ref_y = feat[0,0,3], feat[0,0,4]
    # scala: distanza spalle mediata sui frame validi
    ds_list = []
    for t in range(feat.shape[0]):
        ds_list.append(_shoulder_distance(feat[t,:,0], feat[t,:,1]))
    d_s = float(np.median(ds_list))
    if not np.isfinite(d_s) or d_s < EPS:
        d_s = 1.0
    feat[:,:,0] = (feat[:,:,0] - ref_x) / d_s
    feat[:,:,1] = (feat[:,:,1] - ref_y) / d_s
    feat[:,:,3] = (feat[:,:,3] - ref_x) / d_s
    feat[:,:,4] = (feat[:,:,4] - ref_y) / d_s
    return feat


def minmax_normalize(feat, img_w=1920.0, img_h=1080.0):
    """Min-max su dimensioni immagine (default storico PedGNN, best su JAAD)."""
    feat = feat.astype(np.float32).copy()
    feat[:,:,0] /= img_w
    feat[:,:,1] /= img_h
    feat[:,:,3] /= img_w
    feat[:,:,4] /= img_h
    return feat


def bbox_local_perframe(feat, img_w=1920.0, img_h=1080.0, bbox_px=None):
    """
    VARIANTE BBOX-LOCAL — normalizzazione "classica" da skeleton action-recog.

    Ogni frame e' normalizzato IN MODO INDIPENDENTE rispetto alla PROPRIA
    bounding box: i keypoint vengono mappati nello spazio [0,1] del box.
        x_norm = (x - x1) / w_bbox
        y_norm = (y - y1) / h_bbox
    dove (x1,y1) e' l'angolo alto-sinistra del box e (w,h) le sue dimensioni,
    tutti in PIXEL per quel frame.

    Effetto: la posa e' scale-invariant E position-invariant (un pedone
    lontano/vicino con la stessa posa da' lo stesso tensore). La TRAIETTORIA
    ASSOLUTA viene ELIMINATA di proposito: ogni frame e' agganciato al proprio
    box, non a un riferimento comune. Utile come ablation "sola posa, niente
    dove va il pedone".

    Canali center (cx,cy): sono il centro del box, quindi dopo la mappatura
    cadono a ~(0.5, 0.5) COSTANTE in ogni frame -> informazione quasi nulla.
    Vengono comunque trasformati (restano coerenti con i keypoint); se non li
    vuoi, mettili a 0 dal config con use_center_channels=false.

    Richiede bbox_px: array [T,4] con la bbox in PIXEL come (x1,y1,x2,y2)
    per ogni frame della finestra. Se assente per un frame, fallback a
    scala = distanza spalle e riferimento = bbox-center del frame.
    """
    if bbox_px is None:
        raise ValueError(
            "bbox_local_perframe richiede bbox_px (array [T,4] in pixel, "
            "formato x1,y1,x2,y2). Passalo con "
            "normalize_pose(..., bbox_px=...)."
        )
    feat = feat.astype(np.float32).copy()
    bbox_px = np.asarray(bbox_px, dtype=np.float32)
    for t in range(feat.shape[0]):
        x1, y1, x2, y2 = bbox_px[t]
        w = x2 - x1
        h = y2 - y1
        # fallback robusto se il box e' degenere
        if not np.isfinite(w) or w < EPS:
            w = _shoulder_distance(feat[t, :, 0], feat[t, :, 1])
        if not np.isfinite(h) or h < EPS:
            h = w
        feat[t, :, 0] = (feat[t, :, 0] - x1) / w
        feat[t, :, 1] = (feat[t, :, 1] - y1) / h
        # center (cx,cy) mappato nello stesso spazio del box -> ~0.5,0.5
        feat[t, :, 3] = (feat[t, :, 3] - x1) / w
        feat[t, :, 4] = (feat[t, :, 4] - y1) / h
    return feat


def _torso_length(x, y):
    """
    Lunghezza del torso = distanza Neck <-> CHip. Piu' robusta della distanza
    spalle quando il pedone e' di profilo (le spalle si sovrappongono e la
    loro distanza collassa, mentre il torso resta stabile da ogni angolazione).
    Fallback: distanza spalle, poi estensione verticale, poi 1.0.
    """
    d = np.sqrt((x[NECK] - x[CHIP])**2 + (y[NECK] - y[CHIP])**2)
    if not np.isfinite(d) or d < EPS:
        d = _shoulder_distance(x, y)
    if not np.isfinite(d) or d < EPS:
        valid = np.isfinite(y)
        d = max(y[valid].max() - y[valid].min(), EPS) if valid.sum() >= 2 else 1.0
    return d


def hip_reference(feat, img_w=1920.0, img_h=1080.0):
    """
    VARIANTE HIP — centro = anca (CHip), scala = lunghezza torso (Neck-CHip).

    Differenze rispetto a 'reference_point':
      - punto di riferimento: il BACINO (CHip) invece del bbox-center.
        Il bacino e' il baricentro anatomico stabile: non si sposta quando il
        pedone allarga braccia/gambe (mentre il bbox-center si').
      - scala: lunghezza del TORSO invece della distanza spalle.
        Piu' robusta quando il pedone e' di profilo.

    Pipeline:
      keypoint (x,y): (coord - CHip_frame) / torso_len  -> posa centrata sul
                      bacino, scale-invariant.
      center  (cx,cy): min-max su immagine -> traiettoria ASSOLUTA preservata.
      conf: invariato.
    """
    feat = feat.astype(np.float32).copy()
    for t in range(feat.shape[0]):
        x, y = feat[t, :, 0], feat[t, :, 1]
        ref_x, ref_y = x[CHIP], y[CHIP]
        scale = _torso_length(x, y)
        if not np.isfinite(ref_x) or not np.isfinite(ref_y):
            ref_x, ref_y = feat[t, 0, 3], feat[t, 0, 4]   # fallback bbox-center
        feat[t, :, 0] = (x - ref_x) / scale
        feat[t, :, 1] = (y - ref_y) / scale
    feat[:, :, 3] /= img_w
    feat[:, :, 4] /= img_h
    return feat


def hip_reference_seq(feat, img_w=1920.0, img_h=1080.0, bbox_height=None):
    """
    VARIANTE D — centro = bbox-center del PRIMO frame, scala = altezza bbox.

    Come 'reference_point_seq' (riferimento FISSO al primo frame osservato,
    non ricalcolato ad ogni frame), ma con una differenza:
      - scala: altezza della bounding box al frame 0, invece della distanza
        spalle mediana sulla sequenza. L'altezza bbox viene dal tracker
        (sempre disponibile, non dipende dalla qualita' della pose
        detection), quindi e' una scala piu' affidabile quando la posa e'
        rumorosa o parzialmente occlusa.

    Il punto di riferimento e' il bbox-center del frame 0 (canali cx,cy),
    come in reference_point_seq.

    Sia i keypoint (x,y) sia il canale center (cx,cy) vengono trasformati
    con lo STESSO riferimento fisso, cosi' il movimento del pedone durante
    la finestra osservata resta visibile (stesso principio di
    reference_point_seq):
        coord_norm = (coord - bbox_center_frame0) / bbox_height_frame0

    Richiede bbox_height: array [T] con l'altezza della bbox in PIXEL per
    ogni frame della finestra (calcolata dalla bbox originale, PRIMA della
    normalizzazione in [0,1] — vedi pie_dataset.py:_get_pose). Se il frame 0
    non e' valido, si ripiega sulla mediana delle altezze della finestra.
    """
    if bbox_height is None:
        raise ValueError(
            "hip_reference_seq richiede bbox_height (array [T], altezza "
            "bbox in pixel per frame). Passalo con "
            "normalize_pose(..., bbox_height=...)."
        )
    feat = feat.astype(np.float32).copy()
    bbox_height = np.asarray(bbox_height, dtype=np.float32)

    # riferimento: bbox-center del primo frame (canali cx,cy)
    ref_x, ref_y = feat[0, 0, 3], feat[0, 0, 4]

    scale = float(bbox_height[0]) if len(bbox_height) and np.isfinite(bbox_height[0]) else np.nan
    if not np.isfinite(scale) or scale < EPS:
        valid = np.isfinite(bbox_height) & (bbox_height > EPS)
        scale = float(np.median(bbox_height[valid])) if valid.any() else 1.0

    feat[:, :, 0] = (feat[:, :, 0] - ref_x) / scale
    feat[:, :, 1] = (feat[:, :, 1] - ref_y) / scale
    # center: traiettoria in scala immagine (come reference_point/hip_reference)
    feat[:, :, 3] /= img_w
    feat[:, :, 4] /= img_h
    
    '''
    feat[:, :, 3] = (feat[:, :, 3] - ref_x) / scale
    feat[:, :, 4] = (feat[:, :, 4] - ref_y) / scale
    feat[:, :, 3] = (feat[:, :, 3] - ref_x) 
    feat[:, :, 4] = (feat[:, :, 4] - ref_y) 
    '''
    return feat


_NORM_FUNCS = {
    "reference_point":         reference_point,          # A (raccomandata)
    "reference_point_seq":     reference_point_seq,      # B
    "reference_point_perframe": reference_point_perframe, # A0 (rotta, ablation)
    "hip_reference":           hip_reference,            # C (centro anca, scala torso)
    "hip_reference_seq":       hip_reference_seq,        # D (centro bbox 1° frame, scala altezza bbox)
    "bbox_local":              bbox_local_perframe,      # E (classica frame-per-frame, no traiettoria)
    "minmax":                  minmax_normalize,
}

# Metodi che richiedono l'altezza bbox come input aggiuntivo
_NEEDS_BBOX_HEIGHT = {"hip_reference_seq"}
# Metodi che richiedono la bbox completa in pixel [T,4] (x1,y1,x2,y2)
_NEEDS_BBOX_PX = {"bbox_local"}


def normalize_pose(feat, method="reference_point", img_w=1920.0, img_h=1080.0,
                   bbox_height=None, bbox_px=None):
    if method in ("none", None):
        return feat.astype(np.float32)
    if method not in _NORM_FUNCS:
        raise ValueError(f"Normalizzazione non supportata: {method}. "
                         f"Disponibili: {list(_NORM_FUNCS)} + 'none'")
    if method in _NEEDS_BBOX_HEIGHT:
        return _NORM_FUNCS[method](feat, img_w, img_h, bbox_height=bbox_height)
    if method in _NEEDS_BBOX_PX:
        return _NORM_FUNCS[method](feat, img_w, img_h, bbox_px=bbox_px)
    return _NORM_FUNCS[method](feat, img_w, img_h)


def drop_head_joints(feat: np.ndarray) -> np.ndarray:
    """
    Rimuove i 5 nodi testa (naso, occhi, orecchie) dal tensore pose.

    IMPORTANTE: va chiamata DOPO derive_19_joints, concat_center,
    fill_missing e normalize_pose. La derivazione di Neck/CHip e la
    reference-point normalization usano le SPALLE (e volendo le anche), che
    NON sono nodi testa e quindi restano. Cosi' l'informazione anatomica per
    normalizzare non viene persa: si taglia solo alla fine.

    feat : [T, N, C] con N=19 (ordine giunti standard).
    out  : [T, 14, C] con i nodi nell'ordine compatto di get_active_joints
           (il giunto piu' alto rimasto e' il Neck).
    """
    active, _ = get_active_joints(exclude_head=True)
    return np.ascontiguousarray(feat[:, active, :])


def fill_missing(feat: np.ndarray) -> np.ndarray:
    """
    Sostituisce eventuali NaN (frame senza posa) prima del modello.
    Strategia: forward-fill lungo il tempo, poi backward-fill, infine 0.
    Conserva la continuita' temporale meglio dello zero secco.
    """
    feat = feat.copy()
    T = feat.shape[0]
    # forward fill
    for t in range(1, T):
        nan_mask = ~np.isfinite(feat[t])
        feat[t][nan_mask] = feat[t - 1][nan_mask]
    # backward fill
    for t in range(T - 2, -1, -1):
        nan_mask = ~np.isfinite(feat[t])
        feat[t][nan_mask] = feat[t + 1][nan_mask]
    feat[~np.isfinite(feat)] = 0.0
    return feat.astype(np.float32)
