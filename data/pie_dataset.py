"""
PIE Dataset — wrapper sull'interfaccia ufficiale
=================================================
Versione MULTI-STREAM.

Ogni sample contiene SEMPRE le feature grezze; poi il MODELLO decide quali
stream usare in base al config (sezione `streams`):

    - keypoints  : [T, 19, C]  pose GCN. C = 5 (x,y,conf,cx,cy) oppure 3
                   (x,y,conf) se use_center_channels=False.
    - kinematics : [T, 5]      cinematica pura: 4 feature bbox + ego_speed.
                   bbox in variante 'xyxy' (x1,y1,x2,y2) o 'cxcywh'
                   (cx,cy,w,h), entrambe normalizzate in [0,1]. ego / 120 km/h.
    - crop_feat  : [T, 768]    feature visuale ConvNeXt del crop del pedone.

Feature legacy ancora prodotte per compatibilita':
    - bbox       : bbox normalizzata in [0,1]  ([x1,y1,x2,y2] / [W,H])
    - bbox_delta : differenza frame-by-frame delle bbox normalizzate
    - ego_speed  : velocita ego-vehicle normalizzata, sequenza [T, 1]

Protocollo Kotseruba WACV 2021:
  - min_track_size = obs_len + 60 = 76
  - TTE [30, 60] frame prima del crossing_point
  - overlap = 0.6 -> step = 6 frame
  - Split: train=set01/02/04, val=set05/06, test=set03
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

OBS_LEN = 16
TTE_MIN = 30   # frame (~1s)
TTE_MAX = 60   # frame (~2s)
IMG_W   = 1920.0
IMG_H   = 1080.0
SPEED_NORM = 120.0   # km/h, fattore di normalizzazione ego-speed

# split PIE -> set ids (per caricare solo i pkl necessari di pose/crop)
SPLIT_SETS = {
    "train": ["set01", "set02", "set04"],
    "val":   ["set05", "set06"],
    "test":  ["set03"],
}


def _frame_num_from_path(path: str) -> int:
    """Estrae il frame number da '.../video_0001/01053.png' -> 1053."""
    stem = Path(str(path)).stem
    return int(stem)


def _set_video_from_path(path: str):
    """Estrae (set_id, video_id) da un image path PIE."""
    set_id = video_id = None
    for part in Path(str(path)).parts:
        if part.startswith("set"):
            set_id = part
        elif part.startswith("video_"):
            video_id = part
    return set_id, video_id


def get_pie_interface(pie_root: str):
    """Inizializza pie_data.py cercandolo in piu posizioni."""
    for p in [
        Path(pie_root) / "utilities",
        Path(__file__).parent,
    ]:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

    try:
        from pie_data import PIE
    except ImportError:
        raise ImportError(
            "pie_data.py non trovato.\n"
            "Copialo con: cp /path/to/PIE/utilities/pie_data.py data/"
        )

    pie_data_root = Path(pie_root) / "annotations"
    if not (pie_data_root / "annotations").exists():
        pie_data_root = Path(pie_root)

    print(f"PIE data root: {pie_data_root}")
    return PIE(data_path=str(pie_data_root))


def _bbox_to_kinematics(bboxes_norm: np.ndarray, bbox_format: str) -> np.ndarray:
    """
    bboxes_norm : [T, 4] bbox normalizzata in [0,1] come (x1,y1,x2,y2).
    bbox_format : 'xyxy' -> (x1,y1,x2,y2) invariato
                  'cxcywh' -> (cx,cy,w,h), tutti in [0,1]
    return      : [T, 4] float32
    """
    if bbox_format == "xyxy":
        return bboxes_norm.astype(np.float32).copy()
    elif bbox_format == "cxcywh":
        x1, y1, x2, y2 = (bboxes_norm[:, 0], bboxes_norm[:, 1],
                          bboxes_norm[:, 2], bboxes_norm[:, 3])
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w  = (x2 - x1)
        h  = (y2 - y1)
        return np.stack([cx, cy, w, h], axis=1).astype(np.float32)
    else:
        raise ValueError(f"bbox_format non valido: {bbox_format} "
                         f"(usa 'xyxy' o 'cxcywh')")


def build_samples(
    pie_sequences: Dict,
    obs_len:  int   = OBS_LEN,
    overlap:  float = 0.6,
    tte_min:  int   = TTE_MIN,
    tte_max:  int   = TTE_MAX,
    anchor_endpoints: bool = False,
    endpoint_step:    int  = 6,
) -> List[Dict]:
    """
    Costruisce i sample seguendo il protocollo Kotseruba WACV 2021.

    Due modalità di windowing (invariate rispetto alla versione precedente):

    1) overlap-based (default, anchor_endpoints=False):
       step = round(obs_len * (1 - overlap)). Va bene per il TRAIN.

    2) end-point-anchored (anchor_endpoints=True):
       le finestre TERMINANO negli stessi punti TTE con step ASSOLUTO
       (endpoint_step). Da usare per VAL/TEST per restare confrontabili col
       benchmark anche quando obs_len != 16.

    Ogni sample contiene la bbox normalizzata grezza + delta + ego_speed +
    i metadati per il match di pose e crop (frame number assoluti).
    """
    bboxes_all = pie_sequences["bbox"]
    intbin_all = pie_sequences.get("activities") or pie_sequences.get("intention_binary")
    obd_all    = pie_sequences.get("obd_speed", None)
    gps_all    = pie_sequences.get("gps_speed", None)
    pids_all   = pie_sequences.get("pid") or pie_sequences.get("ped_id")
    images_all = pie_sequences.get("image", None)   # per il match pose/crop per frame

    # -- Distribuzione raw ----------------------------------------------
    all_labels    = [int(intbin_all[i][0][0]) for i in range(len(bboxes_all))]
    n_cross_raw   = sum(l == 1 for l in all_labels)
    n_nocross_raw = sum(l == 0 for l in all_labels)
    n_total_raw   = len(all_labels)
    print(f"\n[INFO] Distribuzione tracks nel dataset raw:")
    print(f"  Crossing:     {n_cross_raw:4d}  ({100*n_cross_raw/n_total_raw:.1f}%)")
    print(f"  Non-crossing: {n_nocross_raw:4d}  ({100*n_nocross_raw/n_total_raw:.1f}%)")
    print(f"  Totale:       {n_total_raw:4d}")
    print(f"  Ratio NC/C:   {n_nocross_raw/max(n_cross_raw,1):.2f}:1")
    print(f"  ego-speed disponibile: "
          f"{'SI' if (obd_all is not None and gps_all is not None) else 'NO'}")

    step = max(1, round(obs_len * (1.0 - overlap)))

    # -- Costruzione sample ---------------------------------------------
    samples = []

    for i in range(len(bboxes_all)):
        track_bboxes = bboxes_all[i]
        T      = len(track_bboxes)
        label  = int(intbin_all[i][0][0])
        ped_id = pids_all[i][0][0] if pids_all[i] else f"ped_{i}"

        if T < obs_len + tte_min:
            continue

        obd_track = obd_all[i] if obd_all is not None else None
        gps_track = gps_all[i] if gps_all is not None else None
        img_track = images_all[i] if images_all is not None else None

        # -- Determinazione degli START in base alla modalità ------------
        if anchor_endpoints:
            end_hi = T - tte_min
            end_lo = T - tte_max
            w_ends = list(range(end_hi, end_lo - 1, -endpoint_step))
            w_ends = [we for we in w_ends if we - obs_len >= 0]
            starts = [we - obs_len for we in sorted(w_ends)]
        else:
            start_idx = max(0, T - obs_len - tte_max)
            end_idx   = T - obs_len - tte_min
            if end_idx < 0 or end_idx < start_idx:
                continue
            starts = list(range(start_idx, end_idx + 1, step))

        for w_start in starts:
            w_end = w_start + obs_len

            obs_bboxes = np.array(track_bboxes[w_start:w_end], dtype=np.float32)
            if len(obs_bboxes) < obs_len:
                continue

            # Bbox normalizzata in [0, 1] + delta frame-by-frame
            bboxes_norm = obs_bboxes.copy()
            bboxes_norm[:, [0, 2]] /= IMG_W
            bboxes_norm[:, [1, 3]] /= IMG_H
            bbox_delta = np.zeros_like(bboxes_norm)
            bbox_delta[1:] = bboxes_norm[1:] - bboxes_norm[:-1]

            # Ego speed (sequenza), normalizzata. Se non disponibile -> zeri.
            ego_speed_seq = np.zeros((obs_len, 1), dtype=np.float32)
            if obd_track is not None and gps_track is not None:
                obd_slice = obd_track[w_start:w_end]
                gps_slice = gps_track[w_start:w_end]
                for j in range(min(obs_len, len(obd_slice))):
                    ov = obd_slice[j][0] if isinstance(obd_slice[j], list) else obd_slice[j]
                    gv = gps_slice[j][0] if isinstance(gps_slice[j], list) else gps_slice[j]
                    ego_speed_seq[j, 0] = (float(ov) + float(gv)) / (2.0 * SPEED_NORM)

            # --- Metadati per il match di pose/crop (frame number assoluto) ---
            set_id = video_id = None
            frame_numbers = None
            if img_track is not None:
                img_slice = img_track[w_start:w_end]
                frame_numbers = [_frame_num_from_path(p) for p in img_slice]
                set_id, video_id = _set_video_from_path(img_slice[0])

            samples.append({
                "ped_id":     ped_id,
                "w_start":    w_start,
                "tte":        T - w_end,
                "bbox":       bboxes_norm.astype(np.float32),
                "bbox_delta": bbox_delta.astype(np.float32),
                "ego_speed":  ego_speed_seq.astype(np.float32),
                "label":      np.int64(label),
                "set_id":     set_id,
                "video_id":   video_id,
                "frames":     frame_numbers,
            })

    n_pos = sum(s["label"] == 1 for s in samples)
    n_neg = sum(s["label"] == 0 for s in samples)
    if anchor_endpoints:
        print(f"\n  tracks: {len(bboxes_all)}  ->  samples: {len(samples)} "
              f"(end-point anchored, step={endpoint_step}, obs_len={obs_len})")
    else:
        print(f"\n  tracks: {len(bboxes_all)}  ->  samples: {len(samples)} "
              f"(step={step}, overlap={overlap})")
    print(f"  crossing: {n_pos}, non-crossing: {n_neg} "
          f"(ratio {n_pos/max(n_neg,1):.2f}:1)")
    return samples


class PIEDataset(Dataset):
    """
    PyTorch Dataset multi-stream per PCIP su PIE.

    Ogni item restituisce (le chiavi presenti dipendono dagli stream attivi):
        keypoints  : [T, 19, C]  pose (se use_pose).  C=5 o 3 (use_center_channels)
        kinematics : [T, 5]      bbox (variante) + ego  (se use_kinematics)
        crop_feat  : [T, 768]    feature ConvNeXt        (se use_crop)
        bbox       : [T, 4]      legacy, sempre presente
        bbox_delta : [T, 4]      legacy
        ego_speed  : [T, 1]      legacy
        label      : scalar long (0=non-crossing, 1=crossing)

    Quali stream vengano usati dal modello e' deciso dal config (streams.*).
    """

    def __init__(
        self,
        pie_root:       str,
        split:          str,
        obs_len:        int  = OBS_LEN,
        min_track_size: int  = None,
        overlap:        float = 0.6,
        # --- pose stream ---
        pose_dir:       str  = None,
        pose_norm:      str  = "reference_point",
        use_center_channels: bool = True,
        use_confidence: bool = True,    # False -> rimuove il canale conf dai giunti
        exclude_head:   bool = False,   # rimuove i 5 nodi testa -> 14 giunti
        # --- kinematics stream ---
        use_kinematics: bool = False,
        bbox_format:    str  = "xyxy",
        # --- crop stream ---
        crop_dir:       str  = None,
        # --- windowing ---
        anchor_endpoints: bool = False,
        endpoint_step:    int  = 6,
    ):
        assert split in ("train", "val", "test")
        print(f"\n=== PIEDataset [{split}] ===")

        if min_track_size is None:
            min_track_size = obs_len + TTE_MAX
        print(f"min_track_size: {min_track_size}  "
              f"(= obs_len {obs_len} + tte_max {TTE_MAX})")

        pie = get_pie_interface(pie_root)

        print(f"Generando sequenze [{split}]...")
        sequences = pie.generate_data_trajectory_sequence(
            split,
            fstride=1,
            sample_type="all",
            seq_type="crossing",
            min_track_size=min_track_size,
            height_rng=[0, float("inf")],
            squarify_ratio=0,
            data_split_type="default",
        )

        self.samples = build_samples(sequences, obs_len, overlap=overlap,
                                     anchor_endpoints=anchor_endpoints,
                                     endpoint_step=endpoint_step)

        split_sets = SPLIT_SETS[split]

        # ---- Pose stream (opzionale) --------------------------------------
        self.use_pose = pose_dir is not None
        self.pose_norm = pose_norm
        self.use_center_channels = use_center_channels
        self.use_confidence = use_confidence
        self.exclude_head = exclude_head
        self._pose_cache = None
        if self.use_pose and not use_confidence:
            print("  [Pose] use_confidence=False -> canale conf rimosso dai giunti")
        if self.use_pose and exclude_head:
            print("  [Pose] exclude_head=True -> 14 giunti (testa rimossa, "
                  "Neck in cima)")
        if self.use_pose:
            from pose_cache import PoseCache
            self._pose_cache = PoseCache(pose_dir, set_ids=split_sets)
            self._check_coverage(self._pose_cache, "Pose")

        # ---- Kinematics stream (opzionale) --------------------------------
        self.use_kinematics = use_kinematics
        self.bbox_format = bbox_format
        if self.use_kinematics:
            assert bbox_format in ("xyxy", "cxcywh"), \
                f"bbox_format non valido: {bbox_format}"
            print(f"  [Kinematics] attivo — bbox_format={bbox_format}, "
                  f"+ ego_speed -> 5 feature")

        # ---- Crop stream (opzionale) --------------------------------------
        self.use_crop = crop_dir is not None
        self._crop_cache = None
        if self.use_crop:
            from crop_cache import CropCache
            self._crop_cache = CropCache(crop_dir, set_ids=split_sets)
            self._check_coverage(self._crop_cache, "Crop")

    def __len__(self):
        return len(self.samples)

    def _check_coverage(self, cache, name: str):
        """Diagnostica: quante finestre hanno feature complete? Solo report."""
        full = miss = partial = 0
        for s in self.samples:
            if s["frames"] is None or not cache.has_track(
                    s["set_id"], s["video_id"], s["ped_id"]):
                miss += 1
                continue
            cov = cache.coverage(
                s["set_id"], s["video_id"], s["ped_id"], s["frames"])
            if cov >= 0.999:
                full += 1
            elif cov > 0:
                partial += 1
            else:
                miss += 1
        n = max(len(self.samples), 1)
        print(f"  [{name}] coverage: full={full} ({100*full/n:.1f}%)  "
              f"partial={partial}  missing={miss}")

    # ---------------------------------------------------------------- pose
    def _get_pose(self, s) -> torch.Tensor:
        """Costruisce il tensore pose [T,19,C] per un sample (normalizzato).
        C = 5 (x,y,conf,cx,cy) oppure 3 (x,y,conf) se use_center_channels=False."""
        from pose_preproc import (derive_19_joints, concat_center,
                                  normalize_pose, fill_missing,
                                  drop_head_joints)
        T = s["bbox"].shape[0]
        # canali attivi: (x,y) sempre; conf se use_confidence; (cx,cy) se center
        C = 2
        if self.use_confidence:
            C += 1
        if self.use_center_channels:
            C += 2
        N = 14 if self.exclude_head else 19
        if (not self.use_pose) or s["frames"] is None or \
                not self._pose_cache.has_track(s["set_id"], s["video_id"], s["ped_id"]):
            return torch.zeros((T, N, C), dtype=torch.float32)

        kp17, _ = self._pose_cache.get_window(
            s["set_id"], s["video_id"], s["ped_id"], s["frames"], fill="nan")
        kp19 = derive_19_joints(kp17)                 # [T,19,3]
        bbox_px = s["bbox"].copy()
        bbox_px[:, [0, 2]] *= IMG_W
        bbox_px[:, [1, 3]] *= IMG_H
        center = np.stack([(bbox_px[:, 0] + bbox_px[:, 2]) / 2.0,
                           (bbox_px[:, 1] + bbox_px[:, 3]) / 2.0], axis=1)
        bbox_height = bbox_px[:, 3] - bbox_px[:, 1]
        feat = concat_center(kp19, center)            # [T,19,5] pixel
        feat = fill_missing(feat)
        feat = normalize_pose(feat, method=self.pose_norm,
                              img_w=IMG_W, img_h=IMG_H,
                              bbox_height=bbox_height,
                              bbox_px=bbox_px)             # [T,19,5]
        if self.exclude_head:
            feat = drop_head_joints(feat)               # [T,14,5]
        # selezione canali dal layout normalizzato (x,y,conf,cx,cy):
        #   x,y sempre; conf solo se use_confidence; cx,cy solo se center.
        keep = [0, 1]                                   # x, y
        if self.use_confidence:
            keep.append(2)                              # conf
        if self.use_center_channels:
            keep += [3, 4]                              # cx, cy
        feat = feat[:, :, keep]                         # [T, N, C]
        return torch.from_numpy(np.ascontiguousarray(feat))

    # ----------------------------------------------------------- kinematics
    def _get_kinematics(self, s) -> torch.Tensor:
        """[T,5] = 4 feature bbox (variante) + ego_speed."""
        bbox_kin = _bbox_to_kinematics(s["bbox"], self.bbox_format)   # [T,4]
        kin = np.concatenate([bbox_kin, s["ego_speed"]], axis=1)      # [T,5]
        return torch.from_numpy(kin.astype(np.float32))

    # ----------------------------------------------------------------- crop
    def _get_crop(self, s) -> torch.Tensor:
        """[T,768] feature ConvNeXt. fill dell'ultimo frame valido se buchi
        (non dovrebbero esistere), con warning."""
        T = s["bbox"].shape[0]
        if (not self.use_crop) or s["frames"] is None or \
                not self._crop_cache.has_track(s["set_id"], s["video_id"], s["ped_id"]):
            print(f"[WARN] crop mancante per track "
                  f"{s['set_id']}/{s['video_id']}/{s['ped_id']} -> zeri")
            return torch.zeros((T, 768), dtype=torch.float32)

        feats, mask = self._crop_cache.get_window(
            s["set_id"], s["video_id"], s["ped_id"], s["frames"], fill="nan")
        if not mask.all():
            print(f"[WARN] crop buchi ({int((~mask).sum())}/{T}) per "
                  f"{s['set_id']}/{s['video_id']}/{s['ped_id']} -> fill")
            feats = _fill_missing_seq(feats)
        return torch.from_numpy(np.ascontiguousarray(feats.astype(np.float32)))

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        item = {
            "bbox":       torch.from_numpy(s["bbox"]),
            "bbox_delta": torch.from_numpy(s["bbox_delta"]),
            "ego_speed":  torch.from_numpy(s["ego_speed"]),
            "label":      torch.tensor(s["label"], dtype=torch.long),
            "ped_id":     s["ped_id"],
        }
        if self.use_pose:
            item["keypoints"] = self._get_pose(s)         # [T,19,C]
        if self.use_kinematics:
            item["kinematics"] = self._get_kinematics(s)  # [T,5]
        if self.use_crop:
            item["crop_feat"] = self._get_crop(s)         # [T,768]
        return item

    def get_class_weights(self) -> torch.Tensor:
        labels  = np.array([s["label"] for s in self.samples])
        n_pos   = (labels == 1).sum()
        n_neg   = (labels == 0).sum()
        n_total = len(labels)
        w_pos = n_total / (2.0 * n_pos) if n_pos > 0 else 1.0
        w_neg = n_total / (2.0 * n_neg) if n_neg > 0 else 1.0
        print(f"  Class weights — crossing: {w_pos:.3f}, "
              f"non-crossing: {w_neg:.3f}")
        return torch.tensor([w_neg, w_pos], dtype=torch.float32)


def _fill_missing_seq(feat: np.ndarray) -> np.ndarray:
    """forward-fill poi backward-fill lungo il tempo su [T, D] (o [T,...])."""
    feat = feat.copy()
    T = feat.shape[0]
    for t in range(1, T):
        m = ~np.isfinite(feat[t])
        feat[t][m] = feat[t - 1][m]
    for t in range(T - 2, -1, -1):
        m = ~np.isfinite(feat[t])
        feat[t][m] = feat[t + 1][m]
    feat[~np.isfinite(feat)] = 0.0
    return feat
