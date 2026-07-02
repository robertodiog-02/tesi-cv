"""
PIE Dataset — wrapper sull'interfaccia ufficiale
=================================================
Versione "geometry + ego (opzionale)".

Ogni sample contiene SEMPRE (poi il modello decide cosa usare in base al config):
    - bbox       : bounding box normalizzata in [0, 1]  ([x1, y1, x2, y2] / [W, H])
    - bbox_delta : differenza frame-by-frame delle bbox normalizzate
    - ego_speed  : velocita ego-vehicle normalizzata, sequenza [T, 1]

NESSUNA informazione sul traffico (semaforo, crosswalk) viene piu usata.
L'uso effettivo di bbox_delta e ego_speed e' deciso a livello di MODELLO
tramite le chiavi di config (use_bbox_delta, use_ego_speed).

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

    Due modalità di windowing:

    1) overlap-based (default, anchor_endpoints=False):
       step = round(obs_len * (1 - overlap)). Lo step dipende da obs_len:
       cambiando obs_len, gli END-POINT delle finestre si spostano.
       Va bene per il TRAIN (augmentation).

    2) end-point-anchored (anchor_endpoints=True):
       le finestre TERMINANO negli stessi punti TTE indipendentemente da
       obs_len, con uno step ASSOLUTO (endpoint_step, default 6 = come il
       benchmark originale a obs_len=16/overlap=0.6). Solo la lunghezza della
       finestra cambia; il punto di valutazione resta fisso.
       Va usato per VAL/TEST per restare confrontabili col benchmark anche
       quando obs_len != 16.

    Ogni sample contiene bbox normalizzata + delta + ego_speed.
    """
    bboxes_all = pie_sequences["bbox"]
    intbin_all = pie_sequences.get("activities") or pie_sequences.get("intention_binary")
    obd_all    = pie_sequences.get("obd_speed", None)
    gps_all    = pie_sequences.get("gps_speed", None)
    pids_all   = pie_sequences.get("pid") or pie_sequences.get("ped_id")
    images_all = pie_sequences.get("image", None)   # per il match pose per frame

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
            # Gli END-POINT sono ancorati ai punti TTE con step assoluto.
            # end-point e = w_end - 1 (ultimo frame osservato).
            # TTE = T - w_end  deve stare in [tte_min, tte_max].
            # => w_end in [T - tte_max, T - tte_min], step = endpoint_step.
            # I punti di valutazione coincidono con quelli del benchmark
            # a obs_len=16 (stesso end_lo/end_hi, stesso step).
            end_hi = T - tte_min          # w_end massimo
            end_lo = T - tte_max          # w_end minimo
            # ancoraggio: il primo end-point parte da end_hi e si scende
            # a passi di endpoint_step (così l'end-point più "tardo" è fisso).
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

            # --- Metadati per il match delle pose (frame number assoluto) ---
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
                "frames":     frame_numbers,   # frame number assoluti della finestra
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
    PyTorch Dataset per PCIP su PIE.

    Ogni item restituisce:
        bbox       : [T, 4]  bbox normalizzata in [0, 1]
        bbox_delta : [T, 4]  delta frame-by-frame
        ego_speed  : [T, 1]  velocita ego normalizzata
        label      : scalar long (0=non-crossing, 1=crossing)

    Quali feature vengano effettivamente usate e' deciso dal modello
    (vedi use_bbox_delta / use_ego_speed nel config).
    """

    def __init__(
        self,
        pie_root:       str,
        split:          str,
        obs_len:        int  = OBS_LEN,
        min_track_size: int  = None,
        overlap:        float = 0.6,
        pose_dir:       str  = None,
        pose_norm:      str  = "reference_point",
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
            sample_type="beh",
            seq_type="crossing",
            min_track_size=min_track_size,
            height_rng=[0, float("inf")],
            squarify_ratio=0,
            data_split_type="default",
        )

        self.samples = build_samples(sequences, obs_len, overlap=overlap,
                                     anchor_endpoints=anchor_endpoints,
                                     endpoint_step=endpoint_step)

        # ---- Pose (opzionale, additivo) -----------------------------------
        self.use_pose = pose_dir is not None
        self.pose_norm = pose_norm
        self._pose_cache = None
        if self.use_pose:
            # split PIE -> set ids (per caricare solo i pkl necessari)
            split_sets = {
                "train": ["set01", "set02", "set04"],
                "val":   ["set05", "set06"],
                "test":  ["set03"],
            }[split]
            from pose_cache import PoseCache
            self._pose_cache = PoseCache(pose_dir, set_ids=split_sets)
            self._check_pose_coverage()

    def __len__(self):
        return len(self.samples)

    def _check_pose_coverage(self):
        """Diagnostica: quante finestre hanno pose complete? Solo report."""
        full = miss = partial = 0
        for s in self.samples:
            if s["frames"] is None or not self._pose_cache.has_track(
                    s["set_id"], s["video_id"], s["ped_id"]):
                miss += 1
                continue
            cov = self._pose_cache.coverage(
                s["set_id"], s["video_id"], s["ped_id"], s["frames"])
            if cov >= 0.999:
                full += 1
            elif cov > 0:
                partial += 1
            else:
                miss += 1
        n = max(len(self.samples), 1)
        print(f"  [Pose] coverage: full={full} ({100*full/n:.1f}%)  "
              f"partial={partial}  missing={miss}")

    def _get_pose(self, s) -> torch.Tensor:
        """Costruisce il tensore pose [T,19,5] per un sample (normalizzato)."""
        from pose_preproc import (derive_19_joints, concat_center,
                                  normalize_pose, fill_missing)
        T = s["bbox"].shape[0]
        if (not self.use_pose) or s["frames"] is None or \
                not self._pose_cache.has_track(s["set_id"], s["video_id"], s["ped_id"]):
            return torch.zeros((T, 19, 5), dtype=torch.float32)

        kp17, _ = self._pose_cache.get_window(
            s["set_id"], s["video_id"], s["ped_id"], s["frames"], fill="nan")
        kp19 = derive_19_joints(kp17)                 # [T,19,3] (+Neck,+CHip)
        # bbox-center in PIXEL (la bbox in s["bbox"] e' normalizzata: la
        # de-normalizziamo per restare nello stesso spazio dei keypoint)
        bbox_px = s["bbox"].copy()
        bbox_px[:, [0, 2]] *= IMG_W
        bbox_px[:, [1, 3]] *= IMG_H
        center = np.stack([(bbox_px[:, 0] + bbox_px[:, 2]) / 2.0,
                           (bbox_px[:, 1] + bbox_px[:, 3]) / 2.0], axis=1)
        bbox_height = bbox_px[:, 3] - bbox_px[:, 1]    # [T] altezza bbox in pixel
        feat = concat_center(kp19, center)            # [T,19,5] pixel
        feat = fill_missing(feat)                     # rimuove NaN
        feat = normalize_pose(feat, method=self.pose_norm,
                              img_w=IMG_W, img_h=IMG_H,
                              bbox_height=bbox_height)
        return torch.from_numpy(feat)

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
            item["keypoints"] = self._get_pose(s)     # [T,17,5]
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
