"""
Crop-feature cache loader — match per FRAME NUMBER ASSOLUTO
==========================================================
Gemello di pose_cache.py, ma per le feature visuali ConvNeXt (768-dim)
estratte dai crop del pedone.

Carica i pkl `pie_convnext_features_setXX.pkl` e fornisce, per ogni
(set, video, pid), una lookup frame_number -> feature[768].

PERCHE' IL MATCH PER FRAME NUMBER (come per le pose)
----------------------------------------------------
`pie_data.generate_data_trajectory_sequence` (seq_type=crossing) taglia la
track fino al crossing_point, applica fstride e puo' scartare frame con
l'height-check. La lista di bbox risultante NON ha piu' il frame number
esplicito, quindi propaghiamo il frame number in ogni sample (vedi
pie_dataset.py) e qui facciamo il match per frame id. Agganciare per
posizione-in-lista produrrebbe disallineamenti silenziosi.

Struttura pkl attesa (verificata sul file di esempio):
    { 'setXX': { 'video_YYYY': { '<pid>': {
          <frame_number:int>: {'visual_features': np.ndarray[768] float32},
          ...
    }}}}

Il `pid` e' nel formato PIE '<set>_<video>_<obj>' (es. '5_1_1731'),
identico a quello restituito da pie_data e usato dal PoseCache.
"""

import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

FEAT_DIM = 768
FEAT_KEY = "visual_features"


class CropCache:
    """
    Lookup feature-crop per (set, video, pid, frame_number).

    Uso tipico:
        cache = CropCache("data/crops", set_ids=["set05", "set06"])
        feats, mask = cache.get_window(set_id, video_id, pid, frame_numbers)
        # feats -> [len(frame_numbers), 768], mask -> [T] bool
    """

    def __init__(self, crop_dir: str, set_ids=None, verbose: bool = True,
                 file_pattern: str = "pie_convnext_features_set{num}.pkl"):
        self.crop_dir = Path(crop_dir)
        self.verbose = verbose
        self.file_pattern = file_pattern
        # index[(set, video, pid)] = {frame_number: row_idx}
        self._frame_index: Dict[Tuple[str, str, str], Dict[int, int]] = {}
        # store[(set, video, pid)] = feats[T, 768]
        self._store: Dict[Tuple[str, str, str], np.ndarray] = {}
        self._n_tracks = 0
        self._load(set_ids)

    # ------------------------------------------------------------------ load
    def _candidate_files(self, set_ids):
        if set_ids is None:
            files = sorted(self.crop_dir.glob(
                self.file_pattern.replace("{num}", "*")))
        else:
            files = []
            for sid in set_ids:
                num = str(sid).replace("set", "").zfill(2)
                f = self.crop_dir / self.file_pattern.format(num=num)
                if f.exists():
                    files.append(f)
        return files

    def _load(self, set_ids):
        files = self._candidate_files(set_ids)
        if not files:
            raise FileNotFoundError(
                f"Nessun pkl crop trovato in {self.crop_dir} "
                f"(pattern: {self.file_pattern})"
            )
        for f in files:
            with open(f, "rb") as fh:
                data = pickle.load(fh)
            for set_id, videos in data.items():
                for video_id, peds in videos.items():
                    for pid, frame_dict in peds.items():
                        # frame_dict: {frame_number: {'visual_features': [768]}}
                        frames = sorted(int(fn) for fn in frame_dict.keys())
                        feats = np.stack([
                            np.asarray(frame_dict[fn][FEAT_KEY],
                                       dtype=np.float32)
                            for fn in frames
                        ], axis=0)                       # [T, 768]
                        assert feats.shape[1] == FEAT_DIM, \
                            f"feature dim inattesa {feats.shape} per {pid}"
                        key = (str(set_id), str(video_id), str(pid))
                        self._store[key] = feats
                        self._frame_index[key] = {
                            int(fn): i for i, fn in enumerate(frames)
                        }
                        self._n_tracks += 1
            if self.verbose:
                print(f"[CropCache] caricato {f.name}")
        if self.verbose:
            print(f"[CropCache] tracks totali: {self._n_tracks}")

    # ------------------------------------------------------------- normalize
    @staticmethod
    def _normalize_pid(pid) -> str:
        while isinstance(pid, (list, tuple)) and len(pid) > 0:
            pid = pid[0]
        return str(pid)

    @staticmethod
    def _video_from_path_or_id(video) -> str:
        v = str(video)
        if "video_" in v and "/" not in v:
            return v
        for part in Path(v).parts:
            if part.startswith("video_"):
                return part
        return v

    # ------------------------------------------------------------------- get
    def has_track(self, set_id: str, video_id: str, pid) -> bool:
        key = (str(set_id), self._video_from_path_or_id(video_id),
               self._normalize_pid(pid))
        return key in self._store

    def get_window(
        self,
        set_id: str,
        video_id: str,
        pid,
        frame_numbers,
        fill: str = "nan",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Restituisce le feature per la finestra di frame richiesta.

        Args:
            frame_numbers: lista/array di frame number assoluti (len = T).
            fill: 'nan' o 'zero' per i frame senza feature.

        Returns:
            feat_win : [T, 768]
            mask     : [T] bool, True dove la feature e' presente
        """
        key = (str(set_id), self._video_from_path_or_id(video_id),
               self._normalize_pid(pid))
        T = len(frame_numbers)
        fill_val = np.nan if fill == "nan" else 0.0
        feat_win = np.full((T, FEAT_DIM), fill_val, dtype=np.float32)
        mask = np.zeros(T, dtype=bool)

        idx_map = self._frame_index.get(key)
        if idx_map is None:
            return feat_win, mask

        feat_all = self._store[key]
        for t, fn in enumerate(frame_numbers):
            row = idx_map.get(int(fn))
            if row is not None:
                feat_win[t] = feat_all[row]
                mask[t] = True
        return feat_win, mask

    def coverage(self, set_id, video_id, pid, frame_numbers) -> float:
        _, mask = self.get_window(set_id, video_id, pid, frame_numbers)
        return float(mask.mean()) if len(mask) else 0.0
