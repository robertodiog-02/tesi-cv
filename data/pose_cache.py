"""
Pose cache loader — match per FRAME NUMBER ASSOLUTO
====================================================
Carica i pkl `pie_hrnet_poses_setXX.pkl` e fornisce, per ogni
(set, video, pid), una lookup frame_number -> (keypoints[17,3], bbox[4]).

PERCHE' IL MATCH PER FRAME NUMBER E' OBBLIGATORIO
-------------------------------------------------
`pie_data.generate_data_trajectory_sequence` (seq_type=crossing):
  - taglia la track fino al `crossing_point`;
  - applica `[::seq_stride]` (fstride);
  - puo' scartare frame con l'height-check.
Il risultato e' una lista di bbox SENZA piu' il frame number esplicito.
Le pose invece sono indicizzate per frame number assoluto.
Agganciarle per posizione-in-lista produce disallineamenti silenziosi
(nessun crash, solo metriche peggiori). Quindi propaghiamo il frame number
dentro ogni sample (vedi pie_dataset.py) e qui facciamo il match per frame id.

Struttura pkl attesa:
    { 'setXX': { 'video_YYYY': { '<pid>': {
          'frames'   : np.ndarray[T_abs]      (frame number assoluti, contigui)
          'keypoints': np.ndarray[T_abs,17,3] (x, y, conf)  -- layout COCO-17
          'bbox'     : np.ndarray[T_abs,4]
    }}}}

Il `pid` nel pkl e' nel formato PIE '<set>_<video>_<obj>' (es. '1_1_1'),
identico a quello restituito da pie_data (pid_annots key).
"""

import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

NUM_JOINTS = 17


class PoseCache:
    """
    Lookup pose per (set, video, pid, frame_number).

    Uso tipico:
        cache = PoseCache("data/poses")          # cartella con i pkl
        kp = cache.get_window(set_id, video_id, pid, frame_numbers)
        # kp -> [len(frame_numbers), 17, 3], con NaN dove la posa manca
    """

    def __init__(self, pose_dir: str, set_ids=None, verbose: bool = True):
        self.pose_dir = Path(pose_dir)
        self.verbose = verbose
        # index[(set, video, pid)] = {frame_number: row_idx}
        self._frame_index: Dict[Tuple[str, str, str], Dict[int, int]] = {}
        # store[(set, video, pid)] = (keypoints[T,17,3], bbox[T,4])
        self._store: Dict[Tuple[str, str, str], Tuple[np.ndarray, np.ndarray]] = {}
        self._n_tracks = 0
        self._load(set_ids)

    # ------------------------------------------------------------------ load
    def _candidate_files(self, set_ids):
        if set_ids is None:
            # tutti i pkl che matchano il pattern
            files = sorted(self.pose_dir.glob("pie_hrnet_poses_set*.pkl"))
        else:
            files = []
            for sid in set_ids:
                # accetta sia 'set01' che '1'
                num = sid.replace("set", "").zfill(2)
                f = self.pose_dir / f"pie_hrnet_poses_set{num}.pkl"
                if f.exists():
                    files.append(f)
        return files

    def _load(self, set_ids):
        files = self._candidate_files(set_ids)
        if not files:
            raise FileNotFoundError(
                f"Nessun pkl pose trovato in {self.pose_dir} "
                f"(pattern: pie_hrnet_poses_setXX.pkl)"
            )
        for f in files:
            with open(f, "rb") as fh:
                data = pickle.load(fh)
            for set_id, videos in data.items():
                for video_id, peds in videos.items():
                    for pid, payload in peds.items():
                        frames = np.asarray(payload["frames"]).astype(np.int64)
                        kp = np.asarray(payload["keypoints"], dtype=np.float32)
                        bb = np.asarray(payload["bbox"], dtype=np.float32)
                        assert kp.shape[1:] == (NUM_JOINTS, 3), \
                            f"keypoints shape inattesa {kp.shape} per {pid}"
                        key = (set_id, video_id, str(pid))
                        self._store[key] = (kp, bb)
                        self._frame_index[key] = {
                            int(fn): i for i, fn in enumerate(frames)
                        }
                        self._n_tracks += 1
            if self.verbose:
                print(f"[PoseCache] caricato {f.name}")
        if self.verbose:
            print(f"[PoseCache] tracks totali: {self._n_tracks}")

    # ------------------------------------------------------------- normalize
    @staticmethod
    def _normalize_pid(pid) -> str:
        """pie_data puo' restituire il pid annidato ([[pid]]). Srotola fino
        alla stringa '<set>_<video>_<obj>'."""
        while isinstance(pid, (list, tuple)) and len(pid) > 0:
            pid = pid[0]
        return str(pid)

    @staticmethod
    def _video_from_path_or_id(video) -> str:
        """Accetta 'video_0001' oppure un frame-path; estrae 'video_XXXX'."""
        v = str(video)
        if "video_" in v and "/" not in v:
            return v
        # estrai da un path tipo .../set01/video_0001/00123.png
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
        Restituisce le pose per la finestra di frame richiesta.

        Args:
            frame_numbers: lista/array di frame number assoluti (len = T).
            fill: 'nan' o 'zero' per i frame senza posa.

        Returns:
            kp_win   : [T, 17, 3]
            mask     : [T] bool, True dove la posa e' presente
        """
        key = (str(set_id), self._video_from_path_or_id(video_id),
               self._normalize_pid(pid))
        T = len(frame_numbers)
        fill_val = np.nan if fill == "nan" else 0.0
        kp_win = np.full((T, NUM_JOINTS, 3), fill_val, dtype=np.float32)
        mask = np.zeros(T, dtype=bool)

        idx_map = self._frame_index.get(key)
        if idx_map is None:
            return kp_win, mask  # track non in cache -> tutto fill

        kp_all, _ = self._store[key]
        for t, fn in enumerate(frame_numbers):
            row = idx_map.get(int(fn))
            if row is not None:
                kp_win[t] = kp_all[row]
                mask[t] = True
        return kp_win, mask

    def coverage(self, set_id, video_id, pid, frame_numbers) -> float:
        _, mask = self.get_window(set_id, video_id, pid, frame_numbers)
        return float(mask.mean()) if len(mask) else 0.0
