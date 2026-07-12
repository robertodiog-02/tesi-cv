#!/usr/bin/env python3
"""
================================================================================
 SEGMENTAZIONE VIDEO CON RIMAPPATURA E DEBUG STRISCE
================================================================================
 Modifica: Gestione interruzione Ctrl+C e controllo ID per le strisce pedonali.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
import signal
from PIL import Image
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

MODEL_NAME = "facebook/mask2former-swin-large-mapillary-vistas-semantic"

# ID ufficiali Mapillary
MAPILLARY_ROAD_ID = 13
MAPILLARY_SIDEWALK_ID = 15
# Alcune versioni di Mapillary usano ID diversi per le strisce (es. 26 o 27)
MAPILLARY_CROSSWALK_IDS = [26, 27] 

CUSTOM_PALETTE = np.array([
    [128, 64, 128],   # 0: Road (Viola)
    [244, 35, 232],   # 1: Sidewalk (Rosa)
    [0, 0, 0],        # 2: Sfondo
], dtype=np.uint8)

video_writer = None
cap = None

def signal_handler(sig, frame):
    print("\n[info] Interruzione rilevata! Chiudo il video correttamente...")
    if video_writer: video_writer.release()
    if cap: cap.release()
    sys.exit(0)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="Path al file video")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    return ap.parse_args()

def pick_device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def main():
    global video_writer, cap
    args = parse_args()
    signal.signal(signal.SIGINT, signal_handler)
    
    if not os.path.isfile(args.video): sys.exit("File non trovato.")

    out = args.out or os.path.join(os.path.dirname(args.video), "seg_out_continuous")
    os.makedirs(out, exist_ok=True)
    
    device = pick_device()
    processor = Mask2FormerImageProcessor.from_pretrained(MODEL_NAME)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_NAME).to(device).eval()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(out, "video_segmentato.mp4")
    video_writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    print(f"[info] Elaborazione in corso. Debug strisce attivo.")

    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok: break

        if frame_idx % args.stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            inputs = processor(images=frame_rgb, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)
            
            pred_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[(h, w)])[0]
            raw_pred = pred_map.cpu().numpy().astype(np.uint8)

            # Logica migliorata: forziamo tutti gli ID delle strisce dentro la Road
            clean_pred = np.full_like(raw_pred, 2, dtype=np.uint8)
            
            is_road = (raw_pred == MAPILLARY_ROAD_ID)
            is_crosswalk = np.isin(raw_pred, MAPILLARY_CROSSWALK_IDS)
            is_sidewalk = (raw_pred == MAPILLARY_SIDEWALK_ID)
            
            clean_pred[is_road | is_crosswalk] = 0
            clean_pred[is_sidewalk] = 1

            color = CUSTOM_PALETTE[clean_pred]
            overlay = (0.5 * frame_rgb + 0.5 * color).astype(np.uint8)
            
            video_writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            if frame_idx % 30 == 0: print(f"Frame {frame_idx} processato.")

        frame_idx += 1

    cap.release()
    video_writer.release()
    print(f"[fatto] Video salvato in {out_path}")

if __name__ == "__main__":
    main()