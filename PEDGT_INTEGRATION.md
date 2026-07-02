# Integrazione PedGT — note di replica e ambiguità del paper

Questo documento descrive l'integrazione del modello **PedGT** (Riaz et al.,
*PedGT: Enhancing Pedestrian Intention Prediction using a Skeleton-based
Graph-Transformer*, IEEE IV 2025) nel repository, e traccia **tutte le scelte
fatte dove il paper è ambiguo**. Serve per la sezione "implementazione /
riproducibilità" della tesi.

L'integrazione è **completamente additiva**: la baseline GRU geometry-only
continua a funzionare identica. PedGT si attiva solo via config
(`configs/pedgt_pie.yaml`).

---

## File aggiunti / modificati

Aggiunti:
- `data/skeleton.py` — scheletro a 19 giunti (PedGNN Fig.4), adiacenza, edge_index PyG
- `data/pose_cache.py` — loader pose con **match per frame number assoluto**
- `data/pose_preproc.py` — concat bbox-center + normalizzazioni (reference-point, min-max)
- `models/pedgt.py` — modello PedGT (GCN×3 PyG + Transformer×2)
- `configs/pedgt_pie.yaml` — config con iperparametri del paper

Modificati (in modo non distruttivo):
- `data/pie_dataset.py` — propaga frame number; carica pose se `pose_dir` settato
- `train.py` — `build_model`/`collate_fn`/`run_epoch` gestiscono PedGT e keypoints
- `requirements.txt` — aggiunto `torch-geometric`

---

## Come si lancia

```bash
# baseline (invariata)
python train.py --config configs/baseline_base.yaml

# PedGT (richiede i pkl pose in data/poses/pie_hrnet_poses_setXX.pkl)
python train.py --config configs/pedgt_pie.yaml
```

---

## Il punto critico: "prendere i frame giusti"

Le pose sono indicizzate per **frame number assoluto** (`frames` nei pkl).
`pie_data` invece, per `seq_type=crossing`, taglia la track fino al
`crossing_point`, applica `[::fstride]` e può scartare frame con l'height-check:
il risultato perde il frame number esplicito.

**Soluzione**: propaghiamo il path immagine (`pie_sequences["image"]`), da cui
estraiamo `set_id`, `video_id` e il **frame number** per ogni elemento della
finestra. Il match pose↔frame avviene **per frame number**, mai per posizione
nella lista. Frame senza posa → NaN + maschera, poi forward/backward-fill.
Questo elimina il disallineamento silenzioso (che non causa crash, solo
metriche peggiori).

---

## Ambiguità del paper e scelte adottate

1. **`obs_len = 26` vs protocollo del repo (`16`).**
   PedGT usa 26 frame. Il config PedGT li imposta a 26; la baseline resta a 16.
   Non vanno mescolati: i confronti numerici con PedGT richiedono `obs_len=26`.

2. **"1-frame overlapping sliding window".**
   Ambiguo: può significare *step=1* (overlap≈totale) o *overlap=1 frame*
   (step=25). La motivazione del paper ("maintain temporal consistency")
   indica step=1. Adottato `overlap=0.96` → `step=1`. **Da verificare** se i
   conteggi di sample non tornassero.

3. **Pose estimator diverso.**
   Il paper PedGT usa AlphaPose; noi usiamo **HRNet** (`pie_hrnet_poses`).
   Verificato geometricamente che il layout è **COCO-17 standard** (ordine
   testa→caviglie, simmetria L/R, spalle a idx 5/6). Da questi 17 si derivano
   i 2 giunti aggiuntivi (Neck, CHip) richiesti dallo scheletro a 19 nodi.

4. **Matrice di adiacenza — RISOLTA (paper PedGNN/PedSynth, Fig. 4).**
   PedGT cita "the graph structure outlined in [5]". Il paper [5]
   (PedGNN/PedSynth, Riaz et al. 2024) **pubblica lo scheletro esatto** nella
   sua Fig. 4: **19 giunti** connessi come grafo non orientato. I 19 = 17
   COCO/AlphaPose + **Neck** e **CHip**, che AlphaPose non fornisce e che il
   paper deriva esplicitamente (Sec. IV-A):
       Neck = media(LShoulder, RShoulder)
       CHip = media(LHip, RHip)
   Implementato in `skeleton.py` (edge list di Fig. 4) e `pose_preproc.py`
   (`derive_19_joints`). L'input a PedGT diventa **[T, 19, 5]**.
   Nota: i parametri del modello NON cambiano (285,826) passando da 17 a 19
   nodi, perché la GCN condivide i pesi tra i nodi.

5. **Concatenazione bbox-center: prima o dopo la normalizzazione?**
   Il paper non lo specifica. Scelta adottata (in `pose_preproc.py`):
   si concatena in **pixel** → `[T,19,5]` (x,y,conf,cx,cy), poi si normalizza
   l'**intero tensore**. È l'unica scelta coerente: normalizzare le pose ma
   lasciare il centro in pixel darebbe canali a scale incompatibili.

6. **Reference-point normalization — CORRETTA dopo bug diagnosticato.**
   Confermata come default per PIE (best in Tab. II). **Attenzione**: la lettura
   ingenua (centrare OGNI frame sul proprio bbox-center e scalare per le spalle,
   applicata anche ai canali cx,cy) **azzera il canale center e DISTRUGGE la
   traiettoria temporale** del pedone. Sperimentalmente questo fa collassare il
   modello (F1=0.000, predice una sola classe) — è il bug che causava
   F1_test≈0.50 invece di 0.91.

   Implementazione corretta (`pose_preproc.reference_point`, default):
     - keypoint (canali x,y): `(coord - bbox_center_frame) / d_s` -> posa
       intra-frame scale/position-invariant;
     - center (canali cx,cy): **min-max su immagine** -> la traiettoria
       ASSOLUTA del pedone è preservata nel tempo (è il segnale chiave per il
       crossing, come dice il paper: "their relation to the pedestrian's
       overall location").
     - `conf` invariato; fallback robusto se `d_s≈0`.

   Sanity check (task proxy "movimento laterale" su set01):
     reference_point (corretta)       -> acc 0.76, F1 0.68
     reference_point_perframe (rotta) -> acc 0.64, F1 0.00

   Varianti disponibili in `normalize_pose(method=...)`:
     - `reference_point` (A, default, raccomandata)
     - `reference_point_seq` (B, riferimento = primo frame della sequenza)
     - `reference_point_perframe` (A0, la versione rotta, solo per ablation)
     - `minmax` (default storico PedGNN, best su JAAD)

7. **Gestione dei NC senza `crossing_point`.**
   Il paper dice "final frame of a trial" ma non dettaglia. Qui si eredita la
   logica TTE esistente del repo (finestra in `[T−obs−tte_max, T−obs−tte_min]`).

8. **PedSynth.**
   Escluso (l'ablation mostra che su PIE l'apporto è marginale e il best deriva
   dalla normalizzazione, non da PedSynth). I numeri JAAD del paper, che invece
   dipendono da PedSynth, non sono quindi replicabili 1:1 senza quel dataset.

---

## Iperparametri PedGT (dal paper, in `configs/pedgt_pie.yaml`)

| Voce | Valore |
|------|--------|
| obs_len | 26 |
| nodi grafo | 19 (17 COCO + Neck + CHip) |
| GCN | 5→64, ×3, ReLU+BN dopo i primi 2 |
| spatial out | 64 → proiezione a 128 |
| Transformer | 2 layer, 4 head, d_model=128 |
| dropout | 0.7 |
| optimizer | Adam |
| lr | 1e-4 (PIE) / 5e-4 (JAAD) |
| weight_decay | 5e-5 |
| epoche | 30 |
| batch | 64 |
| normalizzazione | reference-point (PIE) / min-max (JAAD) |

Parametri totali del modello: ~286k (leggero, coerente con la tesi di
efficienza del paper).
