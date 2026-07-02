"""
PROVA DIAGNOSTICA — Test set usato come Validation set
======================================================
ATTENZIONE: questo NON è un esperimento metodologicamente valido.
Serve SOLO a osservare, epoca per epoca, come si comportano le metriche
sul TEST set durante il training (curve di apprendimento sul test).

In un setup corretto il test set si guarda UNA volta sola, alla fine.
Qui invece lo usiamo come validation per:
  - plottare le curve (loss/f1/acc/auc) sul test ad ogni epoca;
  - vedere quando il modello raggiunge il picco sul test e quanto poi degrada;
  - capire se il gap train/val osservato è specifico del val set o generale.

NON usare i numeri prodotti qui per la tesi come risultati ufficiali:
la selezione del "best model" avverrebbe sul test (data leakage).

Uso:
    python train_test_as_val.py --config configs/pedgt_pie.yaml

Equivale a:  python train.py --config <config> --test_as_val
ma è un file separato per chiarezza, come richiesto.
I risultati finiscono in checkpoints/<nome>_TESTASVAL/ così non
sovrascrivono il run normale, e le curve sono in quella cartella.
"""

import sys
import runpy


def main():
    # inietta il flag --test_as_val e delega a train.main()
    if "--test_as_val" not in sys.argv:
        sys.argv.append("--test_as_val")
    # esegue train.py come __main__ con gli stessi argomenti
    sys.argv[0] = "train.py"
    runpy.run_module("train", run_name="__main__")


if __name__ == "__main__":
    main()
