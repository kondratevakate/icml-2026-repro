"""
Reproduction (toy, real data) of Claim 1 of:
  "PyHealth 2.0: A Comprehensive Open-Source Toolkit for Accessible and Reproducible ..."
  ICML 2026 (orid gMLVFN9hl8).

Claim 1: PyHealth 2.0 unifies 15+ datasets, 20+ clinical tasks, 25+ models, 5+ interpretability
         methods with support for diverse clinical data modalities, enabling predictive modeling
         "in as few as 7 lines of code".

We do not attempt to verify the exact counts (15/20/25/5) -- that is a catalog claim, checkable by
reading their source tree, not by running code. What we DO verify empirically is the *capability*
claim: that the toolkit's stated end-to-end path (dataset -> task -> model -> interpretability)
actually works, using the REAL PyHealth 2.0 code (installed editable from GitHub master, since the
package requires Python >=3.12 but this environment has one syntax-compatible relaxation to 3.11 --
documented) against the PUBLIC, no-auth MIMIC-III Clinical Database Demo (100 patients,
physionet.org/content/mimiciii-demo/1.4/).

TOY: 100 demo patients is far below the scale of a real MIMIC-III (40k+ patients) study; this
checks that the pipeline RUNS and produces sane outputs, not the paper's benchmark numbers.
"""
import sys
import time
import types
import numpy as np
import torch

DATA_ROOT = r"C:/Projects/02_academia/icml-repro/data/mimic3demo"

# PyHealth's DaskCluster picks processes=not in_notebook(), and litdata's in_notebook() is just
# `"ipykernel" in sys.modules`. Process-based dask on Windows repeatedly deadlocked while caching
# the (tiny, 100-patient) demo event dataframe. Faking the ipykernel check makes dask use a
# threaded local cluster instead -- same computation, just a different (and on Windows, far more
# reliable) execution backend for this small dataset. Documented, not a silent hack.
sys.modules.setdefault("ipykernel", types.ModuleType("ipykernel"))


def main():
    t0 = time.time()
    from pyhealth.datasets import MIMIC3Dataset
    from pyhealth.tasks import MortalityPredictionMIMIC3

    print("=== Step 1: load MIMIC-III demo via PyHealth's own MIMIC3Dataset ===")
    dataset = MIMIC3Dataset(
        root=DATA_ROOT,
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
    )
    dataset.stats()
    print(f"  loaded in {time.time()-t0:.1f}s")

    print("\n=== Step 2: apply a built-in clinical task (MortalityPredictionMIMIC3) ===")
    task = MortalityPredictionMIMIC3()
    samples = dataset.set_task(task)
    n = len(samples)
    print(f"  produced {n} task samples from the demo cohort (100 patients)")
    if n == 0:
        print("  [!] 0 samples: the demo cohort likely lacks patients with >1 admission "
              "(the task needs a current + a NEXT visit to define the mortality label). "
              "This is an honest capability finding, not a bug we should paper over.")
        return {"n_samples": 0}

    print("\n=== Step 3: train a built-in model on the samples ===")
    from pyhealth.datasets import get_dataloader, split_by_patient
    from pyhealth.models import Transformer
    from pyhealth.trainer import Trainer

    train_ds, val_ds, test_ds = split_by_patient(samples, [0.7, 0.15, 0.15])
    train_loader = get_dataloader(train_ds, batch_size=8, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=8, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=8, shuffle=False)
    print(f"  split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    model = Transformer(dataset=samples, feature_keys=["conditions", "procedures", "drugs"],
                        label_key="mortality", mode="binary")
    trainer = Trainer(model=model, metrics=["accuracy", "roc_auc"])
    trainer.train(train_dataloader=train_loader, val_dataloader=val_loader, epochs=3)
    result = trainer.evaluate(test_loader)
    print(f"  test metrics (toy scale, 100 patients): {result}")

    print("\n=== Step 4: apply an interpretability method ===")
    try:
        from pyhealth.interpret.methods.chefer import CheferRelevance
        interp = CheferRelevance(model)
        batch = next(iter(test_loader))
        rel = interp.get_relevance_matrix(**batch)
        print(f"  interpretability method ran; relevance shape info: "
              f"{type(rel)} (keys: {list(rel.keys()) if isinstance(rel, dict) else 'n/a'})")
    except Exception as e:
        print(f"  [!] interpretability step failed: {type(e).__name__}: {e}")

    print(f"\ntotal wall time: {time.time()-t0:.1f}s")
    return {"n_samples": n, "result": result}


if __name__ == "__main__":
    main()
