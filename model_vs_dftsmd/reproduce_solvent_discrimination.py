#!/usr/bin/env python
"""Recompute the solvent-discrimination statistics of the paper (Sec. III.A) from the files in this directory.

    python reproduce_solvent_discrimination.py            (needs numpy and pandas)

Data: single-point solvation energies dE (kcal/mol) of the 4,511 Solv@TUM pairs at the database geometry,
AnySolv (`de_mod_kcal`) against our DFT-SMD reference (`de_dft_kcal`), in per_pair/forces_solvatum{,_topup}_<model>.csv.
The pairs are split into the 36 training solvents and all other ("unseen") solvents; within each slice:

  eligible solute  a solute with at least 4 pairs in the slice
  centering        y = dE - (mean dE over the solute's pairs), applied to both the reference and the model
  pooled R2        1 - sum (yhat - y)^2 / sum y^2 over all eligible pairs: the coefficient of determination
                   about y = x, with no fitted parameters
  per-solute R2    the same coefficient of determination over the pairs of one solute; median over solutes
  null model       every pair predicted by its solute's mean reference dE; R2 against the uncentered reference
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MIN_N = 4
MODELS = {"model_moe": "MoE", "model_compact": "compact"}
# the 36 DFT-SMD training solvents, as `solvent_key`
TRAINED = {
    "1,2-dichloroethane", "1,4-dioxane", "1-octanol", "1-pentanol", "1-propanol", "2,2,2-trifluoroethanol",
    "N,N-dimethylformamide", "N-methylformamide (E/Z mixture)", "acetone", "acetonitrile", "aniline", "benzaldehyde",
    "benzene", "bromobenzene", "carbon disulfide", "carbon tetrachloride", "chlorobenzene", "chloroform",
    "cyclohexanone", "dichloromethane", "diethyl ether", "diiodomethane", "dimethyl sulfoxide (DMSO)", "ethanol",
    "ethyl ethanoate", "iodobenzene", "methanol", "n-hexadecane", "n-hexane", "n-pentane", "nitromethane",
    "o-dichlorobenzene", "tetrahydrofuran", "toluene", "tributylphosphate", "water",
}
assert len(TRAINED) == 36


def load(model):
    df = pd.concat([pd.read_csv(HERE / f"per_pair/forces_solvatum_{model}.csv"),
                    pd.read_csv(HERE / f"per_pair/forces_solvatum_topup_{model}.csv")])
    df = df.drop_duplicates(["name", "solvent_key"])
    assert len(df) == 4511
    df["solute"] = df.name.str.split("__").str[0]
    df["trained"] = df.solvent_key.isin(TRAINED)
    return df


def slice_stats(df):
    n = df.groupby("solute").size()
    d = df[df.solute.isin(n[n >= MIN_N].index)].copy()
    d["y"] = d.de_dft_kcal - d.groupby("solute").de_dft_kcal.transform("mean")
    d["yhat"] = d.de_mod_kcal - d.groupby("solute").de_mod_kcal.transform("mean")
    d["res2"] = (d.yhat - d.y) ** 2
    d["y2"] = d.y ** 2
    g = d.groupby("solute")[["res2", "y2"]].sum()
    per_solute = 1 - g.res2 / g.y2
    ybar = d.groupby("solute").de_dft_kcal.transform("mean")
    null = 1 - ((ybar - d.de_dft_kcal) ** 2).sum() / ((d.de_dft_kcal - d.de_dft_kcal.mean()) ** 2).sum()
    return {"solvents": df.solvent_key.nunique(), "solutes": len(g), "pairs": len(d),
            "pooled": 1 - g.res2.sum() / g.y2.sum(), "median": float(np.median(per_solute)), "null": null}


print(f"{'model':8s} {'slice':8s} {'solvents':>8s} {'solutes':>8s} {'pairs':>6s} {'pooled R2':>10s} "
      f"{'median R2':>10s} {'null R2':>8s}")
for model, label in MODELS.items():
    df = load(model)
    for name, mask in [("trained", df.trained), ("unseen", ~df.trained)]:
        s = slice_stats(df[mask])
        print(f"{label:8s} {name:8s} {s['solvents']:8d} {s['solutes']:8d} {s['pairs']:6d} {s['pooled']:10.4f} "
              f"{s['median']:10.4f} {s['null']:8.3f}")
