#!/usr/bin/env python
"""Recompute the SAMPL6 log P numbers of the paper (Sec. III.C, Table A6) from the files in this folder.

    python reproduce_sampl6.py

Needs numpy and pandas. Reads conformers.csv (per-conformer free energies), logP.csv (per-compound
predictions as written by the driver), logP_experimental_values.csv (SAMPL6 reference data), and
geometries/. log P = (G_water - G_octanol) / (kT ln 10) at 298.15 K, with G per phase either the
lowest conformer free energy or the Boltzmann ensemble free energy -kT ln sum_i exp(-G_i/kT).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
KT = 8.617333262e-5 * 298.15  # eV
KT_LN10 = KT * np.log(10.0)  # eV per log10 unit
MODELS = {"model_moe": "MoE", "model_compact": "compact"}
MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Cl": 35.45}


def ensemble(g: np.ndarray) -> float:
    g0 = g.min()
    return g0 - KT * np.log(np.exp(-(g - g0) / KT).sum())


def log_p(conf: pd.DataFrame, col: str = "G_eV") -> pd.DataFrame:
    rows = []
    for (model, mol), d in conf.groupby(["model", "mol"]):
        w = d.loc[d.phase == "water", col].to_numpy()
        o = d.loc[d.phase == "1-octanol", col].to_numpy()
        rows.append({"model": model, "mol": mol, "n_conf": len(w),
                     "logP_lowest": (w.min() - o.min()) / KT_LN10,
                     "logP_ensemble": (ensemble(w) - ensemble(o)) / KT_LN10})
    return pd.DataFrame(rows)


def read_xyz(path: Path):
    lines = path.read_text().splitlines()
    n = int(lines[0])
    atoms = [l.split() for l in lines[2:2 + n]]
    return lines[1], [a[0] for a in atoms], np.array([[float(v) for v in a[1:4]] for a in atoms])


def half_ln_inertia(sym, xyz) -> float:
    m = np.array([MASS[s] for s in sym])
    x = xyz - (m[:, None] * xyz).sum(0) / m.sum()
    inertia = sum(mi * (xi @ xi * np.eye(3) - np.outer(xi, xi)) for mi, xi in zip(m, x))
    return 0.5 * np.log(np.linalg.eigvalsh(inertia)).sum()


def show(label: str, value: float, paper: str, fmt: str = "{:.2f}") -> None:
    v = fmt.format(value)
    print(f"  {label:<46s} {v:>8s}   paper {paper:>6s}   {'ok' if v == paper else 'MISMATCH'}")


conf = pd.read_csv(HERE / "conformers.csv")
pred = pd.read_csv(HERE / "logP.csv")
exp = pd.read_csv(HERE / "logP_experimental_values.csv").rename(
    columns={"Molecule ID": "mol", "logP mean": "logP_exp", "logP SEM": "logP_sem"})

# Consistency checks
n_conf = conf.drop_duplicates(["mol", "conf"])
assert len(conf) == 47 * 2 * 2 and len(n_conf) == 47, "expected 47 conformers x 2 phases x 2 models"
assert (conf.status == "ok").all() and conf.converged.all() and (conf.n_imag_big == 0).all()
for _, r in n_conf.iterrows():
    f = HERE / "geometries" / r.mol / f"conf{r.conf:02d}.xyz"
    comment, sym, _ = read_xyz(f)
    e_rel = float(re.search(r"E_gas_rel = ([-\d.]+)", comment).group(1))
    assert abs(e_rel - r.E_xtb_gas_rel_kcal) < 1e-6, f
    assert 3 * len(sym) - 6 == r.n_modes, f
assert len(list((HERE / "geometries").glob("SM*/conf*.xyz"))) == 47
chk = pred.merge(exp, on="mol", suffixes=("", "_ref"))
assert np.allclose(chk.logP_exp, chk.logP_exp_ref) and np.allclose(chk["sem"], chk.logP_sem)
mine = log_p(conf)
cmp = mine.merge(pred, on=["model", "mol"], suffixes=("", "_file"))
assert len(cmp) == 22 and (cmp.n_conf == cmp.n_conf_pairs).all()
dev = max((cmp.logP_lowest - cmp.logP_lowest_file).abs().max(),
          (cmp.logP_ensemble - cmp.logP_ensemble_file).abs().max())
assert dev < 1e-3, dev
print(f"checks passed: 47 conformer files match conformers.csv; log P recomputed from G_eV agrees with "
      f"logP.csv to {dev:.4f}; experimental values in logP.csv match logP_experimental_values.csv\n")

# Sec. III.C
per_mol = n_conf.groupby("mol").size()
print("Sec. III.C setup")
print(f"  compounds {per_mol.size} (paper 11), conformers {per_mol.sum()} (paper 47), "
      f"per compound {per_mol.min()}-{per_mol.max()} (paper: up to 8)")

res = mine.merge(exp[["mol", "logP_exp"]], on="mol")
rmse = {}
for ckpt, name in MODELS.items():
    d = res[res.model == ckpt]
    for est in ("ensemble", "lowest"):
        rmse[name, est] = np.sqrt(((d[f"logP_{est}"] - d.logP_exp) ** 2).mean())

print("\nSec. III.C / Table A6: RMSE of log10 P against experiment (11 compounds)")
show("MoE, Boltzmann ensemble", rmse["MoE", "ensemble"], "0.51")
show("MoE, lowest-free-energy conformer", rmse["MoE", "lowest"], "0.53")
show("compact, Boltzmann ensemble", rmse["compact", "ensemble"], "0.51")
show("compact, lowest-free-energy conformer", rmse["compact", "lowest"], "0.54")

# Translational / rotational terms (approximate check)
# Rotational term only (sigma = 1); moments of inertia from the starting conformers in geometries/.
half_ln_i = {(p.parent.name, int(p.stem[4:])): half_ln_inertia(*read_xyz(p)[1:])
             for p in (HERE / "geometries").glob("SM*/conf*.xyz")}
conf["G_rot_eV"] = conf.G_eV - KT * np.array([half_ln_i[m, c] for m, c in zip(conf.mol, conf.conf)])
shift = log_p(conf, "G_rot_eV").merge(mine, on=["model", "mol"], suffixes=("_rot", ""))
d_max = max((shift.logP_lowest_rot - shift.logP_lowest).abs().max(),
            (shift.logP_ensemble_rot - shift.logP_ensemble).abs().max())
print("\nSec. III.C: change in log10 P from adding rotational terms (approximate: moments of inertia")
print(f"  of the starting conformers): max |dlogP| = {d_max:.3f} over 11 compounds, both models and "
      f"estimators (paper: < 0.01)")

print("\nPer compound (log10 P):")
tab = res.pivot(index="mol", columns="model", values=["logP_ensemble", "logP_lowest"])
tab.columns = [f"{MODELS[m]}_{c.replace('logP_', '')}" for c, m in tab.columns]
tab.insert(0, "exp", exp.set_index("mol").logP_exp)
print(tab.round(2).to_string())
