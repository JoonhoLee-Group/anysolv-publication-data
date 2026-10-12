#!/usr/bin/env python
"""Recompute the zwitterion numbers of the paper (Sec. III.D, Fig. 3, Table A6) from the files in this folder.

    python reproduce_zwitterions.py

Needs numpy and pandas. Reads configurations.csv, dft_labels.csv, predictions.csv, force_cosines.csv,
and geometries/. Errors are err_kcal = dE_solv(model) - dE_solv(DFT-SMD), kcal/mol (positive = the
model under-solvates); eV values are err_kcal / 23.0605.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
KCAL_PER_EV = 23.0605
KCAL_PER_EH = 627.5094740631
MOE, COMPACT, MOE_NOAUG = "model_moe", "model_compact", "model_moe_no_zwitterion"
SOLVENTS = ["n-hexane", "benzene", "toluene", "chloroform", "acetonitrile", "water"]
CUTOFF = 24.0  # Å, 4 message-passing layers x 6 Å


def show(label: str, value: float, paper: str, fmt: str) -> None:
    v = fmt.format(value)
    print(f"  {label:<60s} {v:>7s}   paper {paper:>6s}   {'ok' if v == paper else 'MISMATCH'}")


def read_xyz(path: Path):
    lines = path.read_text().splitlines()
    n = int(lines[0])
    atoms = [l.split() for l in lines[2:2 + n]]
    return lines[1], np.array([a[0] for a in atoms]), np.array([[float(v) for v in a[1:4]] for a in atoms])


def r_nc(sym, xyz) -> float:
    """Distance from the ammonium N to the carboxylate C (the C bonded to both O)."""
    (i_n,) = np.flatnonzero(sym == "N")
    o = xyz[sym == "O"]
    (i_c,) = [i for i in np.flatnonzero(sym == "C") if (np.linalg.norm(o - xyz[i], axis=1) < 1.6).sum() == 2]
    return float(np.linalg.norm(xyz[i_n] - xyz[i_c]))


cfg = pd.read_csv(HERE / "configurations.csv")
dft = pd.read_csv(HERE / "dft_labels.csv")
pred = pd.read_csv(HERE / "predictions.csv")
fcos = pd.read_csv(HERE / "force_cosines.csv")

# Consistency checks
assert len(cfg) == 76 and cfg.kind.value_counts().to_dict() == {"scan": 48, "extended": 14, "cyclic": 14}
for _, c in cfg.iterrows():
    comment, sym, xyz = read_xyz(HERE / "geometries" / c.mol / f"{c.config_id}.xyz")
    assert len(sym) == c.natoms == 3 * c.n + 7 and comment.split()[:2] == [c.config_id, f"kind={c.kind}"]
    assert abs(r_nc(sym, xyz) - c.r_NC) < 1e-3, c.config_id
assert len(list((HERE / "geometries").glob("*/*.xyz"))) == 76

bad = dft[dft.status != "ok"]
assert len(dft) == 312 and len(bad) == 6 and set(bad.config_id) == {"18-aminooctadecanoic__cyclic"}
ok = dft[dft.status == "ok"]
assert np.allclose((ok.E_smd_Eh - ok.E_gas_Eh) * KCAL_PER_EH, ok.dE_solv_kcal, atol=1e-8)
assert len(pred) == 4 * len(ok) == 1224
m = pred.merge(ok[["config_id", "solvent_key", "dE_solv_kcal"]], on=["config_id", "solvent_key"], validate="m:1")
assert len(m) == len(pred) and np.allclose(m.dE_ref_kcal, m.dE_solv_kcal, atol=1e-8)
assert np.allclose(pred.dE_pred_kcal - pred.dE_ref_kcal, pred.err_kcal, atol=1e-8)
print("checks passed: 76 geometries match configurations.csv (atoms, r(N...C)); 306 of 312 DFT-SMD labels ok "
      "(the 6 failures are the n = 17 cyclic conformer); predictions.csv errors = model - DFT-SMD\n")

# Setup
ext_r = cfg[cfg.kind == "extended"].set_index("n").r_NC
print("Sec. III.D setup")
print(f"  n = {', '.join(map(str, sorted(cfg.n.unique())))} (paper: glycine, n = 1, to n = 23)")
print(f"  extended r(N...C) {ext_r.min():.1f}-{ext_r.max():.1f} Å; first beyond {CUTOFF:.0f} Å: "
      f"n = {ext_r[ext_r > CUTOFF].index.min()} at {ext_r[ext_r > CUTOFF].min():.2f} Å")
print(f"  solvents: {', '.join(sorted(pred.solvent_key.unique(), key=SOLVENTS.index))}")


def errs(ckpt: str, kind: str, rmin: float = 0.0) -> pd.Series:
    d = pred[(pred.ckpt == ckpt) & (pred.kind == kind) & (pred.r_NC > rmin)]
    return d.err_kcal.abs()


e_ext, e_cyc, e_far = errs(MOE, "extended"), errs(MOE, "cyclic"), errs(MOE, "extended", CUTOFF)
print(f"\nSec. III.D: MoE model vs DFT-SMD (extended {len(e_ext)}, cyclic {len(e_cyc)}, "
      f"extended beyond {CUTOFF:.0f} Å {len(e_far)} rows)")
show(f"extended, MAE (eV) [{e_ext.mean() / KCAL_PER_EV:.5f}]", e_ext.mean() / KCAL_PER_EV, "0.013", "{:.3f}")
show("extended, MAE (kcal/mol)", e_ext.mean(), "0.31", "{:.2f}")
show("extended, max (eV)", e_ext.max() / KCAL_PER_EV, "0.049", "{:.3f}")
show("extended, max (kcal/mol)", e_ext.max(), "1.14", "{:.2f}")
show("cyclic, MAE (eV)", e_cyc.mean() / KCAL_PER_EV, "0.008", "{:.3f}")
show("cyclic, MAE (kcal/mol)", e_cyc.mean(), "0.18", "{:.2f}")
show("cyclic, max (eV)", e_cyc.max() / KCAL_PER_EV, "0.060", "{:.3f}")
show("cyclic, max (kcal/mol)", e_cyc.max(), "1.39", "{:.2f}")
show("extended, r(N...C) > 24 Å, MAE (eV)", e_far.mean() / KCAL_PER_EV, "0.026", "{:.3f}")
show("extended, r(N...C) > 24 Å, MAE (kcal/mol)", e_far.mean(), "0.61", "{:.2f}")
f = fcos[fcos.ckpt == MOE].set_index("solvent_key").f_cos
show(f"force cosine, r > 24 Å, lowest solvent [{f.idxmin()}, {f.min():.4f}]", f.min(), "0.999", "{:.3f}")

w = pred[(pred.ckpt == MOE_NOAUG) & (pred.kind == "extended") & (pred.solvent_key == "water")].set_index("r_NC")
r_quoted = w.index[np.isclose(w.index, 24.9, atol=0.05)][0]
print("\nSec. III.D: model without the charge-separation augmentation (model_moe_no_zwitterion), extended, water")
show(f"error at r(N...C) = {r_quoted:.1f} Å (kcal/mol)", w.err_kcal[r_quoted], "15.8", "{:.1f}")
print(f"  (not quoted: the error keeps rising to {w.err_kcal.max():.1f} kcal/mol at {w.err_kcal.idxmax():.1f} Å)")

g = ok[(ok.kind == "extended") & ok.r_NC.between(10, 30)].drop_duplicates("config_id")
print(f"\nSec. III.D: gas-phase DFT Hirshfeld charges, extended chains with 10 <= r(N...C) <= 30 Å "
      f"({len(g)} conformers, r = {g.r_NC.min():.1f}-{g.r_NC.max():.1f} Å)")
show(f"NH3 group, mean (range {g.q_NH3_gas.min():+.3f} to {g.q_NH3_gas.max():+.3f})", g.q_NH3_gas.mean(), "+0.66", "{:+.2f}")
show(f"COO group, mean (range {g.q_COO_gas.min():+.3f} to {g.q_COO_gas.max():+.3f})", g.q_COO_gas.mean(), "-0.84", "{:+.2f}")

c_ext, c_cyc = errs(COMPACT, "extended"), errs(COMPACT, "cyclic")
print("\nTable A6: zwitterions, error against DFT-SMD (kcal/mol)")
show("MoE, extended, MAE", e_ext.mean(), "0.31", "{:.2f}")
show("MoE, extended, max", e_ext.max(), "1.14", "{:.2f}")
show("MoE, cyclic, MAE", e_cyc.mean(), "0.18", "{:.2f}")
show("MoE, cyclic, max", e_cyc.max(), "1.39", "{:.2f}")
show("compact, extended, MAE", c_ext.mean(), "0.25", "{:.2f}")
show("compact, extended, max", c_ext.max(), "1.27", "{:.2f}")
show("compact, cyclic, MAE", c_cyc.mean(), "0.21", "{:.2f}")
show("compact, cyclic, max", c_cyc.max(), "1.54", "{:.2f}")

for ckpt, line in ((MOE, "solid lines and inset"), (MOE_NOAUG, "dashed lines")):
    print(f"\nFig. 3, {line} ({ckpt}): err_kcal of the extended conformers")
    tab = pred[(pred.ckpt == ckpt) & (pred.kind == "extended")].pivot(
        index=["n", "r_NC"], columns="solvent_key", values="err_kcal")[SOLVENTS]
    print(tab.round(2).to_string())
