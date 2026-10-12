#!/usr/bin/env python
"""Recompute the metadynamics results of the paper (Sec. III.G, Fig. 5, App. C) from the COLVAR files.

    python reproduce_alanine.py                          (needs numpy and pandas)
    python reproduce_alanine.py --fes-out DIR            (also writes the Fig. 5 surfaces, fes_<phase>.csv)
    python reproduce_alanine.py --chen ala2_dataset.npz  (adds the comparison with Chen et al.)

Reweighting: discard the first 20 % of each walker, then weight each frame by exp(rbias/kT).
FES: 5-degree bins, F = -kT ln p, zero at the minimum.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PHASES = ["gas", "chcl3", "meoh", "water"]
KT = 2.478957                      # kJ/mol, PLUMED's kT at 298.15 K
KJ = 4.184
KARPLUS = {"VuisterBax1993": (6.51, -1.76, 1.60), "HuBax1997": (7.09, -1.42, 1.55),
           "Ramachandran1971": (6.55, -1.55, 1.35)}
J_EXP = {"chcl3": 7.5, "meoh": 6.8, "water": 6.0}   # Madison and Kopple, JACS 102, 4855 (1980)


def basins(phi, psi, d):
    """Basins of App. C (degrees; d in Angstrom)."""
    ext = (psi >= 50) | (psi <= -120)
    win = (phi >= -120) & (phi < 0) & ext
    return {"C7eq": win & (d < 2.5), "PPII": win & (d >= 2.5), "aR": (phi < 0) & (psi > -120) & (psi < 50),
            "C5": (phi < -120) & ext, "phi>0": phi > 0}


def karplus(phi, p):
    c = np.cos(np.radians(phi - 60))
    return p[0] * c ** 2 + p[1] * c + p[2]


def reweight(walkers, frac=1.0):
    """Frames with t <= frac * t_max of each walker, first 20 % of those discarded; normalized weights."""
    a = np.vstack([x[(x[:, 0] <= frac * x[:, 0].max()) & (x[:, 0] >= 0.2 * frac * x[:, 0].max())] for x in walkers])
    w = np.exp((a[:, 4] - a[:, 4].max()) / KT)
    return np.degrees(a[:, 1]), np.degrees(a[:, 2]), a[:, 5], w / w.sum()


def dF(phi, psi, d, w):
    """dF(phi > 0 vs phi < 0) and dF(C7eq vs PPII + aR), kcal/mol."""
    b = basins(phi, psi, d)
    return (-KT / KJ * np.log(w[phi > 0].sum() / w[phi < 0].sum()),
            -KT / KJ * np.log(w[b["C7eq"]].sum() / (w[b["PPII"]].sum() + w[b["aR"]].sum())))


ap = argparse.ArgumentParser()
ap.add_argument("--fes-out", type=Path, help="directory for fes_<phase>.csv (phi_deg, psi_deg, F_kcal)")
ap.add_argument("--chen", type=Path, help="Chen et al.'s ala2_dataset.npz ('coords', 10^6 x 22 x 3)")
args = ap.parse_args()

W = {p: [np.loadtxt(HERE / f"metad/{p}/COLVAR.{w}", comments="#") for w in range(8)] for p in PHASES}
dt = np.diff(W["water"][0][:2, 0])[0] / 50          # COLVAR is printed every 50 steps
print("setup: " + ", ".join(f"{p} {len(W[p])} walkers x {min(x[-1, 0] for x in W[p]) / 1000:.0f} ps" for p in PHASES)
      + f"; time step {dt:g} fs; {sum(int(x[-1, 0] / dt) for x in W['water']):,} force evaluations per phase")

R = {p: reweight(W[p]) for p in PHASES}
print("\npopulations (%)")
for p in PHASES:
    phi, psi, d, w = R[p]
    pops = {k: 100 * w[m].sum() for k, m in basins(phi, psi, d).items()}
    print(f"  {p:6s} " + ", ".join(f"{k} {v:.1f}" for k, v in pops.items()) + f"; PPII+aR {pops['PPII'] + pops['aR']:.1f}")
    if args.fes_out:
        H, e, _ = np.histogram2d(phi, psi, bins=72, range=[[-180, 180], [-180, 180]], weights=w)
        with np.errstate(divide="ignore"):
            F = -KT / KJ * np.log(H)
        F -= F[np.isfinite(F)].min()
        c = (e[:-1] + e[1:]) / 2
        args.fes_out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"phi_deg": np.repeat(c, 72), "psi_deg": np.tile(c, 72),
                      "F_kcal": np.where(np.isfinite(F), F, np.nan).ravel()}).to_csv(args.fes_out / f"fes_{p}.csv", index=False)

print("\n3J(HN,HA) (Hz): Vuister-Bax, Hu-Bax, mean, Ramachandran; measured")
for p in PHASES:
    phi, psi, d, w = R[p]
    J = {k: (w * karplus(phi, v)).sum() for k, v in KARPLUS.items()}
    m = (J["VuisterBax1993"] + J["HuBax1997"]) / 2
    exp = f"; measured {J_EXP[p]}, error {m - J_EXP[p]:+.2f}" if p in J_EXP else ""
    print(f"  {p:6s} {J['VuisterBax1993']:.2f} {J['HuBax1997']:.2f} mean {m:.2f}; "
          f"Ramachandran {J['Ramachandran1971']:.2f} ({J['Ramachandran1971'] - m:+.2f}){exp}")

print("\nconvergence: spread of dF(phi>0 vs phi<0) and dF(C7eq vs PPII+aR) over the last quarter (kcal/mol)")
for p in PHASES:
    v = np.array([dF(*reweight(W[p], f)) for f in np.arange(0.75, 1.0001, 0.025)])
    print(f"  {p:6s} {np.ptp(v[:, 0]):.3f} {np.ptp(v[:, 1]):.3f}")

if args.chen:
    z = np.load(args.chen, mmap_mode="r")
    if "coords" in z:
        def dih(p0, p1, p2, p3):
            b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
            b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
            v = b0 - np.sum(b0 * b1, -1, keepdims=True) * b1
            u = b2 - np.sum(b2 * b1, -1, keepdims=True) * b1
            return np.degrees(np.arctan2(np.sum(np.cross(b1, v) * u, -1), np.sum(v * u, -1)))
        X = z["coords"]                    # atom order of ISSNet_ala2_example/simu/start_pdbs/ala2_sample_0.pdb
        cphi, cpsi, cd, cth = (np.empty(len(X)) for _ in range(4))
        for s in range(0, len(X), 100_000):
            x = np.asarray(X[s:s + 100_000])
            cphi[s:s + len(x)] = dih(x[:, 4], x[:, 6], x[:, 8], x[:, 14])
            cpsi[s:s + len(x)] = dih(x[:, 6], x[:, 8], x[:, 14], x[:, 16])
            cd[s:s + len(x)] = np.linalg.norm(x[:, 5] - x[:, 17], axis=-1)
            cth[s:s + len(x)] = dih(x[:, 7], x[:, 6], x[:, 8], x[:, 9])     # HN-N-CA-HA
    else:                                  # per-frame dihedrals already computed (phi, psi, d, theta)
        cphi, cpsi, cd, cth = (np.asarray(z[k], float) for k in ("phi", "psi", "d", "theta"))
    Jc = {k: karplus(cphi, v).mean() for k, v in KARPLUS.items()}
    mean = (Jc["VuisterBax1993"] + Jc["HuBax1997"]) / 2
    Jt = np.mean([(v[0] * np.cos(np.radians(cth)) ** 2 + v[1] * np.cos(np.radians(cth)) + v[2]).mean()
                  for k, v in KARPLUS.items() if k != "Ramachandran1971"])
    print(f"\nChen et al. ({len(cphi):,} configurations, equal weights): 3J {Jc['VuisterBax1993']:.2f} / {Jc['HuBax1997']:.2f}, "
          f"mean {mean:.2f} Hz (error {mean - J_EXP['water']:+.2f}); from HN-N-CA-HA {Jt:.2f} Hz ({Jt - mean:+.2f})")
    phi, psi, d, w = R["water"]
    cb, ob = basins(cphi, cpsi, cd), basins(phi, psi, d)
    cp = {k: m.mean() for k, m in cb.items()}
    op = {k: w[m].sum() for k, m in ob.items()}
    print("  populations (%) Chen / this work: " + ", ".join(f"{k} {100 * cp[k]:.1f}/{100 * op[k]:.1f}" for k in cp)
          + f"; C5 ratio {cp['C5'] / op['C5']:.2f}")
    print("  basin free energy -kT ln p, Chen - this work (kcal/mol): "
          + ", ".join(f"{k} {-KT / KJ * np.log(cp[k] / op[k]):+.2f}" for k in cp))
    for k in ["PPII", "aR"]:
        print(f"  mean phi in {k}: Chen {cphi[cb[k]].mean():.1f}, this work {(phi[ob[k]] * w[ob[k]]).sum() / w[ob[k]].sum():.1f} deg")
