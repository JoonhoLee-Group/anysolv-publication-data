#!/usr/bin/env python
"""Recompute the checks of the Ace-Ala-NMe simulations quoted in Sec. III.G and App. C.

    python reproduce_checks.py          (needs numpy and pandas)

Centering: x and y are the model and DFT-SMD solvation energies of one solvent minus their means over its frames;
R^2 = 1 - sum (x - y)^2 / sum y^2, and the centered error is |x - y|.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NATOMS = 22
EV_KCAL = 23.0605                  # kcal/mol per eV
YEAR_S = 365.25 * 86400
SOLVENTS = {"chcl3": "chloroform", "meoh": "methanol", "water": "water"}

# Energy conservation
print(f"NVE (water, C7eq): max |Etot - Etot(0)| and linear drift, per atom ({NATOMS} atoms)")
for m, name in [("moe", "UMA-S-1.2 + AnySolv"), ("uma", "UMA-S-1.2 alone")]:
    for dt, ref in zip(["0.5", "1"], {"moe": [0.22, 1.3], "uma": [0.21, 1.3]}[m]):
        d = pd.read_csv(HERE / f"nve/{m}_dt{dt}.csv")
        e = d.Etot_eV.to_numpy()
        dev = 1000 * np.abs(e - e[0]).max() / NATOMS
        drift = 1000 * np.polyfit(d.t_ps.to_numpy(), e, 1)[0] / NATOMS
        print(f"  {name:20s} dt {dt:3s} fs, {d.t_ps.iloc[-1]:.0f} ps: {dev:.2f} meV/atom (paper {ref}); "
              f"drift {drift:.1e} meV/atom/ps (paper < 0.002)")

# DFT-SMD check
c = pd.read_csv(HERE / "dftsmd_check/check_vs_smd.csv")
PAPER = {"chcl3": (0.9996, 0.79), "meoh": (0.9997, 0.79), "water": (0.9992, 0.85)}


def centered(g, col):
    x, y = g[col] - g[col].mean(), g.dE_dft_eV - g.dE_dft_eV.mean()
    return 1 - ((x - y) ** 2).sum() / (y ** 2).sum(), (x - y).abs().max()


print("\nDFT-SMD check (centered solvation energies; force cosine = mean over frames)")
worst = 0.0
for p, g in c.groupby("phase", sort=False):
    r2m, _ = centered(g, "dE_moe_eV")
    r2a, ea = centered(g, "dE_alpb_eV")
    worst = max(worst, ea)
    print(f"  {SOLVENTS[p]:10s} n {len(g)}: AnySolv R2 {r2m:.4f} (paper {PAPER[p][0]}), cos {g.dF_cos.mean():.5f} "
          f"(paper > 0.9998); ALPB R2 {r2a:.2f} (paper {PAPER[p][1]}), max centered error {ea:.3f} eV")
print(f"  ALPB max centered error over the three solvents: {worst:.2f} eV = {worst * EV_KCAL:.1f} kcal/mol "
      f"(paper 0.12 eV, 2.8 kcal/mol)")

# DFT-SMD solvation forces in the frame files against the rms column
rms = {}
for p in SOLVENTS:
    L = (HERE / f"dftsmd_check/frames_{p}.xyz").read_text().splitlines()
    i = 0
    while i < len(L):
        n = int(L[i])
        fid = L[i + 1].split("fid=")[1].split()[0]
        F = np.array([[float(v) for v in line.split()[4:7]] for line in L[i + 2:i + 2 + n]])
        rms[fid] = np.sqrt((F ** 2).mean())
        i += 2 + n
diff = np.abs(c.fid.map(rms) - c.dF_dft_rms_eVA).max()
print(f"  frames: {len(rms)} geometries; max |rms(dF_dft) - dF_dft_rms_eVA| {diff:.1e} eV/A")

# Cost
tm = pd.read_csv(HERE / "timing_md.csv")
ms = tm[tm.n_procs == 1].ms_per_step.mean()
s = c.orca_wall_s_smd.mean()
print(f"\ncost: one MD step, UMA-S-1.2 + AnySolv, one process on an A100 MIG 3g.20gb slice: {ms:.0f} ms (paper 84)")
print(f"  DFT-SMD energy + gradient on 16 cores, mean over {len(c)} frames: {s:.0f} s (paper 182)")
for p in SOLVENTS:
    cv = [np.loadtxt(HERE / f"metad/{p}/COLVAR.{w}", comments="#", usecols=0) for w in range(8)]
    nsteps = sum(int(round(x[-1] / (x[1] - x[0]) * 50)) for x in cv)        # COLVAR every 50 steps
    print(f"  {SOLVENTS[p]:10s} {nsteps:,} force evaluations x {s:.0f} s = {nsteps * s / YEAR_S:.1f} years "
          "(paper: two million, more than a decade)")
