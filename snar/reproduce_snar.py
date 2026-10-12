#!/usr/bin/env python
"""Recompute the S_NAr numbers of the paper (Sec. III.F, Fig. 4) from the files in this directory.

    python reproduce_snar.py            (needs numpy and pandas)

An intermediate = class stepwise or borderline (survey.csv `cls`). MoE rows only, except the check.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PHASES = ["gas", "thf", "acetonitrile", "dmso", "methanol", "water"]
INT = ["stepwise", "borderline"]

rxn = pd.read_csv(HERE / "reactions.csv").set_index("reaction")
survey = pd.read_csv(HERE / "survey.csv")
legs = pd.read_csv(HERE / "legs.csv", low_memory=False)
s = survey[survey.model == "model_moe"]
assert len(rxn) == 120 and len(s) == 120 * len(PHASES)
cls = s.pivot(index="reaction", columns="phase", values="cls")[PHASES]
D = s.pivot(index="reaction", columns="phase", values="D_elec_kcal")[PHASES]
has = cls.isin(INT)
kwan = rxn.kwan_mechanism.reindex(cls.index)

# Cost: every attempt of the MoE reaction units, including the cross-phase follow-up
L = legs[legs.model == "model_moe"]
n_min = L.leg.isin(["int", "rc", "pc"]).sum()
n_ts = L.leg.isin(["ts_add", "ts_elim", "ts_conc"]).sum()
n_irc = L.irc_method.notna().sum()
print(f"cost: {len(s)} reaction-phase pairs; {n_min} minimizations, {n_ts} TS searches, {n_irc} IRCs; "
      f"{L.wall_s.sum() / 3600:.1f} GPU-hours (A100 MIG 3g.20gb slices)")

# Fig. 4
shown = has.any(axis=1)
print(f"\nFig. 4: {shown.sum()} reactions with an intermediate in at least one phase; "
      f"{(~shown).sum()} without, of which {(kwan[~shown] == 'concerted').sum()} concerted in Kwan et al.")

# DMSO vs Kwan et al.
dm = has["dmso"]
print("\nDMSO, intermediate (this work) vs Kwan et al.'s class")
print(pd.crosstab(kwan, dm.rename("intermediate")).to_string())
agree = ((kwan == "stepwise") == dm).sum()
disp = dm & (kwan == "concerted")
print(f"agree on whether an intermediate exists: {agree}/120; Kwan-stepwise with an intermediate: "
      f"{(dm & (kwan == 'stepwise')).sum()}/{(kwan == 'stepwise').sum()}")
print(f"disputed: {disp.sum()}; D < 1 kcal/mol in {(D.dmso[disp] < 1).sum()}; "
      f"median D {D.dmso[disp].median():.2f}, max {D.dmso[disp].max():.2f} kcal/mol")
print(f"gas phase, the disputed intermediates: {cls.gas[disp].value_counts().to_dict()}")
for lab, sel in [("disputed", disp), ("Kwan-stepwise", dm & (kwan == "stepwise"))]:
    dd = (D.gas - D.dmso)[sel].dropna()
    print(f"  {lab}: D(gas) > D(DMSO) for {(dd > 0).sum()}/{len(dd)} with D in both phases, "
          f"median {dd.median():+.1f}, min {dd.min():+.1f} kcal/mol")

# Trends in DMSO
d = s[s.phase == "dmso"].set_index("reaction")
print("\nDMSO intermediates by leaving group (this work / Kwan et al.):",
      {x: f"{(dm & (d.X == x)).sum()}/{((kwan == 'stepwise') & (d.X == x)).sum()}" for x in ["F", "Cl", "Br"]})
print("by ring (this work / Kwan et al.):",
      {r: f"{(dm & (d.ring == r)).sum()}/{((kwan == 'stepwise') & (d.ring == r)).sum()}" for r in d.ring.unique()})
deep = d.loc[D.dmso[D.dmso > 4].index]
print(f"wells deeper than 4 kcal/mol: {len(deep)}, leaving groups {sorted(deep.X.unique())}, "
      f"rings {sorted(deep.ring.unique())}")

# Phases
print("\nper phase: intermediates / unresolved / median D of the intermediates (kcal/mol)")
for p in PHASES:
    print(f"  {p:13s} {has[p].sum():3d} {(cls[p] == 'unresolved').sum():3d} {D[p][has[p]].median():6.2f}")
# Mechanism switches between phases (unresolved cells ignored); 'aprotic' = among THF, MeCN, DMSO
res = cls != "unresolved"
switch = {}
for r in cls.index:
    apro = {has.loc[r, p] for p in ["thf", "acetonitrile", "dmso"] if res.loc[r, p]}
    if len(apro) > 1:
        switch[r] = "aprotic"
        continue
    groups = [g for g, ph in [("protic", ["methanol", "water"]), ("gas", ["gas"])]
              if any(res.loc[r, p] and has.loc[r, p] not in apro for p in ph)]
    if groups:
        switch[r] = "+".join(groups)
print(f"reactions whose mechanism changes with the phase: {len(switch)}, {pd.Series(switch).value_counts().to_dict()}")
print(f"  aprotic: {[r for r, g in switch.items() if 'aprotic' in g]}, "
      f"D = {D.loc[[r for r, g in switch.items() if 'aprotic' in g], ['thf', 'acetonitrile', 'dmso']].round(2).values.tolist()}")
for p in ["thf", "acetonitrile", "methanol", "water", "gas"]:
    both = has[p] & has["dmso"]
    dd = (D[p] - D.dmso)[both].dropna()
    print(f"  D({p}) - D(DMSO), reactions with an intermediate in both: n = {len(dd)}, median {dd.median():+.2f} kcal/mol")

# Geometries vs survey.csv
bad, n = [], 0
for _, row in survey.iterrows():
    for leg in row.legs.split("+"):
        f = HERE / "geoms" / row.model / f"{row.reaction}_{row.phase}_{leg}.xyz"
        lines = f.read_text().splitlines()
        meta = dict(t.split("=", 1) for t in lines[1].split() if "=" in t)
        xyz = np.array([[float(v) for v in ln.split()[1:4]] for ln in lines[2:2 + int(lines[0])]])
        i, x, y = (int(meta[k]) for k in ("ipso_index", "X_index", "Y_index"))
        rcx, rcy = np.linalg.norm(xyz[i] - xyz[x]), np.linalg.norm(xyz[i] - xyz[y])
        ok = (abs(rcx - row[f"{leg}_rCX"]) < 2e-3 and abs(rcy - row[f"{leg}_rCY"]) < 2e-3
              and abs(float(meta["E_elec_eV"]) - row[f"{leg}_E_eV"]) < 1e-5)
        n += 1
        if not ok:
            bad.append(f.name)
print(f"\ngeometries: {n} files, {len(bad)} disagree with survey.csv {bad[:5]}")
