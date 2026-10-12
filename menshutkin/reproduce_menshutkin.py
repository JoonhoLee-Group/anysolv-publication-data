#!/usr/bin/env python
"""Recompute the Menshutkin numbers of the paper (Sec. III.E, Tables IV and V) from the files in this directory.

    python reproduce_menshutkin.py            (needs numpy and pandas)

Prints each number next to the value quoted in the paper, [in brackets]; '*' marks a value that does not round
to the quoted one (or falls outside a quoted bound or range) and is listed again at the end.

G = G_eV of the last accepted row per (model, phase, leg) of a legs.csv; dG_barrier = G(ts) - G(sm),
dG_rxn = G(prod) - G(sm). The gas phase is the gas-phase MLIP alone. DFT-SMD: dft/dft_free_energies.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EV = 23.0605478                          # kcal/mol per eV
SOLV = ["hexane", "chcl3", "methanol", "acetonitrile", "dmso", "water"]
PHASES = ["gas"] + SOLV
ATOMS = ["C", "Cl", "H", "H", "H", "N", "H", "H", "H"]   # r(C-N) = atoms 0-5, r(C-Cl) = atoms 0-1, N-H = 6, 7, 8
BASES = [("UMA-S-1.2", "model"), ("MACE-POLAR-1-M", "macepolar"), ("OrbMol-v2", "orbmolv2"),
         ("AIMNet2", "aimnet2"), ("AIMNet2-NSE", "aimnet2nse")]
OMOL25 = ["UMA-S-1.2", "MACE-POLAR-1-M", "OrbMol-v2"]
PREFIX = {"macepolar": "polar1m", "orbmolv2": "orbmolv2", "aimnet2": "aimnet2", "aimnet2nse": "aimnet2nse"}

# values quoted in the paper; Table IV columns DFT-SMD, MoE, compact
P4 = {"bar": {"gas": (43.1, 43.0, 43.0), "hexane": (32.8, 32.7, 33.5), "chcl3": (27.0, 26.8, 28.2),
              "methanol": (23.8, 23.5, 25.0), "acetonitrile": (23.4, 23.7, 25.1), "dmso": (23.3, 23.4, 24.8),
              "water": (23.1, 24.0, 25.2)},
      "rxn": {"gas": (41.2, 41.4, 41.4), "hexane": (18.9, 18.8, 18.9), "chcl3": (1.1, -0.3, -5.5),
              "methanol": (-10.9, -11.4, -19.6), "acetonitrile": (-11.3, -12.0, -21.1),
              "dmso": (-11.9, -12.4, -20.7), "water": (-11.8, -12.7, -21.2)},
      "rCCl_ts": {"gas": (2.45, 2.46, 2.46), "hexane": (2.33, 2.32, 2.32), "chcl3": (2.27, 2.26, 2.24),
                  "methanol": (2.23, 2.22, 2.22), "acetonitrile": (2.23, 2.23, 2.23), "dmso": (2.23, 2.23, 2.22),
                  "water": (2.23, 2.22, 2.22)},
      "rCCl_prod": {"gas": (2.77, 2.77, 2.77), "hexane": (3.00, 3.06, 3.04), "chcl3": (3.25, 3.43, 4.60),
                    "methanol": (3.62, 3.51, 4.49), "acetonitrile": (3.61, 3.49, 4.50), "dmso": (3.60, 3.52, 4.50),
                    "water": (3.54, 3.59, 4.57)}}
# Table V: gas dG_barrier, gas dG_rxn, solvated barrier MSE, MAE, dG_rxn MSE, MAE
P5 = {"UMA-S-1.2": (0.0, 0.2, 0.1, 0.3, -0.7, 0.7), "MACE-POLAR-1-M": (0.4, 0.2, 0.5, 0.5, -0.4, 0.4),
      "OrbMol-v2": (0.2, 0.1, 0.2, 0.4, -0.8, 0.8), "AIMNet2": (4.7, 5.0, 6.5, 6.5, 3.8, 3.8),
      "AIMNet2-NSE": (9.4, 1.5, 10.1, 10.1, 1.3, 1.3)}
NH_CL = {"AIMNet2": ["methanol", "acetonitrile", "dmso"], "AIMNet2-NSE": ["hexane"]}   # Table V footnote a
MISS = []


def q(x, paper, nd=1, tag=""):
    """'x [paper]' with x at nd + 2 decimals (the paper quotes nd); '*' if x does not round to paper."""
    ok = f"{x:.{nd}f}".replace("-0.0", "0.0") == f"{paper:.{nd}f}".replace("-0.0", "0.0")
    if not ok:
        MISS.append(f"{tag} {x:.{nd + 2}f} [{paper}]")
    return f"{x:{nd + 7}.{nd + 2}f} [{paper:{nd + 4}.{nd}f}]{' ' if ok else '*'}"


def within(x, bound, tag):
    """x against a quoted upper bound ('within 0.9'): '*' if x rounds above it."""
    ok = round(x, 1) <= bound
    if not ok:
        MISS.append(f"{tag} {x:.3f} [within {bound}]")
    return f"{x:.2f} [within {bound}]{'' if ok else '*'}"


def span(lo, hi, paper, tag):
    """a recomputed range against a quoted one, e.g. paper = '0.7-2.0'."""
    ok = f"{lo:.1f}-{hi:.1f}" == paper
    if not ok:
        MISS.append(f"{tag} {lo:.3f}-{hi:.3f} [{paper}]")
    return f"{lo:.2f} to {hi:.2f} [{paper}]{'' if ok else '*'}"


def read_xyz(f):
    lines = f.read_text().splitlines()
    assert int(lines[0]) == 9 and [ln.split()[0] for ln in lines[2:11]] == ATOMS, f
    meta = dict(t.split("=", 1) for t in lines[1].split() if "=" in t)
    return lines[1].split(), meta, np.array([[float(v) for v in ln.split()[1:4]] for ln in lines[2:11]])


def geom(folder, model, phase, leg):
    """xyz file of a stationary point: dft/dft_<phase>_<leg>.xyz, model/<model>_<phase>_<leg>.xyz (UMA-S-1.2),
    <folder>/<base>_<model>_<phase>_<leg>.xyz (other gas-phase MLIPs; model 'none' = the MLIP alone)."""
    name = f"{model}_{phase}_{leg}.xyz" if folder in ("dft", "model") else f"{PREFIX[folder]}_{model}_{phase}_{leg}.xyz"
    return HERE / folder / name


def dist(x, i, j):
    return float(np.linalg.norm(x[i] - x[j]))


def last_accepted(legs):
    return legs[legs.accepted == True].groupby(["model", "phase", "leg"]).tail(1).set_index(  # noqa: E712
        ["model", "phase", "leg"])


def dG(acc, model, col="G_eV"):
    G = acc.loc[model][col].unstack("leg") if col == "G_eV" else acc.loc[model].eval(col).unstack("leg")
    return pd.DataFrame({"bar": (G.ts - G.sm) * EV, "rxn": (G["prod"] - G.sm) * EV})


def g_qrrho(e_elec, freqs, T=298.15, floor=50.0, fold=100.0):
    """Vibration-only harmonic G (eV): modes below the floor raised to it, imaginary modes up to `fold`
    folded to the floor, larger imaginary modes (the TS mode) dropped."""
    nu = np.array([float(v) for v in freqs.split(";")])
    nu = np.where(nu < 0, np.where(-nu <= fold, floor, np.nan), np.maximum(nu, floor))
    e = nu[~np.isnan(nu)] * 1.239841984e-4          # cm-1 -> eV
    kT = 8.617333262e-5 * T
    return e_elec + np.sum(e / 2 + kT * np.log1p(-np.exp(-e / kT)))


# Data
dfte = pd.read_csv(HERE / "dft/dft_free_energies.csv").set_index(["phase", "leg"])
G = dfte.G_eV.unstack("leg")
dft = pd.DataFrame({"bar": (G.ts - G.sm) * EV, "rxn": (G["prod"] - G.sm) * EV})
dxyz = {(p, leg): read_xyz(geom("dft", "dft", p, leg))[2] for p in PHASES for leg in ("sm", "ts", "prod")}
legs = {b: pd.read_csv(HERE / b / "legs.csv") for _, b in BASES}
acc = {b: last_accepted(l) for b, l in legs.items()}
uma = acc["model"]
mxyz = {(m, p, leg): read_xyz(geom("model", m, p, leg))[2]
        for m in ("model_moe", "model_compact") for p in PHASES for leg in ("sm", "ts", "prod")}

# Table IV
t4 = {"bar": {}, "rxn": {}, "rCCl_ts": {}, "rCCl_prod": {}}
for p in PHASES:
    t4["bar"][p] = [dft.bar[p]] + [dG(uma, m).bar[p] for m in ("model_moe", "model_compact")]
    t4["rxn"][p] = [dft.rxn[p]] + [dG(uma, m).rxn[p] for m in ("model_moe", "model_compact")]
    # model TS r(C-Cl): from the TS files (the published legs have no TS r_CCl; model/REGENERATED.md)
    t4["rCCl_ts"][p] = [dist(dxyz[p, "ts"], 0, 1)] + [dist(mxyz[m, p, "ts"], 0, 1) for m in ("model_moe", "model_compact")]
    t4["rCCl_prod"][p] = [dist(dxyz[p, "prod"], 0, 1)] + [uma.r_CCl[m, p, "prod"] for m in ("model_moe", "model_compact")]
for key, title, nd in [("bar", "dG_barrier (kcal/mol)", 1), ("rxn", "dG_rxn (kcal/mol)", 1),
                       ("rCCl_ts", "r(C-Cl) at the TS (A)", 2), ("rCCl_prod", "r(C-Cl) of the product (A)", 2)]:
    print(f"Table IV, {title}: recomputed [paper]\n  {'phase':13s}" + "".join(f"{c:>{2 * nd + 14}s}" for c in ("DFT-SMD", "MoE", "compact")))
    for p in PHASES:
        print(f"  {p:13s}" + "".join(q(v, P4[key][p][i], nd, f"Table IV {key} {p} {c}") for i, (v, c) in
                                     enumerate(zip(t4[key][p], ("DFT-SMD", "MoE", "compact")))))

# Sec. III.E text
err = {m: dG(uma, m).loc[SOLV] - dft.loc[SOLV] for m in ("model_moe", "model_compact")}
print(f"\ntext: DFT-SMD barrier gas {q(dft.bar['gas'], 43.1, 1, 'text')}, water {q(dft.bar['water'], 23.1, 1, 'text')}; "
      f"dG_rxn gas {q(dft.rxn['gas'], 41.2, 1, 'text')}, water {q(dft.rxn['water'], -11.8, 1, 'text DFT-SMD dG_rxn water')}")
print(f"text: MoE, six solvents: max |barrier error| {within(err['model_moe'].bar.abs().max(), 0.9, 'text MoE barrier')}, "
      f"max |dG_rxn error| {within(err['model_moe'].rxn.abs().max(), 1.4, 'text MoE dG_rxn')}")
dr = {k: max(abs(dist(mxyz["model_moe", p, "ts"], 0, i) - dist(dxyz[p, "ts"], 0, i)) for p in PHASES)
      for k, i in (("C-N", 5), ("C-Cl", 1))}
dr["C-N published r_CN"] = max(abs(uma.r_CN["model_moe", p, "ts"] - dist(dxyz[p, "ts"], 0, 5)) for p in PHASES)
print("text: MoE TS vs DFT-SMD, seven phases, max |dr| (A): " + ", ".join(f"{k} {v:.4f}" for k, v in dr.items())
      + f" [within 0.01]{'' if max(dr.values()) < 0.01 else '*'}")
c = err["model_compact"]
print(f"text: compact, six solvents: barriers too high by {span(c.bar.min(), c.bar.max(), '0.7-2.0', 'text compact barrier')}; "
      f"dG_rxn in chloroform and the polar solvents too negative by "
      f"{span(-c.rxn.drop('hexane').max(), -c.rxn.drop('hexane').min(), '6.6-9.8', 'text compact dG_rxn')}")
print("text: product r(C-Cl) DFT / MoE / compact: " + ", ".join(
    f"{p} {t4['rCCl_prod'][p][0]:.2f}/{t4['rCCl_prod'][p][1]:.2f}/{t4['rCCl_prod'][p][2]:.2f}" for p in SOLV))
# GFN2-xTB/ALPB in place of AnySolv at the MoE stationary points: G - dsolv_anysolv + dsolv_xtb per leg
xtb = dG(uma, "model_moe", "G_eV - dsolv_anysolv_eV + dsolv_xtb_eV").loc[SOLV] - dft.loc[SOLV]
print(f"text: GFN2-xTB/ALPB swap, MoE geometries, six solvents: barriers too high by "
      f"{span(xtb.bar.min(), xtb.bar.max(), '1.2-6.0', 'text xTB barrier')}; dG_rxn errors {xtb.rxn.min():+.2f} to "
      f"{xtb.rxn.max():+.2f}, max |error| {q(xtb.rxn.abs().max(), 9.6, 1, 'text xTB dG_rxn')}(either direction)")

# Table V
print("\nTable V (errors vs DFT-SMD, kcal/mol): gas dG_barrier, gas dG_rxn | solvated barrier MSE, MAE | "
      "dG_rxn MSE, MAE (footnote a excluded); recomputed [paper]")
t5 = {}
for label, b in BASES:
    gas = dG(acc[b], "model_moe" if b == "model" else "none").loc["gas"] - dft.loc["gas"]   # base alone
    e = dG(acc[b], "model_moe").loc[SOLV] - dft.loc[SOLV]
    er = e.rxn.drop(NH_CL.get(label, []))
    t5[label] = (gas.bar, gas.rxn, e.bar.mean(), e.bar.abs().mean(), er.mean(), er.abs().mean(), e.bar.min())
    print(f"  {label:15s}" + "".join(q(v, P5[label][i], 1, f"Table V {label} col{i + 1}") for i, v in enumerate(t5[label][:6]))
          + (f"  (dG_rxn over {len(er)} solvents)" if len(er) < 6 else ""))
mae = lambda i: [t5[k][i] for k in OMOL25]  # noqa: E731
print(f"text: OMol25-trained bases, gas: max |error| {within(max(abs(t5[k][i]) for k in OMOL25 for i in (0, 1)), 0.5, 'text OMol25 gas')}; "
      f"solvated MAE barriers {span(min(mae(3)), max(mae(3)), '0.3-0.5', 'text OMol25 MAE barrier')}, "
      f"dG_rxn {span(min(mae(5)), max(mae(5)), '0.4-0.8', 'text OMol25 MAE dG_rxn')}")
print(f"text: gas barrier AIMNet2 {q(t5['AIMNet2'][0], 4.7, 1, 'text AIMNet2')}, AIMNet2-NSE "
      f"{q(t5['AIMNet2-NSE'][0], 9.4, 1, 'text AIMNet2-NSE')}; smallest "
      f"solvated barrier error {t5['AIMNet2'][6]:+.2f} / {t5['AIMNet2-NSE'][6]:+.2f} [too high in every solvent]")
# products in which Cl- sits on an N-H (min r(Cl...H-N) < 2.5 A) instead of the backside ion pair
nh = []
for label, b in [("DFT-SMD", "dft")] + BASES:
    for p in SOLV:
        x = read_xyz(geom(b, "dft" if b == "dft" else "model_moe", p, "prod"))[2]
        r_hcl = min(dist(x, 1, i) for i in (6, 7, 8))
        if r_hcl < 2.5:
            nh.append(f"{label} {p} (r(Cl...H) {r_hcl:.2f} A)")
print(f"text: MoE products with an N-H...Cl- ion pair [AIMNet2 methanol, acetonitrile, DMSO; AIMNet2-NSE hexane]: {nh}")

# Checks
bad = [f"dft_{p}_{leg}" for p, leg in dxyz   # DFT: comment-line energy (Hartree) = dft_free_energies.csv
       if abs(float(read_xyz(geom("dft", "dft", p, leg))[0][-1]) - dfte.E_elec_Ha[p, leg]) > 1e-9]
reg = last_accepted(pd.read_csv(HERE / "model/regenerated_legs.csv"))
n, nofile = 0, []
for _, d in BASES:
    rows = {"legs.csv": uma, "regenerated_legs.csv": reg} if d == "model" else {None: acc[d]}
    keys = set()
    for f in sorted((HERE / d).glob("*.xyz")):
        words, meta, x = read_xyz(f)
        keys.add(tuple(words[1:4]))
        if meta.get("source") in ("gas_regenerated", "water_rerun"):   # no legs row; compared below
            continue
        r = rows[meta.get("source")].loc[tuple(words[1:4])]
        n += 1
        if not (f == geom(d, *words[1:4]) and abs(dist(x, 0, 5) - r.r_CN) < 6e-4 and abs(float(meta["E_elec_eV"]) - r.E_elec_eV) < 1e-6
                and (np.isnan(r.r_CCl) or abs(dist(x, 0, 1) - r.r_CCl) < 6e-4)):
            bad.append(f.name)
    nofile += [f"{d}: {k}" for k in acc[d].index if k not in keys]
print(f"\ncheck: 21 DFT and {n} MLIP xyz files against the CSV row they belong to: {len(bad)} disagree {bad}; "
      f"accepted legs without a file: {nofile}")
dg = [abs(g_qrrho(r.E_elec_eV, r.freqs_cm) - r.G_eV) for l in list(legs.values()) + [reg.reset_index()]
      for _, r in l[l.accepted == True].iterrows()]  # noqa: E712
de = max((l.E_elec_eV - l.E_gasUMA_eV - l.dsolv_anysolv_eV).abs().max() for l in legs.values())
print(f"check: G_eV recomputed from E_elec_eV and freqs_cm for {len(dg)} accepted rows: max |dG| "
      f"{max(dg) * EV:.4f} kcal/mol; max |E_elec - E_gas - dsolv| {de:.1e} eV")
# regenerated / re-run model structures vs the published legs (model/REGENERATED.md)
print("check: model sm/ts structures that are not the published ones: r_CN file vs published (A), G regenerated - "
      "published (kcal/mol)")
for m in ("model_moe", "model_compact"):
    for p in PHASES:
        for leg in ("sm", "ts"):
            words, meta, x = read_xyz(HERE / f"model/{m}_{p}_{leg}.xyz")
            pub = uma.loc[(m, p, leg)]
            s = f"  {m:13s} {p:12s} {leg:3s} {meta['source']:20s} r_CN {dist(x, 0, 5):.3f} vs {pub.r_CN:.3f}"
            if meta["source"] == "regenerated_legs.csv":
                s += f", dG {(reg.G_eV[m, p, leg] - pub.G_eV) * EV:+.3f}"
            elif "E_elec_eV" in meta:
                s += f", dE_elec {float(meta['E_elec_eV']) - pub.E_elec_eV:+.1e} eV"
            print(s)

print(f"\n{len(MISS)} numbers do not round to the value quoted in the paper:" + "".join(f"\n  {m}" for m in MISS))
