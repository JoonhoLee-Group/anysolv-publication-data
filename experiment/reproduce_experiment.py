"""Recompute the comparison with experiment (Sec. III.B, Table III, Table A5) from the files in this folder.

    python reproduce_experiment.py                  # print every number next to the paper's value
    python reproduce_experiment.py --write-summary  # also rewrite summary_vs_exp.csv

Needs numpy and pandas. The DFT-SMD single points and the MNSol Delta E_sp of the models come from
../model_vs_dftsmd/. Errors are model (or DFT-SMD) minus experiment, kcal/mol; each cell is the MAE and, in
parentheses, the mean signed error (MSE).

Pairs:
  Solv@TUM  status ok and a DFT-SMD single point exists.
  FlexiSol  status ok for model_moe; model_compact is evaluated on the same 529 pairs.
  MNSol     all rows except the 9 monatomic training anchors, split by charge.
  DFT-SMD   evaluated on the model_moe pairs of each row.
  Table A5  pairs ok with both gas-phase potentials.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DFT = HERE.parent / "model_vs_dftsmd"
MODELS = ("model_moe", "model_compact")
ROWS = ("Solv@TUM neutral", "FlexiSol neutral", "MNSol neutral", "MNSol cations", "MNSol anions")

# Values quoted in the manuscript
PAPER_T3 = {
    "Solv@TUM neutral": ["0.92 (-0.27)", "0.89 (-0.24)", "0.99 (-0.48)", "0.92 (-0.26)", "0.89 (-0.22)", "0.98 (-0.45)", "0.93 (-0.24)"],
    "FlexiSol neutral": ["2.48 (+1.29)", "2.46 (+1.03)", "2.47 (+0.52)", "2.48 (+1.29)", "2.45 (+1.03)", "2.45 (+0.57)", "2.51 (+1.34)"],
    "MNSol neutral":    ["0.83 (-0.24)", "0.84 (-0.35)", "0.95 (-0.52)", "0.82 (-0.22)", "0.83 (-0.33)", "0.93 (-0.50)", "0.83 (-0.22)"],
    "MNSol cations":    ["4.77 (+3.92)", "4.60 (+3.64)", "4.56 (+3.64)", "4.78 (+3.92)", "4.60 (+3.64)", "4.60 (+3.69)", "4.79 (+3.94)"],
    "MNSol anions":     ["5.59 (+3.25)", "5.08 (+2.40)", "5.17 (+2.60)", "5.73 (+3.57)", "5.19 (+2.70)", "5.29 (+2.92)", "5.49 (+3.13)"],
}
T3_COLS = [("model_moe", "sp"), ("model_moe", "opt"), ("model_moe", "dG"),
           ("model_compact", "sp"), ("model_compact", "opt"), ("model_compact", "dG"), ("DFT-SMD", "sp")]
PAPER_A5 = {
    "Solv@TUM neutral": ["0.89 (-0.24)", "0.99 (-0.48)", "0.90 (-0.24)", "1.00 (-0.49)"],
    "FlexiSol neutral": ["2.46 (+1.03)", "2.47 (+0.52)", "2.46 (+1.02)", "2.48 (+0.53)"],
    "MNSol neutral":    ["0.84 (-0.35)", "0.95 (-0.52)", "0.84 (-0.35)", "0.96 (-0.52)"],
    "MNSol cations":    ["4.60 (+3.64)", "4.56 (+3.64)", "4.59 (+3.64)", "4.57 (+3.67)"],
    "MNSol anions":     ["5.08 (+2.40)", "5.17 (+2.60)", "5.06 (+2.38)", "5.22 (+2.60)"],
}
A5_COLS = [("umas12", "opt"), ("umas12", "dG"), ("polar1m", "opt"), ("polar1m", "dG")]

ERR_COL = {"sp": "e_sp", "opt": "e_opt", "dG": "e_dG"}
_fails = []


def cell(e):
    e = np.asarray(e, float)
    return f"{np.abs(e).mean():.2f} ({e.mean():+.2f})"


def check(label, got, paper):
    ok = got == paper
    if not ok:
        _fails.append(label)
    print(f"  {label:<44} {got:>14}   paper {paper:>14}   {'ok' if ok else 'MISMATCH'}")


# Loading
def dft_pairs(dataset, model):
    """Per-pair DFT-SMD reference and model single point, keyed by (dataset name, solvent_key)."""
    rec = pd.read_csv(DFT / "dft_smd_reference" / f"dft_smd_records_{dataset}.csv")
    rec["key"] = rec["name"].str.split("#").str[0]
    rec["pair"] = rec["smd_engrad"].str.split("/").str[-1].str.replace(".engrad", "", regex=False)
    sets = ["solvatum", "solvatum_topup"] if dataset == "solvatum" else [dataset]
    pp = pd.concat([pd.read_csv(DFT / "per_pair" / f"forces_{s}_{model}.csv") for s in sets])
    assert rec["pair"].is_unique and set(rec["pair"]) == set(pp["name"])
    out = rec[["key", "solvent_key", "pair"]].merge(pp[["name", "de_dft_kcal", "de_mod_kcal"]],
                                                     left_on="pair", right_on="name")
    assert not out.duplicated(["key", "solvent_key"]).any()
    return out[["key", "solvent_key", "de_dft_kcal", "de_mod_kcal"]]


def load_gibbs(dataset, tag):
    """Solv@TUM / FlexiSol result file; tag = model_moe, model_compact or polar1m_model_moe."""
    r = pd.read_csv(HERE / f"results_{dataset}_{tag}_gibbs.csv")
    r["ok"] = r["status"].eq("ok")
    r["e_sp"], r["e_opt"], r["e_dG"] = r["err_sp_kcal"], r["err_opt_kcal"], r["err_dG_kcal"]
    model = "model_compact" if tag == "model_compact" else "model_moe"
    d = dft_pairs(dataset, model)
    r = r.merge(d, left_on=["name", "solvent_key"], right_on=["key", "solvent_key"], how="left")
    r["e_dft"] = r["de_dft_kcal"] - r["dG_exp_kcal"]
    r["has_dft"] = r["de_dft_kcal"].notna()
    return r


def load_mnsol(tag):
    r = pd.read_csv(HERE / f"results_mnsol_{tag}.csv")
    r["ok"] = r["status"].eq("ok")
    model = "model_compact" if tag == "model_compact" else "model_moe"
    d = dft_pairs("mnsol", model)
    r = r.merge(d, left_on=["file_handle", "solvent_unisolv"], right_on=["key", "solvent_key"], how="left")
    assert r["de_dft_kcal"].notna().all()
    r["e_sp"] = r["de_mod_kcal"] - r["dG_exp_kcal"]      # single point at the MNSol geometry
    r["e_opt"] = r["dE_elec_kcal"] - r["dG_exp_kcal"]
    r["e_dG"] = r["dG_solv_kcal"] - r["dG_exp_kcal"]     # = error_kcal
    r["e_dft"] = r["de_dft_kcal"] - r["dG_exp_kcal"]
    r["has_dft"] = True
    return r


def mnsol_rows(r):
    r = r[r["n_atoms"] > 1]                              # drop the 9 monatomic training anchors
    return {"MNSol neutral": r[r["charge"] == 0], "MNSol cations": r[r["charge"] > 0],
            "MNSol anions": r[r["charge"] < 0]}


def table3_pairs():
    """{row: {model: DataFrame of the pairs of that row}}."""
    rows = {k: {} for k in ROWS}
    for m in MODELS:
        s = load_gibbs("solvatum", m)
        rows["Solv@TUM neutral"][m] = s[s["ok"] & s["has_dft"]]
        f = load_gibbs("flexisol", m)
        rows["FlexiSol neutral"][m] = f[f["ok"]]
        for k, v in mnsol_rows(load_mnsol(m)).items():
            rows[k][m] = v[v["ok"]]
    f = rows["FlexiSol neutral"]
    f["model_compact"] = f["model_compact"].merge(f["model_moe"][["name", "solvent"]])
    return rows


# Main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write-summary", action="store_true", help="rewrite summary_vs_exp.csv")
    args = ap.parse_args()
    summary = []

    # Table III
    rows = table3_pairs()
    print("Table III: MAE (MSE) against experiment, kcal/mol")
    mae = {}
    for row in ROWS:
        moe, cmp_ = rows[row]["model_moe"], rows[row]["model_compact"]
        print(f"{row}: n = {len(moe)} (model_moe), {len(cmp_)} (model_compact)")
        for (model, prot), paper in zip(T3_COLS, PAPER_T3[row]):
            pairs = moe if model == "DFT-SMD" else rows[row][model]
            e = pairs["e_dft" if model == "DFT-SMD" else ERR_COL[prot]]
            mae[row, model, prot] = np.abs(e).mean()
            check(f"{model} {prot}", cell(e), paper)
            summary.append(dict(table="III", row=row, gas_potential="none" if model == "DFT-SMD" else "UMA-S-1.2",
                                model=model, protocol=prot, n=len(e), MAE_kcal=np.abs(e).mean(), MSE_kcal=e.mean()))
    f530 = load_gibbs("flexisol", "model_compact")      # all 530 FlexiSol pairs (compact: all ok)
    print(f"  [info] FlexiSol: DFT-SMD over all 530 pairs would be {cell(f530['e_dft'])}, "
          f"model_compact dG {cell(f530['e_dG'])}; Table III uses the 529 pairs")

    # Sec. III.B text
    print("\nSec. III.B text")
    for row, paper in (("Solv@TUM neutral", "0.99"), ("MNSol neutral", "0.95"), ("FlexiSol neutral", "2.47")):
        check(f"MoE dG MAE, {row}", f"{mae[row, 'model_moe', 'dG']:.2f}", paper)
    neutral = ROWS[:3]
    for model in MODELS:
        # max - min over the row's four MAEs: Delta E_sp, Delta E_opt, Delta G_solv and DFT-SMD
        four = {row: [mae[row, model, p] for p in ("sp", "opt", "dG")] + [mae[row, "DFT-SMD", "sp"]] for row in ROWS}
        spread = {row: max(v) - min(v) for row, v in four.items()}
        worst = max(neutral, key=spread.get)
        if model == "model_moe":
            check("sp/opt/dG/DFT-SMD MAEs agree to within", f"{spread[worst]:.2f}", "0.12")
        print(f"  [info] {model}, max - min of the four MAEs: "
              + ", ".join(f"{r} {spread[r]:.3f}" for r in ROWS)
              + "; the text refers to the neutral rows")
    for row, model, paper in (("MNSol cations", "model_moe", "4.56"), ("MNSol anions", "model_moe", "5.17"),
                              ("MNSol cations", "DFT-SMD", "4.79"), ("MNSol anions", "DFT-SMD", "5.49")):
        check(f"{'MoE dG' if model == 'model_moe' else 'DFT-SMD sp'} MAE, {row}",
              f"{mae[row, model, 'dG' if model == 'model_moe' else 'sp']:.2f}", paper)
    ion_mse = {(r["model"], r["protocol"], r["row"]): r["MSE_kcal"] for r in summary
               if r["row"] in ("MNSol cations", "MNSol anions")}
    mse = [v for k, v in ion_mse.items() if k[0] != "model_compact"]
    check("ion MSE range, MoE and DFT-SMD columns", f"{min(mse):.1f}-{max(mse):.1f}", "2.4-3.9")
    print(f"  [info] unrounded {min(mse):.3f} to {max(mse):.3f}; model_compact columns "
          f"{min(v for k, v in ion_mse.items() if k[0] == 'model_compact'):.3f} to "
          f"{max(v for k, v in ion_mse.items() if k[0] == 'model_compact'):.3f}")
    ions = pd.concat([rows["MNSol cations"]["model_moe"], rows["MNSol anions"]["model_moe"]])
    lab = np.abs(ions["de_mod_kcal"] - ions["de_dft_kcal"]).mean()
    check(f"MoE vs DFT-SMD labels on the {len(ions)} MNSol ions, MAE", f"{lab:.2f}", "0.16")

    # Table A5
    print("\nTable A5: model_moe with UMA-S-1.2 or MACE-POLAR-1-M as the gas-phase potential")
    a5 = {}
    for ds in ("solvatum", "flexisol"):
        u, p = load_gibbs(ds, "model_moe"), load_gibbs(ds, "polar1m_model_moe")
        m = u.merge(p, on=["name", "solvent"], suffixes=("_u", "_p"))
        m = m[m["ok_u"] & m["ok_p"]]
        if ds == "solvatum":
            m = m[m["has_dft_u"]]
        a5["Solv@TUM neutral" if ds == "solvatum" else "FlexiSol neutral"] = m
    u, p = load_mnsol("model_moe"), load_mnsol("polar1m_model_moe")
    m = u.merge(p.drop(columns=["key", "solvent_key"]), on=["file_handle", "solvent"], suffixes=("_u", "_p"))
    m = m[m["ok_u"] & m["ok_p"]].rename(columns={"n_atoms_u": "n_atoms", "charge_u": "charge"})
    a5.update(mnsol_rows(m))
    dmax = 0.0
    for row in ROWS:
        m = a5[row]
        print(f"{row}: n = {len(m)}")
        for (base, prot), paper in zip(A5_COLS, PAPER_A5[row]):
            e = m[ERR_COL[prot] + ("_u" if base == "umas12" else "_p")]
            check(f"{'UMA-S-1.2' if base == 'umas12' else 'MACE-POLAR-1-M'} {prot}", cell(e), paper)
            summary.append(dict(table="A5", row=row, gas_potential="UMA-S-1.2" if base == "umas12" else "MACE-POLAR-1-M",
                                model="model_moe", protocol=prot, n=len(e), MAE_kcal=np.abs(e).mean(), MSE_kcal=e.mean()))
        for prot in ("opt", "dG"):
            d = np.abs(m[ERR_COL[prot] + "_p"]).mean() - np.abs(m[ERR_COL[prot] + "_u"]).mean()
            dmax = max(dmax, abs(d))
    print("\nSec. III.B text")
    check("max |MAE(MACE-POLAR) - MAE(UMA)|, opt and dG", f"{dmax:.2f}", "0.05")
    print(f"  [info] unrounded: {dmax:.4f} kcal/mol (MNSol anions, dG)")

    # Summary file
    summ = pd.DataFrame(summary)
    summ[["MAE_kcal", "MSE_kcal"]] = summ[["MAE_kcal", "MSE_kcal"]].round(4)
    path = HERE / "summary_vs_exp.csv"
    if args.write_summary:
        summ.to_csv(path, index=False)
        print(f"\nwrote {path.name} ({len(summ)} rows)")
    elif path.exists():
        old = pd.read_csv(path)
        same = old.shape == summ.shape and np.allclose(old[["n", "MAE_kcal", "MSE_kcal"]], summ[["n", "MAE_kcal", "MSE_kcal"]])
        if not same:
            _fails.append("summary_vs_exp.csv")
        print(f"\n{path.name} {'matches' if same else 'DIFFERS FROM'} the recomputed values")

    print("\nall numbers reproduce" if not _fails else f"\n{len(_fails)} MISMATCH(ES): {_fails}")
    return 1 if _fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
