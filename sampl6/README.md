# sampl6 — SAMPL6 octanol/water partition coefficients (Sec. III.C, Table A6)

This folder covers the 11 compounds of the SAMPL6 log P challenge. Their experimental values are from Işık et al.
(*J. Comput.-Aided Mol. Des.* **2020**, *34*, 405), and the challenge is assessed by Işık et al. (*J. Comput.-Aided
Mol. Des.* **2020**, *34*, 335). We generated conformers with RDKit (seed 20260830), optimized them with GFN2-xTB in
the gas phase, and kept up to the eight lowest, 47 in total. Each conformer was relaxed in water and in 1-octanol
with UMA-S-1.2 + AnySolv (BFGS, maximum force 0.01 eV/Å), and its vibrational free energy G_{i,s} was then computed
from a finite-difference Hessian at 298.15 K. Frequencies below 50 cm⁻¹, including imaginary ones up to 50 cm⁻¹,
were raised to 50 cm⁻¹; no relaxed structure had a larger imaginary mode. Translational and rotational terms are
omitted. This was done for `model_moe` and `model_compact`, the values of the `model` columns.

log P = (G_water − G_octanol) / (kT ln 10), with kT at 298.15 K, so a positive value favors octanol. G_s is either
the lowest conformer free energy in solvent s, so the two solvents may pick different conformers ("lowest"), or the
Boltzmann ensemble free energy G_s = −kT ln Σ_i exp(−G_{i,s}/kT) ("ensemble").

## Files

| file | content |
|---|---|
| `molecule_ID_and_SMILES.csv` | the 11 compounds and their isomeric SMILES (the SAMPL6 challenge input). SM09 and SM12 are listed as HCl salts; the neutral parent (without `.Cl`) was used |
| `logP_experimental_values.csv` | experimental log P (`logP mean`, pH-metric) and its standard error (`logP SEM`), copied unchanged from the SAMPL6 repository (github.com/samplchallenges/SAMPL6, `physical_properties/logP/experimental_data/`) |
| `geometries/<mol>/conf<NN>.xyz` | the 47 GFN2-xTB gas-phase starting conformers. Comment line: GFN2-xTB energy relative to `conf00` (kcal/mol) |
| `conformers.csv` | one row per conformer × solvent × model (188): relaxed energy and free energy |
| `logP.csv` | one row per compound × model (22): predicted log P with both estimators, next to experiment |
| `reproduce_sampl6.py` | recomputes the paper's numbers (see below) |

## Columns

`conformers.csv`

| column | meaning |
|---|---|
| `mol`, `conf` | compound and conformer index; the starting structure is `geometries/<mol>/conf<NN>.xyz` |
| `model` | `model_moe` or `model_compact` |
| `phase` | `water` or `1-octanol` |
| `E_xtb_gas_rel_kcal` | GFN2-xTB gas-phase energy of the starting conformer relative to `conf00`, kcal/mol (the xyz comment) |
| `E_opt_eV` | E(UMA-S-1.2) + ΔE(AnySolv) at the relaxed geometry, eV |
| `G_eV` | G_{i,s} = `E_opt_eV` + zero-point energy + thermal vibrational energy − T S_vib, eV |
| `zpe_eV` | zero-point energy, eV |
| `n_modes` | vibrational modes (3N − 6) |
| `n_floored` | real modes below 50 cm⁻¹ raised to 50 cm⁻¹ |
| `n_imag_small`, `n_imag_big` | imaginary modes up to 50 cm⁻¹ (treated as 50 cm⁻¹) and above 50 cm⁻¹ (none) |
| `converged`, `opt_steps`, `status` | BFGS outcome (all converged; at most 190 steps) |
| `seconds` | wall time of the row, s (not used in the paper) |

`logP.csv`

| column | meaning |
|---|---|
| `mol`, `model` | compound, model (`model_moe` or `model_compact`) |
| `logP_exp`, `sem` | experimental log P and its SEM, as in `logP_experimental_values.csv` |
| `logP_lowest`, `logP_ensemble` | predicted log P from the lowest conformer and from the Boltzmann ensemble (rounded to 3 decimals) |
| `n_conf_pairs` | conformers relaxed in both solvents |
| `n_imag_big_total`, `n_unconverged` | imaginary modes above 50 cm⁻¹ and unconverged relaxations over the compound (all 0) |

## Reproduce

`python reproduce_sampl6.py` (numpy, pandas) recomputes log P from `G_eV`, which matches `logP.csv` to 0.0005. It
then prints the RMSEs against experiment of Sec. III.C and Table A6, and checks every xyz file against
`conformers.csv`. As an approximate check of the paper's statement that translational and rotational terms change log P
by less than 0.01, it adds the rotational free energy of the starting conformers. The translational term is the same
for every conformer and cancels.

## Not included

- The relaxed water and 1-octanol geometries, which the driver did not save, and the vibrational frequencies (only
  the mode counts are stored).
- The RMSEs of the SAMPL6 submissions quoted for comparison (0.49–1.42 for the six DFT-SMD submissions, 0.38 for the
  best physical method). These are in Işık et al. 2020 (*34*, 335) and the challenge repository.
- The driver code.

## AI Use

Portions of the code in this repository were generated with the assistance of AI-based coding tools. All numerical values reported in the associated manuscript have been independently checked and verified by the authors.
