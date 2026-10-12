# zwitterions — charge-separated ω-amino-acid zwitterions (Sec. III.D, Fig. 3, Table A6)

This folder covers the zwitterions H₃N⁺–(CH₂)ₙ–COO⁻ for n = 1, 2, 3, 4, 5, 7, 9, 11, …, 23: 14 chain lengths from
glycine to 24-aminotetracosanoic acid (10–76 atoms). Each is built as an extended conformer (all-anti, r(N···C)
2.4–29.9 Å) and as a cyclic, salt-bridged conformer (r(N···C) 2.4–3.8 Å). All structures are GFN2-xTB/ALPB(water)
geometries and were used unchanged for every DFT and model calculation. The reference is DFT-SMD
(ωB97M-V/def2-TZVPD, ORCA 6.0.1, the input of the training labels), with one gas-phase leg and one SMD leg per solvent
at the same geometry. The solvents are n-hexane, benzene, toluene, chloroform, acetonitrile, and water. Every energy is
a single-point solvation energy ΔE_solv = E_SMD − E_gas, and errors are model − DFT-SMD (positive = the model
under-solvates).

Models (`ckpt` columns): `model_moe` and `model_compact` are the published models. `model_moe_no_zwitterion` and
`model_compact_no_zwitterion` are earlier models trained without the charge-separation augmentation.
`model_moe_no_zwitterion` gives the dashed curves of Fig. 3, and the `model_compact_no_zwitterion` rows are not used in
the paper.

Configuration kinds (`kind`):
- `extended`: N···C distance held at its all-anti value during the xTB optimization. All six solvents.
- `cyclic`: pulled to r(N···C) = 3.1 Å (2.35 Å for glycine), then optimized freely. All six solvents.
- `scan`: four points per molecule for n ≥ 3, with r(N···C) held at values evenly spaced between the cyclic and
  extended ones. Labelled in water, chloroform, and n-hexane only, and not used in the paper.

The gas-phase SCF of the n = 17 cyclic conformer (`18-aminooctadecanoic__cyclic`) did not converge. Its six rows in
`dft_labels.csv` are `scf_failed` and have no gas energy; their SMD legs give a carboxylate Hirshfeld charge of
about −0.10 and are not used either. The conformer has no rows in `predictions.csv`, so the cyclic statistics cover 13
conformers (78 rows per model) and the extended ones 14 (84 rows).

## Files

| file | content |
|---|---|
| `configurations.csv` | the 76 configurations (14 extended, 14 cyclic, 48 scan) |
| `geometries/<mol>/<config_id>.xyz` | their structures. Comment line: `<config_id> kind=<kind> r=<r_NC>` |
| `dft_labels.csv` | DFT-SMD single points, one row per configuration × solvent (312; 306 `ok`) |
| `predictions.csv` | model and DFT-SMD solvation energies, one row per configuration × solvent × checkpoint (4 × 306) |
| `force_cosines.csv` | solvation-force agreement beyond the 24 Å effective cutoff, per checkpoint and solvent |
| `reproduce_zwitterions.py` | recomputes the paper's numbers (see below) |

## Columns

`configurations.csv`: `config_id` = `<mol>__<kind>` (scan points `<mol>__scan00` … `scan03`), `mol`, `n`, `kind`,
`r_NC` (N to carboxylate C, Å), `r_target` (the N···C constraint or pull target of the xTB optimization, Å), `R_g`
(radius of gyration, Å), `n_heavy`, `natoms`, `seed` (RDKit embedding seed), `status`, `note`.

`dft_labels.csv`

| column | meaning |
|---|---|
| `config_id`, `mol`, `n`, `kind`, `r_NC` | as in `configurations.csv` |
| `solvent_key` | solvent |
| `E_gas_Eh`, `E_smd_Eh` | gas-phase and SMD total energies, Hartree |
| `dE_solv_Eh`, `dE_solv_kcal` | E_smd − E_gas, Hartree and kcal/mol |
| `q_NH3_gas`, `q_COO_gas`, `q_NH3_smd`, `q_COO_smd` | Hirshfeld charges summed over N + 3 H and over C + 2 O, gas and SMD legs, e |
| `status`, `note` | `ok` or `scf_failed`; `note` is empty |

`predictions.csv`

| column | meaning |
|---|---|
| `config_id`, `mol`, `n`, `kind`, `r_NC`, `R_g`, `n_heavy`, `natoms` | as in `configurations.csv` |
| `solvent_key`, `ckpt` | solvent, model (see above) |
| `dE_eV`, `dE_pred_kcal` | model ΔE_solv, eV and kcal/mol |
| `E_gas_Eh`, `E_smd_Eh`, `dE_ref_kcal`, `q_NH3_gas`, `q_COO_gas` | DFT-SMD legs, ΔE_solv (kcal/mol), and gas-phase charges, copied from `dft_labels.csv` |
| `err_kcal` | `dE_pred_kcal` − `dE_ref_kcal`, kcal/mol |
| `inv_r` | 1/`r_NC`, Å⁻¹ |
| `split` | `inside` (n ≤ 11) or `beyond` (n ≥ 13); an analysis label that the paper does not use |
| `born` | Born factor 1 − 1/ε of the solvent |

`force_cosines.csv`: `ckpt`, `solvent_key`. `f_cos` is the cosine similarity between the predicted and DFT-SMD
solvation forces of a configuration, averaged over the configurations with r(N···C) > 24 Å. `f_rmse_eVA` is the force
RMSE averaged the same way, eV/Å. These are summary values from the analysis run; the per-configuration forces are not
included.

## Reproduce

`python reproduce_zwitterions.py` (numpy, pandas) prints every number of Sec. III.D and the zwitterion rows of
Table A6. These are the MAE and maximum error of the extended and cyclic conformers, the MAE beyond 24 Å (extended
conformers with r(N···C) > 24 Å: n = 19, 21, 23), the force cosine, the no-augmentation error in water, and the
gas-phase Hirshfeld charges (extended conformers with 10 ≤ r(N···C) ≤ 30 Å). It also prints the curves of Fig. 3,
which are `err_kcal` of the extended rows against `r_NC`, and checks every geometry against `configurations.csv`.

## Not included

- The DFT-SMD and model solvation forces (gradients) and the per-configuration force metrics, of which only the
  averages in `force_cosines.csv` are here.
- The ORCA input and output files and the driver code.

## AI Use

Portions of the code in this repository were generated with the assistance of AI-based coding tools. All numerical values reported in the associated manuscript have been independently checked and verified by the authors.
