# model_vs_dftsmd — model vs DFT-SMD single points (Sec. III.A)

AnySolv against our own DFT-SMD single points (ωB97M-V/def2-TZVPD, ORCA 6.0.1, the level of theory of the training labels)
at the database geometries of MNSol, Solv@TUM, and FlexiSol. All energies are vertical; no geometry was relaxed.

`python reproduce_solvent_discrimination.py` (numpy, pandas) recomputes the solvent-discrimination statistics of Sec. III.A:
31 trained / 56 unseen Solv@TUM solvents, 171 / 143 eligible solutes, pooled centered R² 0.998 / 0.944 (MoE) and
0.991 / 0.898 (compact), median per-solute R² 0.9996 / 0.986 and 0.9992 / 0.956, and the null-model R² of 0.88. Both R²
are coefficients of determination about y = x with no fitted parameters; the definitions are in the script's docstring.

## Files

| file | content |
|---|---|
| `per_pair/forces_<set>_<model>.csv` | one row per solute–solvent pair, `<set>` = `mnsol`, `solvatum`, `solvatum_topup`, `flexisol`; `<model>` = `model_moe`, `model_compact` |
| `dft_smd_reference/dft_smd_records_<set>.csv` | the DFT-SMD reference: `E_gas_Ha`, `E_smd_Ha`, and `dG_smd_kcal` = ΔE (identical to `de_dft_kcal` in `per_pair/`) |
| `summary_model_vs_dftsmd.csv` | one row per set and model: energy MAE (kcal/mol), bias, R², force-component and force-magnitude MAE (eV/Å), cosine statistics |
| `reproduce_solvent_discrimination.py` | the solvent-discrimination statistics above |

## `per_pair/` columns

| column | meaning |
|---|---|
| `name` | `<solute slug>[-<index>]-<hash8>__<solvent>`; the solute identity is the part before `__` |
| `solvent_key`, `charge`, `n_atoms` | solvent (Minnesota Solvent Descriptor Database name), total charge, number of atoms |
| `de_dft_kcal`, `de_mod_kcal` | single-point solvation energy ΔE (kcal/mol), DFT-SMD and AnySolv |
| `mae`, `rms`, `maxabs` | error of the solvation forces ΔF over all Cartesian components (eV/Å): MAE, RMSE, and maximum |
| `mag_mae` | MAE of the per-atom force magnitudes \|ΔF\| (eV/Å) |
| `cos` | cosine similarity between the predicted and reference ΔF |
| `dft_rms`, `mod_rms` | RMS of the reference and predicted ΔF (eV/Å) |

## Notes

- Solv@TUM has 4,511 in total.
- MNSol: the paper drops the 9 monatomic ions (`n_atoms = 1`), which were training labels, leaving 2,884 pairs. The `mnsol` row of the summary still includes them and therefore does not match the paper (energy MAE 0.074 vs 0.048 kcal/mol).
