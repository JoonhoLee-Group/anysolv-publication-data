# Provenance of the UMA-S-1.2 model stationary points

`legs.csv` is the attempt log of the run behind Table IV. That run saved the geometries of its 14 products,
but not those of its reactant complexes (`sm`) and transition states (`ts`). The 28 `sm`/`ts` files here
were therefore produced afterwards with the same protocol (seeds, optimizer thresholds, Hessian,
mode-following, Sella saddle refinement). The free energies of Table IV are always the `legs.csv` values.

The comment line of each file names its `source`:

| `source=` | files | origin | `E_elec_eV` in the comment line |
|---|---|---|---|
| `legs.csv` | the 14 products | the run behind Table IV | its `legs.csv` row |
| `regenerated_legs.csv` | `sm`, `ts` in hexane, chcl3, methanol, acetonitrile, dmso (20) | regenerated 2026-09-10 on an A100 | its `regenerated_legs.csv` row |
| `gas_regenerated` | `sm`, `ts` in the gas phase (4) | regenerated 2026-09-30 on a CPU | quoted below |
| `water_rerun` | `sm`, `ts` in water (4) | a later run of the same driver | not available |

The model r‡(C–Cl) of Table IV is measured in these TS files, because the TS rows of `legs.csv` have no
`r_CCl`. Every regenerated or re-run TS reproduces the published r(C–N) to 0.001 Å.
`reproduce_menshutkin.py` prints each file's r(C–N) next to the published one, together with the free-energy
difference for the regenerated files.

## Regenerated reactant complexes and transition states (20 files, `regenerated_legs.csv`)

The driver imported the published functions unchanged and ran them only for the 20 missing legs. Each
leg was seeded from the DFT-SMD stationary point of its phase. `regenerated_legs.csv` has the
`legs.csv` columns, with `r_CCl` filled in. It holds 22 rows, 20 of them accepted; take the last accepted
row per key.

- All 10 transition states reproduce the published r(C–N) to 0.001 Å and the published free energy to
  0.05 kcal/mol. From the DFT-SMD seed, `model_compact_dmso_ts.xyz` collapsed to the reactant complex
  in all six Sella strategies, twice. It was then obtained from the regenerated
  `model_compact_acetonitrile_ts.xyz` as the seed. It is the same saddle as the published one: r(C–N)
  2.100 Å, imaginary mode −812.3 vs −811.1 cm⁻¹, free energy within 0.02 kcal/mol.
- 7 of the 10 reactant complexes reproduce r(C–N) to 0.01 Å. These flat complexes have soft imaginary
  modes, so the end point of the mode-following is sensitive to Hessian noise. Three differ:
  - `model_moe_methanol_sm.xyz`: 3.540 vs 3.551 Å published, free energy within 0.03 kcal/mol.
  - `model_compact_acetonitrile_sm.xyz`: 3.604 vs 3.626 Å, free energy within 0.01 kcal/mol.
  - `model_compact_dmso_sm.xyz`: 3.288 vs 3.518 Å. This is a tighter complex with no imaginary modes,
    0.52 kcal/mol above the published complex. With it and the regenerated TS, the compact DMSO barrier
    and reaction free energy would be 24.30 and −21.17 kcal/mol, instead of the published 24.83 and
    −20.65 (Table IV: 24.8, −20.7). The published values stand, and this file is not the structure
    behind them.

## Gas-phase reactant complex and transition state (4 files, `gas_regenerated`)

The gas legs are UMA-S-1.2 alone, so `model_moe_gas_{sm,ts}.xyz` and `model_compact_gas_{sm,ts}.xyz` are
the same structures. They were regenerated on a CPU with the same UMA-S-1.2 checkpoint (sha256
`ba5c0d912efa22dc238e5fb1b5b7f66ee2e68c48c1b95b7cfd7fe1da5938398b`, fairchem 2.19.0, ASE 3.29.0, Sella
2.6.0, task `omol`, charge 0, multiplicity 1), seeded from `../dft/dft_gas_{sm,ts}.xyz`:

- `sm`: the DFT-SMD gas geometry is already converged on UMA-S-1.2 (max |F| 0.0065 eV/Å < 0.01 eV/Å).
  BFGS therefore takes no step, so the published reactant complex is that geometry, and the file is
  identical to `../dft/dft_gas_sm.xyz`. E = −15146.929713 eV (published −15146.929712), r(C–N) 3.301 Å,
  soft mode −27.4 cm⁻¹ (published −27.4).
- `ts`: Sella order 1 to 0.01 eV/Å from the DFT-SMD gas TS. The fixed-C–N pre-relaxation did not
  converge with ASE's `FixBondLength` here and was skipped. E = −15145.287314 eV (published
  −15145.287315), r(C–N) 1.813 Å, one imaginary mode −464.9 cm⁻¹ (published −464.9). This is the same
  saddle.

## Water reactant complex and transition state (4 files, `water_rerun`)

`model_{moe,compact}_water_{sm,ts}.xyz` were written by a later run of the same driver for the water
`sm` and `ts` legs. That run was the first to save these geometries, and it came after the `legs.csv`
snapshot behind Table IV. Its attempt rows are not included.

- The transition states agree with the published rows: r(C–N) 2.120 and 2.110 Å, imaginary mode −663.9
  vs −663.6 cm⁻¹ (MoE) and −840.9 vs −839.6 cm⁻¹ (compact).
- The reactant complexes differ slightly in r(C–N): 3.652 vs 3.643 Å (MoE), 3.651 vs 3.655 Å (compact).
- That run gave water ΔG‡ and ΔG_rxn of 23.99 and −12.62 kcal/mol (MoE) and 25.13 and −21.25 (compact).
  `legs.csv` gives 23.96 and −12.66, and 25.17 and −21.25. Table IV quotes the `legs.csv` values (24.0,
  −12.7; 25.2, −21.2).

## Products (14 files, `legs.csv`)

The products are the structures of the published rows: r(C–N), r(C–Cl), and `E_elec_eV` match.
The compact products were seeded from the converged MoE product of the same phase rather than from the
DFT-SMD product. On the compact potential they relax to the separated ion pair (r(C–Cl) 4.49–4.60 Å
in chloroform and the polar solvents). An earlier pass seeded from the DFT-SMD products ended in the same
basin, with the same solvation energies to within 0.01 eV.
