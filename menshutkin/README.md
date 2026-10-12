# menshutkin — Menshutkin reaction NH₃ + CH₃Cl (Sec. III.E, Tables IV and V, App. C)

This folder covers the stationary points of NH₃ + CH₃Cl → H₃NCH₃⁺ + Cl⁻ (9 atoms, charge 0, singlet) in the gas phase
and six solvents: n-hexane, chloroform, methanol, acetonitrile, DMSO, and water.

- DFT-SMD is ωB97M-V/def2-TZVPD with SMD (ORCA 6.0.1), the level of theory of the training labels.
- The model rows are UMA-S-1.2 plus AnySolv: `model_moe` and `model_compact` in Table IV, and the MoE model on four
  other gas-phase MLIPs in Table V.
- Each structure was relaxed on its own potential, starting from the DFT-SMD stationary point of the same phase and
  leg, with the protocol of App. C. The exceptions are listed under Runs.
- Gas-phase rows are the MLIP alone, including the `model_moe` and `model_compact` gas rows.

`python reproduce_menshutkin.py` (numpy, pandas) recomputes:

- every cell of Tables IV and V;
- the Menshutkin numbers in the text of Sec. III.E;
- `G_eV` of every model row from its frequencies.

It also checks every xyz file against the CSV row it belongs to.

Identifiers:

- Phases are `gas`, `hexane`, `chcl3`, `methanol`, `acetonitrile`, `dmso`, and `water`.
- Legs are `sm` (reactant complex), `ts` (transition state), and `prod` (ion-pair product).
- The atom order in every xyz file is [C, Cl, H, H, H, N, H, H, H]. r(C–N) is the distance between atoms 1 and 6,
  and r(C–Cl) between atoms 1 and 2.

The free energies use the `G_eV` of the last accepted row per (model, phase, leg). ΔG‡ = G(ts) − G(sm) and ΔG_rxn =
G(prod) − G(sm); the reference is the reactant complex of the same phase, without separated reactants or a
standard-state term. Conversions: 1 eV = 23.0605 kcal/mol and 1 Hartree = 627.5095 kcal/mol.

## Files

| file | content |
|---|---|
| `dft/dft_<phase>_<leg>.xyz` | the 21 DFT-SMD stationary points. Comment line: the ORCA job and the electronic energy in Hartree (SMD energy in the solvents) |
| `dft/dft_free_energies.csv` | `phase`, `leg`, `E_elec_Ha` (= the xyz comment line), and `G_eV`, the free energy in the convention below |
| `model/legs.csv` | UMA-S-1.2 + AnySolv, both models (Table IV). Every optimization attempt of the run behind Table IV: 52 rows, failed attempts included |
| `model/model_<model>_<phase>_<leg>.xyz` | the 42 model stationary points. Comment line: `umas12 <model> <phase> <leg> E_elec_eV=<…> source=<…>`. Only the 14 products are the structures of the `legs.csv` rows. The reactant complexes and transition states were regenerated; `source` says how, and `model/REGENERATED.md` gives the details |
| `model/regenerated_legs.csv` | the legs of the 20 regenerated reactant complexes and transition states (`source=regenerated_legs.csv`) |
| `model/REGENERATED.md` | provenance of the model reactant complexes and transition states, and how they compare with the published legs |
| `macepolar/`, `orbmolv2/`, `aimnet2/`, `aimnet2nse/` | Table V: the same calculations with MACE-POLAR-1-M (`polar1m`), OrbMol-v2, AIMNet2, and AIMNet2-NSE as the gas-phase MLIP. Each folder has a `legs.csv` (every attempt) and the accepted stationary points `<base>_<model>_<phase>_<leg>.xyz`, with comment line `<base> <model> <phase> <leg> E_elec_eV=<…>`. `model = none` is the MLIP alone in the gas phase; it gives the same structures and energies as the gas rows of the two models. **Extra:** the `model_compact` rows and files on these four MLIPs are not in the paper |

## Columns of `legs.csv`

| column | meaning |
|---|---|
| `base` | gas-phase MLIP (`polar1m`, `orbmolv2`, `aimnet2`, `aimnet2nse`). This column is absent from `model/legs.csv`, which is UMA-S-1.2 |
| `model`, `phase`, `leg` | `model_moe`, `model_compact`, or `none`; the phase; `sm`, `ts`, or `prod` |
| `accepted` | the attempt passed the checks of App. C. Use the last accepted row per (model, phase, leg). Rows with `rounds` = 0 and no energies are TS attempts that returned no structure |
| `rounds` | minima: optimization rounds, i.e. the first optimization plus the re-optimizations after a displacement along an imaginary mode (at most 4); TS: Sella strategies tried (up to 6) |
| `r_CN`, `r_CCl` | distances (Å). `r_CCl` is empty in the `sm` and `ts` rows of `model/legs.csv`; Table IV takes r‡(C–Cl) from the TS files |
| `n_imag_big`, `n_imag_small`, `n_floored` | imaginary modes above the acceptance threshold, which include the TS mode; smaller imaginary modes, folded to the floor; real modes below 50 cm⁻¹, raised to it |
| `G_eV`, `E_elec_eV`, `zpe_eV` | free energy (below), electronic energy on the solvated potential, and zero-point energy (eV) |
| `E_gasUMA_eV` | gas-phase energy of the row's MLIP at the same geometry, in eV. The name is historical: on the other MLIPs it is their energy, not UMA's. `E_elec_eV = E_gasUMA_eV + dsolv_anysolv_eV` |
| `dsolv_anysolv_eV` | AnySolv solvation energy (eV; 0 in the gas phase) |
| `dsolv_xtb_eV` | GFN2-xTB/ALPB solvation energy at the same geometry (eV; xtb 6.7.1, ALPB minus gas single point). The GFN2-xTB/ALPB comparison in the text replaces the AnySolv term per leg: `G_eV − dsolv_anysolv_eV + dsolv_xtb_eV` |
| `imag_freqs_cm`, `freqs_cm` | the imaginary frequencies, and all 21 vibrational frequencies (cm⁻¹, imaginary as negative). They come from a finite-difference Hessian of the model forces, δ = 0.01 Å |
| `seconds` | wall time of the attempt (s) |
| `seed_file` | (other MLIPs) start structure, as a path relative to this folder. It is the DFT-SMD stationary point of the same phase and leg, or, for a TS that failed from it, the accepted TS of the neighbouring phase on the same MLIP |
| `dtype`, `hessian_delta_A`, `hessian_asym_eVA2` | (OrbMol-v2, AIMNet2, AIMNet2-NSE) MLIP precision; Hessian step (Å); max \|H − Hᵀ\| of the unsymmetrized Hessian of the combined forces (eV/Å²). `float64(partial)` means that AIMNet2's dispersion and neighbour-list kernels stay float32 |

The free energy is the same for the models and DFT-SMD: the vibration-only harmonic free energy at 298.15 K,
G = E_elec + Σ [hν/2 + kT ln(1 − e^(−hν/kT))], without translation or rotation.

- Modes below 50 cm⁻¹ are raised to 50 cm⁻¹.
- Imaginary modes up to 100 cm⁻¹ are folded to 50 cm⁻¹.
- Larger imaginary modes, i.e. the TS mode, are left out.

ORCA's printed free energies, which include translation and rotation and drop the imaginary modes, are not used.
They give barriers 0.6–3.1 kcal/mol lower.

## Runs

Transition states: pre-relaxation with r(C–N) fixed at the DFT-SMD value (BFGS, 0.02 eV/Å), then Sella order 1 to
0.01 eV/Å. A TS is accepted with exactly one imaginary mode above 100 cm⁻¹ and 1.7 Å < r(C–N) < 2.8 Å. On a failure,
up to six Sella variants are tried.

Minima: BFGS to 0.01 eV/Å. Any imaginary mode above 50 cm⁻¹ triggers a displacement of 0.15 Å along it and a
re-optimization, up to four rounds. After the fourth round, imaginary modes up to 100 cm⁻¹ are accepted.

The UMA-S-1.2 compact products were seeded from the MoE product of the same phase (`model/REGENERATED.md`).

On the other MLIPs, the AnySolv correction (float32, presets `fast_gpu` for MoE and `fast` for compact) was added to
the MLIP with ASE's `SumCalculator`. The MLIP checkpoints were:

| MLIP | checkpoint (sha256) | precision |
|---|---|---|
| MACE-POLAR-1-M | `MACE-POLAR-1-M.model` (`fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`), mace-torch 0.3.16 | float64 |
| OrbMol-v2 | `orbmol-v2-teqabfhg-20260523.ckpt` (`3dbef70b5fdc9124392181b0d8686c79f1c02bf5fefcb878a4eaa51d03e4f5e8`), orb-models 0.7.0 | float64 |
| AIMNet2 | `aimnet2_wb97m_d3_0.pt` (`f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28`), aimnet 0.2.0 | float64 (partial) |
| AIMNet2-NSE | `aimnet2nse_wb97m_0.pt` (`b98b3449b2a3765de6a188edff9344a38cc72df75f9d9bf68f4b898a253c5864`), aimnet 0.2.0 | float64 (partial) |

Notes on these runs (all visible in `legs.csv`):

- **Seeds from a neighbouring phase:** some TS searches failed from the DFT-SMD seed and were seeded from the accepted
  TS of a neighbouring phase on the same MLIP (`seed_file`):
  - MACE-POLAR-1-M + compact: methanol and DMSO, from acetonitrile;
  - AIMNet2 + compact: methanol from acetonitrile, and water from DMSO;
  - AIMNet2-NSE + MoE: acetonitrile from methanol, DMSO from acetonitrile, and water from DMSO.
- **Missing compact reactant complexes on MACE-POLAR-1-M:** the compact reactant complexes in methanol and water were
  never accepted. In three attempts each, an imaginary mode of 103–107 cm⁻¹ remained after four rounds, above the
  100 cm⁻¹ tolerance. So those two `sm` files are missing, and the compact model has no ΔG in those solvents on that
  MLIP. The paper does not use it.
- **N–H⋯Cl⁻ products:** in four cases, the DFT-SMD product seed relaxed to Cl⁻ hydrogen-bonded to an N–H of the
  ammonium group (r(Cl⋯H) 1.8–2.1 Å, against 4.69 Å or more in every other product): AIMNet2 + MoE in methanol,
  acetonitrile, and DMSO, and AIMNet2-NSE + MoE in n-hexane. Footnote a of Table V leaves these solvents out of the
  ΔG_rxn statistics. The script finds them from the geometries.

## Not included

- The DFT-SMD frequencies, Hessians, and ORCA input and output files of the stationary points. `G_eV` of the DFT-SMD
  points therefore cannot be recomputed from this folder.
- The geometries of the reactant complexes and transition states of the run behind Table IV, which were never saved
  (see `model/REGENERATED.md`), and the geometries of rejected attempts.
- The relaxed scans along r(C–N) and the vertical solvation profiles along the gas-phase path, which the paper does
  not use.
- MACE-POLAR-1-L (MoE; gas phase, n-hexane, water), a check that the paper does not report.
- Diagnostics of the runs on the other MLIPs: their gas-phase single points and ORCA's at the DFT-SMD geometries, the
  gas/solvation/thermal decomposition, and the pre-run checks.
- The ORCA gas-phase and SMD(water) single points at the compact model's water product, behind the 9.7 kcal/mol
  over-solvation quoted in Sec. III.E.
- The driver code.

## AI Use

Portions of the code in this repository were generated with the assistance of AI-based coding tools. All numerical values reported in the associated manuscript have been independently checked and verified by the authors.
