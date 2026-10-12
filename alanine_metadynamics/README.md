# alanine_metadynamics — Ace-Ala-NMe metadynamics (Sec. III.G, Fig. 5, App. C)

Well-tempered metadynamics of capped alanine (Ace-Ala-NMe) on φ and ψ with UMA-S-1.2 + `model_moe`, in the gas phase
(UMA-S-1.2 alone), chloroform (`chcl3`), methanol (`meoh`), and water. Each phase has 8 walkers sharing one bias, 250 ps
per walker, a 1 fs time step, and a Langevin thermostat at 298.15 K (friction 1 ps⁻¹). The section and App. C also
quote three checks of these simulations: an energy-conservation test, AnySolv and GFN2-xTB/ALPB against DFT-SMD at 150
sampled frames, and the cost per step. `reproduce_checks.py` recomputes them from the files listed after it. The
overlap with the training set quoted in Sec. III.G is not included.

| file | content |
|---|---|
| `run_md.py` | runs the simulations. Default: one metadynamics walker through ASE + PLUMED (8 processes sharing one directory reproduce the paper's setup). `--no-bias`: plain Langevin MD. See its docstring |
| `seeds/<phase>_<basin>.xyz` | the 20 start structures, one per basin (C7eq, C5, PPII, aR, aL) and phase, relaxed on that phase's potential with φ and ψ held. Walker i starts from basin [C7eq, C5, PPII, aR, aL][i mod 5]. The atom order is the one `run_md.py` uses for the PLUMED atom numbers |
| `metad/<phase>/COLVAR.<walker>` | the output of the published runs, every 50 steps: time (fs), φ, ψ (rad), `metad.bias`, `metad.rbias` (kJ/mol), and d = acetyl C=O ⋯ methylamide H–N distance (Å). The water files repeat their header once after a restart; skip lines starting with `#` |
| `reproduce_alanine.py` | recomputes from the COLVAR files the Fig. 5 surfaces (`--fes-out DIR`), basin populations, ³J(Hᴺ,Hα), and convergence. With `--chen`, it also recomputes the comparison with Chen et al. |
| `reproduce_checks.py` | recomputes the numbers of the three checks and prints each next to the value in the paper |
| `nve/<model>_dt<dt>.csv` | energy-conservation test in water from the C7eq seed: 10 ps of Langevin dynamics at 298.15 K, then 50 ps of velocity Verlet. The files hold the microcanonical part every 10 steps: time (ps from its start), potential, kinetic, and total energy (eV). `moe` = UMA-S-1.2 + `model_moe`, `uma` = UMA-S-1.2 alone, both in float32; time step 0.5 or 1 fs |
| `timing_md.csv` | wall time per step (`ms_per_step`) of 1,000 Langevin steps of UMA-S-1.2 + `model_moe` in water, with `n_procs` processes sharing one A100 MIG 3g.20gb slice (`proc` = process; `rc` ≠ 0, `ms_per_step` = nan: the process ran out of memory). The paper quotes `n_procs` = 1 |
| `dftsmd_check/check_vs_smd.csv` | the DFT-SMD check, one row per frame; see below |
| `dftsmd_check/frames_<phase>.xyz` | the 150 frames of the check (extended xyz, atom order of `seeds/`) with the DFT-SMD solvation force `dF_dft` = −(∇E_SMD − ∇E_gas) per atom (eV/Å, from the ORCA gradients with 1 Eh = 27.211386245988 eV and 1 bohr = 0.529177210903 Å). Comment line: `fid`, `phase`, `walker`, `step`, `phi`, `psi` (degrees), and `basin` |

Requirements:
- `reproduce_alanine.py` and `reproduce_checks.py` need only numpy and pandas.
- `run_md.py` needs `anysolv` with the `model_moe` weights, `fairchem-core` (UMA-S-1.2), and `ase`.
- Metadynamics additionally needs PLUMED ≥ 2.9 with its Python module (`conda install -c conda-forge plumed py-plumed`).

`--chen` takes `ISSNet_ala2_example/dataset/ala2_dataset.npz` from
`ISSNet_ala2_example.zip` at http://ftp.imp.fu-berlin.de/pub/cmb-data/ (715 MB).

## DFT-SMD check (`dftsmd_check/`)

From each of the chloroform, methanol, and water runs, 50 frames were drawn from the structures saved every 500 steps
after the first 20 % of each walker: 8 at random from each of the five basins and 10 at basin edges or barriers
(`selection` = `edge`). Each frame was recomputed with ORCA 6.0.1 at the level of the training labels (App. B.1),
`! wB97M-V def2-TZVPD TightSCF DefGrid3 RKS EnGrad` without and with `SMD(chloroform)`, `SMD(methanol)`, or
`SMD(water)`, charge 0, singlet, `%pal nprocs 16`, `%maxcore 5000`. The ORCA input and output files are not included.

| column | meaning |
|---|---|
| `fid`, `phase` | frame id and phase |
| `walker`, `step` | walker and MD step of the frame in `metad/<phase>/`. φ, ψ, and d of the frame equal those of the COLVAR row at that step, except for the 13 water frames after the restart (step > 192,000), which differ from it by up to 1° and 0.03 Å |
| `selection`, `basin` | how the frame was drawn (a basin, or `edge`) and the basin it lies in (App. C; `b/C5` = C5, `aL/C7ax` = φ > 0) |
| `phi`, `psi` | backbone dihedrals (degrees) |
| `E_gas_dft_eV`, `E_smd_dft_eV` | ωB97M-V/def2-TZVPD energy without and with SMD (eV) |
| `dE_dft_eV`, `dE_moe_eV`, `dE_alpb_eV` | solvation energy from DFT-SMD (E_SMD − E_gas), from `model_moe` (the AnySolv correction alone), and from GFN2-xTB/ALPB (xtb 6.7.1, solvent `alpb_solvent`), eV |
| `E_gas_uma_eV` | UMA-S-1.2 energy (not used in the paper) |
| `dF_cos` | cosine similarity between the `model_moe` and DFT-SMD solvation forces of the frame; the paper quotes the mean per solvent |
| `dF_mae_eVA`, `dF_mag_mae_eVA` | MAE of the `model_moe` solvation forces and of their per-atom magnitudes against DFT-SMD (eV/Å; not quoted in the paper) |
| `dF_dft_rms_eVA`, `dF_moe_rms_eVA` | rms over the 3N components of the DFT-SMD and `model_moe` solvation forces (eV/Å) |
| `orca_wall_s_gas`, `orca_wall_s_smd` | ORCA wall time (s) of the gas-phase and SMD calculations. The paper's 182 s is the mean of `orca_wall_s_smd` |

For the centered comparison, the mean over the 50 frames of a solvent is subtracted from the model and from the DFT-SMD
solvation energies before R² and the errors are computed (see `reproduce_checks.py`). The `model_moe` forces at the
frames were not stored; the force columns are the per-frame results of the original evaluation, and `dF_dft` in the xyz
files allows the comparison to be repeated with the model.


## AI Use

Portions of the code in this repository were generated with the assistance of AI-based coding tools. All numerical values reported in the associated manuscript have been independently checked and verified by the authors.