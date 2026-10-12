# snar — S_NAr mechanism screen (Sec. III.F, Fig. 4, App. C)

This folder covers the 120 anionic S_NAr reactions of Kwan et al. (*Nat. Chem.* **2018**, *10*, 917), with 8 rings × 3 leaving
groups X × 5 nucleophiles Y (charge −1, singlet). We located their stationary points with UMA-S-1.2 + `model_moe` in the gas
phase, THF, acetonitrile, DMSO, methanol, and water. Gas-phase rows are UMA-S-1.2 alone, although their `model` column
says `model_moe`.

`python reproduce_snar.py` (numpy, pandas) recomputes every S_NAr number in the paper: the cost figures, the Fig. 4
counts, the DMSO comparison with Kwan et al., the leaving-group and ring trends, and the per-phase counts and well-depth
shifts. It also checks every file in `geoms/` against `survey.csv`.

Identifiers: reaction = `<ring>_<X>_<Y>`. `OCHO` is formate, `N3` azide, `NMe2` dimethylamide, and `OMe` methoxide.
Phases are `gas`, `thf`, `acetonitrile`, `dmso`, `methanol`, and `water`. Stationary points (legs) are `rc` (reactant
complex), `ts_add` (addition TS), `int` (σ-adduct, the Meisenheimer complex), `ts_elim` (elimination TS), `ts_conc`
(concerted TS), and `pc` (product complex). r(C–X) and r(C–Y) are the distances from the ipso carbon to X and to the
attacking atom of Y.

## Files

| file | content |
|---|---|
| `reactions.csv` | 120 reactions: RDKit-validated SMILES of substrate, nucleophile, product, leaving group, and adduct; `natoms`, `charge`, `spin`; `identity` (X = Y = F); `kwan_mechanism` = Kwan et al.'s B3LYP-D3(BJ)/6-31+G*/IEFPCM(DMSO) class; `kwan_seeds` = which of their structures (from their SI archive, not redistributed here) were used as seeds |
| `survey.csv` | one row per reaction × phase (720): class, well depth, and every accepted stationary point |
| `legs.csv` | every optimization attempt of the screen (4,827 rows), accepted or rejected, including the cross-phase seeding follow-up |
| `geoms/model_moe/<reaction>_<phase>_<leg>.xyz` | the 2,391 accepted stationary points listed in `survey.csv` `legs`. Comment line: `E_elec_eV`, `E_gasUMA_eV`, `dsolv_anysolv_eV`, `G_eV`, imaginary frequencies (cm⁻¹), and 0-based `ipso_index`, `X_index`, `Y_index`. Atom order is Kwan et al.'s and is the same for all phases |

## Classification (`survey.csv`)

The well depth is D = E(lower flanking TS) − E(int), from electronic energies (UMA-S-1.2 + AnySolv). A flanking side
without an accepted TS uses the maximum of a relaxed exit scan along the breaking bond (0.1 Å steps out to 3.5 Å) as an
upper bound.

| column | meaning |
|---|---|
| `cls` | `stepwise` (D > 0.5 kcal/mol, or the adduct is a sink on both sides); `borderline` (0 < D ≤ 0.5); `concerted` (no adduct minimum, connected `ts_conc`); `concerted_flat` (the adduct is a minimum but has no well, D ≤ 0); `concerted_barrierless` (no adduct and no TS; the relaxed 2-D scan path from reactant to product edge rises < 0.5 kcal/mol); `unresolved` (no acceptable TS found; 23 of the 24 are gas phase). The paper's "intermediate" = `stepwise` or `borderline` |
| `mechanism` | `cls` with the three concerted classes merged (the "concerted" of Fig. 4) |
| `D_elec_kcal` | D (kcal/mol); empty without an adduct, and for the two gas-phase rows whose adduct lies below both exit scans (`4-nitrophenyl_F_F`, `4-acetylphenyl_F_F`; drawn in the top bin of Fig. 4) |
| `D_zpe_kcal`, `D_G_kcal` | the same with ZPE-corrected energies and with qRRHO free energies (298.15 K, 50 cm⁻¹ floor); not used for classification |
| `D_source` | `ts` (D from a located TS), `scan_bound` (D is an exit-scan upper bound), `sink` |
| `side_add`, `side_elim` | per flanking side: `ts` (accepted, IRC-connected TS), `sink` (the exit scan never drops below the adduct), `bound` (it drops below, with a maximum < 0.5 kcal/mol), `unlocated` (maximum ≥ 0.5 kcal/mol but no acceptable saddle; the maximum still bounds D), `mirror` (identity reactions: elimination = mirror of addition), `none` (no adduct) |
| `ts_elim_mirror` | identity reaction whose `ts_elim` is the mirror image of its `ts_add` |
| `unresolved_reason` | why a row is `unresolved`, `concerted_flat`, or `concerted_barrierless` |
| `dual_path`, `dE_conc_minus_flank_kcal` | an adduct with a flanking TS and a connected `ts_conc` coexist; E(ts_conc) − E(highest flanking TS), kcal/mol |
| `kwan_mechanism` | copied from `reactions.csv` |
| `legs` | accepted stationary points, `+`-joined; one xyz file each |
| `<leg>_rCX`, `<leg>_rCY` | r(C–X), r(C–Y) of that stationary point, Å |
| `<leg>_E_eV`, `<leg>_G_eV` | electronic energy and qRRHO free energy (298.15 K, 50 cm⁻¹ floor), eV |
| `n_attempts`, `wall_s` | attempts and summed attempt wall time (s) of the unit |

Transition states were accepted with exactly one imaginary mode above 100 cm⁻¹, a mode projecting at least 0.5 onto
the C–X/C–Y stretches, both distances in 1.2–3.5 Å, and no proton transfer, then connected by IRC (App. C). An adduct has
both distances below 1.25 × the sum of covalent radii. Minima are converged to 0.01 eV/Å.

## Attempt log (`legs.csv`)

- `leg` covers the six stationary-point legs plus the scans that seeded or bounded them. `scan1d_ts_add` and
  `scan1d_ts_elim` are exit scans from the adduct, `scan1d_ts_conc` is an approach scan, and `scan2d` is an 18 × 18
  relaxed grid over r(C–X), r(C–Y) = 1.3–3.0 Å. Two leg types are sensitivity checks that no class uses: `B1_scan`
  (gas-phase capture scans of unresolved rows) and `B2_irc` (re-examined soft saddles).
- `seed_tag` gives the starting structure. It is one of `prev` (the same leg in the preceding phase), `kwan_*` (Kwan et
  al.'s geometry), `rdkit_adduct`, `scan1d_*`, `scan2d_*`, `endpoint_of_*` (an IRC endpoint), or `xphase_<phase>` /
  `x:` (cross-phase seeds of the follow-up; `followup = True`).
- Attempt metadata: `accepted`, `reject_reason`, `notes` (per Sella strategy), `strategy`, `rounds`, `sella_steps`, and
  `sella_converged`.
- Geometry: `r_CX` and `r_CY` in Å, and 0-based `ipso_index`, `X_index`, `Y_index`. `swap_xy = True` marks
  identity-reaction attempts in which the two fluorines exchanged roles.
- Energies: `E_elec_eV = E_gasUMA_eV + dsolv_anysolv_eV` on every accepted row; the 78 rejected gas-phase TS attempts where
  it does not hold should use `E_elec_eV`. `G_eV` and `zpe_eV` are the qRRHO free energy and zero-point energy.
- Frequencies (finite-difference Hessian): `freqs_cm`, `imag_freqs_cm`, `n_imag_small`, `n_imag_big`, and `n_floored`.
- TS mode and IRC: `mode_bond_character`, `mode_char_X`, `mode_char_Y`, `irc_method`, `irc_endpoints`, `connected_as`, and
  `endpoint_class` (`*_loose` = the nucleophile or leaving group sits within 2.5 Å but is not bonded; counted as rc/pc).
- `wall_s` is the attempt wall time on one A100 MIG 3g.20gb slice. Its sum over all rows is the 68 GPU-hours
  of the paper. That total includes the follow-up and excludes queue overhead and the separated-species relaxations.

## Not included

- Geometries of rejected attempts, IRC paths, the 2-D scan grids, and the separated reactants and products.
- The 179-solvent single points at the DMSO stationary points, which the paper does not use.
- Kwan et al.'s structures: consult their SI.
- The driver code.

## AI Use

Portions of the code in this repository were generated with the assistance of AI-based coding tools. All numerical values reported in the associated manuscript have been independently checked and verified by the authors.