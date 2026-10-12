#!/usr/bin/env python
"""Molecular dynamics of Ace-Ala-NMe with UMA-S-1.2 + AnySolv, as run for Sec. III.G and App. C.

By default this runs one walker of the paper's well-tempered metadynamics (PLUMED through ASE); --no-bias runs
plain Langevin dynamics. In the gas phase, UMA-S-1.2 is used alone. plumed_input() reproduces the PLUMED input of
the published runs, which used 8 walkers per phase sharing one bias, 250 ps each.

Examples
    python run_md.py --solvent water --ps 1 --device cuda
    for i in 0 1 2 3 4 5 6 7; do
        python run_md.py --solvent methanol --walker $i --n-walkers 8 --ps 250 --device cuda &
    done; wait
    python run_md.py --solvent chloroform --no-bias --ps 10 --device cuda

Outputs in --out (default md_<solvent>/): COLVAR.<walker> and HILLS.<walker> (metadynamics only),
traj.<walker>.xyz (every 500 steps), and md.<walker>.log.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.mixing import SumCalculator
from ase.constraints import FixCom
from ase.io import read, write
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from anysolv import predict_solvation_energy

HERE = Path(__file__).resolve().parent
T_K = 298.15
BASINS = ["C7eq", "C5", "PPII", "aR", "aL"]              # seed order of the walkers
SEED_TAG = {"gas": "gas", "chloroform": "chcl3", "methanol": "meoh", "water": "water"}


class AnySolvDelta(Calculator):
    """The AnySolv solvation correction (dE, dF) as an ASE calculator; add it to a gas-phase calculator."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, solvent, charge=0, spin=1, checkpoint=None, device="cpu", inference_settings="default"):
        super().__init__()
        self.kw = dict(charge=charge, spin=spin, solvent=solvent, checkpoint=checkpoint, device=device,
                       inference_settings=inference_settings)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        dE, dF = predict_solvation_energy(self.atoms, **self.kw)   # the model is loaded once and cached
        self.results = {"energy": dE, "forces": dF}


def plumed_input(out: Path, walker: int, n_walkers: int) -> list[str]:
    """The METAD input of the paper (atom numbers are 1-based, in the atom order of seeds/*.xyz)."""
    walkers = (f" WALKERS_N={n_walkers} WALKERS_ID={walker} WALKERS_DIR={out} WALKERS_RSTRIDE=500"
               if n_walkers > 1 else "")
    hills = "HILLS" if n_walkers > 1 else f"{out}/HILLS.{walker}"   # with walkers: <WALKERS_DIR>/HILLS.<id>
    return ["UNITS LENGTH=A ENERGY=kj/mol TIME=fs",
            "phi: TORSION ATOMS=2,4,5,7",                      # C(acetyl)-N-CA-C
            "psi: TORSION ATOMS=4,5,7,9",                      # N-CA-C-N(methylamide)
            "d: DISTANCE ATOMS=3,19",                          # acetyl O ... methylamide H-N
            f"metad: METAD ARG=phi,psi SIGMA=0.35,0.35 HEIGHT=1.2 PACE=500 BIASFACTOR=6 TEMP={T_K} "
            f"GRID_MIN=-pi,-pi GRID_MAX=pi,pi GRID_BIN=180,180 CALC_RCT RCT_USTRIDE=1 FILE={hills}{walkers}",
            f"PRINT STRIDE=50 ARG=phi,psi,metad.bias,metad.rbias,d FILE={out}/COLVAR.{walker} FMT=%12.6f",
            "FLUSH STRIDE=500"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solvent", default="water", help="'gas' or any AnySolv solvent name (default: water)")
    ap.add_argument("--walker", type=int, default=0, help="walker id (default 0)")
    ap.add_argument("--n-walkers", type=int, default=1, help="walkers sharing the bias (paper: 8)")
    ap.add_argument("--seed-xyz", type=Path, default=None,
                    help="start structure (default: seeds/<phase>_<basin>.xyz for this walker; water seeds for "
                         "solvents without their own)")
    ap.add_argument("--ps", type=float, default=1.0, help="simulation length per walker in ps (paper: 250)")
    ap.add_argument("--dt", type=float, default=1.0, help="time step in fs (paper: 1)")
    ap.add_argument("--no-bias", action="store_true", help="plain Langevin MD without PLUMED")
    ap.add_argument("--checkpoint", default=None, help="AnySolv checkpoint (default: model_moe if present)")
    ap.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    ap.add_argument("--inference", default=None,
                    help="AnySolv inference_settings (default: fast_gpu on CUDA, as in the paper, else default)")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default md_<solvent>/)")
    ap.add_argument("--rng-seed", type=int, default=None, help="seed of the initial velocities (default: walker id)")
    a = ap.parse_args()

    gas = a.solvent.lower() in ("gas", "none", "vacuum")
    out = (a.out or Path(f"md_{'gas' if gas else a.solvent.replace(' ', '_')}")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    seed = a.seed_xyz or HERE / "seeds" / f"{SEED_TAG.get('gas' if gas else a.solvent, 'water')}_{BASINS[a.walker % 5]}.xyz"
    atoms = read(seed)
    atoms.info.update(charge=0, spin=1)
    atoms.calc = None

    # gas-phase potential: UMA-S-1.2 (task omol); solvated = UMA-S-1.2 + AnySolv
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    base = FAIRChemCalculator(pretrained_mlip.get_predict_unit("uma-s-1p2", device=a.device), task_name="omol")
    if gas:
        calc = base
    else:
        inference = a.inference or ("fast_gpu" if a.device.startswith("cuda") else "default")
        calc = SumCalculator([base, AnySolvDelta(a.solvent, checkpoint=a.checkpoint, device=a.device,
                                                 inference_settings=inference)])

    rng = np.random.default_rng(a.walker if a.rng_seed is None else a.rng_seed)
    atoms.set_constraint(FixCom())
    MaxwellBoltzmannDistribution(atoms, temperature_K=T_K, rng=rng)
    Stationary(atoms)
    nsteps = int(round(a.ps * 1000 / a.dt))
    print(f"{seed.name} -> {out}: {'gas phase (UMA-S-1.2)' if gas else a.solvent}, walker {a.walker}/{a.n_walkers}, "
          f"{nsteps} steps of {a.dt} fs, {'no bias' if a.no_bias else 'WT-MetaD on phi/psi'}")

    def run(calculator):
        atoms.calc = calculator
        dyn = Langevin(atoms, timestep=a.dt * units.fs, temperature_K=T_K, friction=1.0 / (1000 * units.fs),
                       fixcm=False, rng=rng)
        dyn.attach(MDLogger(dyn, atoms, str(out / f"md.{a.walker}.log"), header=True, stress=False, peratom=False),
                   interval=500)
        traj = out / f"traj.{a.walker}.xyz"
        traj.unlink(missing_ok=True)
        dyn.attach(lambda: write(traj, atoms, format="extxyz", append=True), interval=500)
        dyn.run(nsteps)

    if a.no_bias:
        run(calc)
    else:
        from ase.calculators.plumed import Plumed
        with Plumed(calc=calc, input=plumed_input(out, a.walker, a.n_walkers), timestep=a.dt * units.fs,
                    atoms=atoms, kT=units.kB * T_K, log=str(out / f"plumed.{a.walker}.log")) as biased:
            run(biased)
    print("done")


if __name__ == "__main__":
    main()
