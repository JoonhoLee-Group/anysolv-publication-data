# anysolv-publication-data

Data behind the numbers in *AnySolv: An equivariant message-passing graph neural network for transferable implicit solvation* (A. Z. Ni and J. Lee). 

## Related resources

| resource | location |
|---|---|
| `anysolv` package (code, compact model weights) | https://github.com/Ant-on-knee/anysolv |
| FAIRChem fork with the solvent embedding | https://github.com/Ant-on-knee/fairchem-solvation |
| MoE model weights; DFT-SMD training, validation, and test labels (`data/{train,val,test}.tar.gz`) | https://huggingface.co/antonknee/anysolv |

## Requirements

Every `reproduce_*.py` needs only Python 3 with numpy and pandas, reads its files relative to its own location, and can be run from any directory, e.g. `python snar/reproduce_snar.py`. `alanine_metadynamics/run_md.py` additionally needs `anysolv`, `fairchem-core`, `ase`, and PLUMED (see that folder's README).

## License

The scripts (`*.py`) are licensed under the [MIT License](LICENSE-CODE); all other files are licensed under [CC BY 4.0](LICENSE).

## AI Use

Portions of the code in this repository were generated with the assistance of AI-based coding tools. All numerical values reported in the associated manuscript have been independently checked and verified by the authors.
