# Third-Party Notices

The repository audit found no copied or adapted third-party source code in the
tracked implementation. The project uses the following external runtime and
development dependencies without vendoring their source:

| Component | Relationship |
|---|---|
| NumPy, pandas, SciPy, statsmodels | Runtime numerical/statistical dependencies |
| PyYAML, tqdm | Runtime configuration/progress dependencies |
| PyTorch, torchvision | Runtime training and vision dependencies |
| tonic, aedat | Optional event-data dependencies |
| pytest, Ruff | Development and validation dependencies |

Each dependency remains governed by its own license and distribution terms.
The release does not relicense or redistribute those dependency source trees.

The CIFAR datasets and CIFAR10-DVS files are not redistributed by this project.
The data acquisition instructions must preserve the original dataset terms.
