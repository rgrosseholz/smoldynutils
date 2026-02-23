# smoldynutils
Utilities for parsing, analyzing, and visualizing Smoldyn simulation outputs.

## Installation
From GitHub with poetry:
```bash
git clone https://github.com/rgrosseholz/smoldynutils.git
cd smoldynutils
poetry install
poetry shell
```

Or via pip:
```bash
pip install smoldynutils
```

## Quickstart
```python
from smoldynutils.parser import SmoldynParser

parser = SmoldynParser(delimiter=",")
trajectories = parser.parse_fixed_grid("molpos_output.txt")

for traj in trajectories:
    print(traj.positions)
```

## Authors
Fabian Ormersbach, Maastricht Centre for Systems Biology and Bioinformatics, Maastricht University  
Ruth Grosseholz, Maastricht Centre for Systems Biology and Bioinformatics, Maastricht University

## License
This project is licensed under the MIT License. See `LICENSE.md` for details.
