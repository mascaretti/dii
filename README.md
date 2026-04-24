# dii

Standalone JAX implementation of the **Differentiable Information Imbalance (DII)**, extracted from [DADApy](https://github.com/sissa-data-science/DADApy) so it can be installed without DADApy's Cython / OpenMP / scipy / scikit-learn / seaborn / networkx build chain.

This package exposes exactly the same `DiffImbalance` class that DADApy exports via `from dadapy import DiffImbalance` — the one used in the upstream tutorial notebook: https://dadapy.readthedocs.io/en/latest/jupyter_example_7.html

## Install

```bash
pip install -e /path/to/dii
```

## Usage

```python
from dii import DiffImbalance

dii = DiffImbalance(data_A=X, data_B=Y, num_epochs=200)
weights, imb = dii.train()
```

See the upstream [notebook_on_differentiable_imbalance_jax.ipynb](https://github.com/sissa-data-science/DADApy/blob/main/examples/notebook_on_differentiable_imbalance_jax.ipynb) for a worked example.

## Attribution

All `DiffImbalance` code is authored by the DADApy authors and licensed under Apache-2.0 (see `LICENSE`). This package is a verbatim extraction of `dadapy/diff_imbalance.py` with a minimal packaging shim; no algorithmic changes.

Upstream: https://github.com/sissa-data-science/DADApy
