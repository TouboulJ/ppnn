# Projection-Pursuit Nearest Neighbors (PP-NN)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**A scikit-learn-compatible Python package for supervised local estimation
using $\varphi$-divergence projection pursuit, replacing the metric neighborhood
of $k$-NN by a distributional one.**

This repository accompanies the paper:

> Jacques Touboul, *Projection-Pursuit Nearest Neighbors: Distributional Neighborhoods
> for High-Dimensional Supervised Learning*, 2026.

The companion **technical report** (full proofs, extended simulations, robustness
analysis) is archived on Zenodo: https://doi.org/10.5281/zenodo.19821616.

## Installation

```bash
git clone https://github.com/TouboulJ/ppnn.git
cd ppnn
pip install -e .
```

Dependencies: `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`.

## Quick start

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from ppnn import PPNNHard, PPkNN, PPNNDispatch

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)

model = PPkNN(K=3, k=15, divergence='Hellinger', task='classification')
model.fit(X_tr, y_tr)
print(f"Accuracy: {model.score(X_te, y_te):.3f}")
```

The package exposes three estimators, all with sklearn-compatible
`fit`/`predict`/`score` methods:

- **`PPNNHard`** — hard assignment to the dominant component, then majority/mean
- **`PPkNN`** — hybrid $k$-NN within the assigned component
- **`PPNNDispatch`** — p-value-gated selection between $k$-NN, PP-NN, and PP-$k$NN

## Reproducing the paper's results

### Controlled dimension-scaling experiment (Section 6, ~4 min)

```bash
cd experiments
python run_dim_scaling.py            # full run, ~4 min on a laptop
python run_dim_scaling.py --fast     # quick validation run, ~1 min
```

Produces:
- `results_dim/dim_scaling.csv` — 600 cells (8 dims × 3 contamination levels × 5 methods × 5 reps)
- `results_dim/dim_scaling_eps{000,010,020}.png` — Figures 1 of the paper
- `results_dim/table_dim.tex` — LaTeX table

### Real-data benchmark (Section 7, ~90 min)

```bash
cd experiments
python run_all.py
```

Reproduces the 8-method × 5-dataset table. Checkpointed and resumable.

## Repository layout

```
ppnn/                  Main Python package
  __init__.py          Public API: PPNNHard, PPkNN, PPNNDispatch, TouboulPP
  core.py              TouboulPP backbone: phi-divergence projection pursuit
  estimators.py        Three estimators (Hard, kNN, Dispatch) + sklearn glue

experiments/           Reproduction scripts
  run_dim_scaling.py   Section 6 controlled experiment
  run_all.py           Section 7 real-data benchmark
  fast_gmm.py          GMM-based oracle substitutes (used by run_dim_scaling)
  README_experiment.md Detailed experiment notes

docs/                  Documentation
  api.md               Public API reference
  paper.pdf            Latest version of the paper
```

## Citation

If you use this code, please cite:

```bibtex
@article{ppnn2026,
  title   = {Projection-Pursuit Nearest Neighbors: Distributional
             Neighborhoods for High-Dimensional Supervised Learning},
  author  = {Touboul Jacques},
  journal = {Statistics and Computing},
  year    = {2026},
  doi     = {[to be inserted]}
}

@misc{ppnn-techreport-2026,
  title  = {Projection-Pursuit Nearest Neighbors: companion technical report},
  author = {Touboul Jacques},
  year   = {2026},
  doi    = {https://doi.org/10.5281/zenodo.19821616}
}
```

The underlying $\varphi$-divergence projection-pursuit framework is due to
Touboul (2010, 2025):

```bibtex
@article{touboul2010,
  author  = {Touboul, J.},
  title   = {Projection Pursuit Through phi-Divergence Minimisation},
  journal = {Entropy},
  volume  = {12},
  number  = {6},
  pages   = {1581--1611},
  year    = {2010},
  doi     = {10.3390/e12061581}
}

@article{touboul2025,
  author  = {Touboul, J.},
  title   = {Robust portfolio construction and high frequency trading
             through projection pursuit},
  journal = {Communications in Statistics -- Simulation and Computation},
  year    = {2025},
  doi     = {10.1080/03610918.2025.2524769}
}
```

## License

MIT License — see [LICENSE](LICENSE).

## Issues and contributions

Bug reports and pull requests are welcome at
<https://github.com/TouboulJ/ppnn/issues>.

The current implementation has known limitations documented in Section 8 of
the paper (five "failure modes"). Contributions addressing these — especially
a Riemannian-Newton optimiser to replace Nelder-Mead — would be particularly
valuable.
