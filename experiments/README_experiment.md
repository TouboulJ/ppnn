# Dimension-Scaling Experiment (companion to Section 6 of the paper)

This directory contains everything you need to reproduce the controlled
dimension-scaling experiment that validates the central theoretical claim
of the paper: that distributional neighborhoods beat metric neighborhoods
in high dimension.

## Files

- `run_dim_scaling.py` : main experiment script (checkpointed, resumable)
- `fast_gmm.py` : fast GMM-based substitutes for PP-NN on Gaussian-mixture data
- `dim_scaling.csv` : results from Claude's sandbox run (570 / 600 cells)
- `dim_scaling_eps000.png` : Figure `fig_dim_scaling_clean` (ε=0)
- `dim_scaling_eps010.png` : ε=0.10 supplementary figure
- `dim_scaling_eps020.png` : Figure `fig_dim_scaling_eps20` (ε=0.20)

## How to re-run locally

```bash
# install deps (only sklearn + pandas + matplotlib needed)
pip install numpy scipy scikit-learn pandas matplotlib

# fast validation run (~3 min)
python run_dim_scaling.py --fast

# full run (~15-30 min on a laptop; resumable via checkpoint)
python run_dim_scaling.py
```

The script writes to `results_dim/dim_scaling.csv` and regenerates the figures.

## Rationale for the design

The experiment uses a 3-class Gaussian mixture in R^d where only 2 directions
carry discriminative signal (the remaining d-2 dimensions are pure noise). A
random orthogonal rotation makes the informative subspace unknown a priori.
This isolates the dimension-scaling question from any optimiser quality
question: on Gaussian-mixture data, GMM fitting is the oracle upper bound
of any projection-pursuit method that approximates it. We use GMM as a
stand-in for "perfect-optimiser PP-NN" to cleanly separate the concept from
the implementation.

## Key result (from dim_scaling.csv)

At ε=0 contamination, classification error vs dimension:

| d   | kNN   | RF    | GMMkNN_full | GMMkNN_PCA |
|-----|------:|------:|------------:|-----------:|
| 5   | 0.009 | 0.013 | 0.009       | 0.021      |
| 10  | 0.009 | 0.015 | 0.008       | 0.013      |
| 20  | 0.011 | 0.025 | 0.014       | 0.017      |
| 30  | 0.013 | 0.025 | 0.020       | 0.012      |
| 50  | 0.015 | 0.034 | 0.031       | 0.011      |
| 80  | 0.020 | 0.038 | 0.074       | 0.011      |
| 120 | 0.026 | 0.047 |  —          | 0.011      |
| 200 | 0.038 | 0.073 |  —          | 0.010      |

GMMkNN_full skipped for d > 80 (full covariance infeasible at that scale).

Three patterns are clear:
- kNN degrades monotonically (4× from d=5 to d=200)
- GMMkNN_full (naive full-cov mixture) fails above d ≈ 50
- GMMkNN_PCA (with dim reduction) stays flat at ≈ 1% error for all d

The crossover where GMMkNN_PCA beats kNN is at d ≈ 30.

## Citation

This experiment is described in Section 6 ("Controlled Validation") of the
paper. The key figure is `fig_dim_scaling_clean` (= dim_scaling_eps000.png).

Author : Touboul Jacques