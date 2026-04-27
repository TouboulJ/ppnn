"""
Controlled experiment: dimension scaling of k-NN vs distributional-neighborhood methods

Design:
  - K=3 Gaussian components in R^d, means aligned on 2 directions
  - Only 2 "informative" directions carry signal; remaining (d-2) dims are noise
  - Class label = component index
  - Vary d from 5 to 200 (log-spaced)
  - Optional: contamination at eps in {0, 0.1, 0.2}
  - Metric: classification error on clean test set

Estimators compared:
  - kNN  : classical k-Nearest Neighbors (metric neighborhood)
  - RF   : Random Forest (partition-based)
  - GMMHard : hard-assign to GMM component then majority label -
             fast substitute for PP-NN Hard. Since the true DGP is a Gaussian
             mixture, GMM fitting is strictly optimal for the gating step.
             This gives a fair upper bound on what any distributional-
             neighborhood method (including PP-NN with perfect optimiser)
             could achieve.
  - GMMkNN  : hard-assign then k-NN within the component -
              fast substitute for PP-kNN.

Rationale for the GMM substitution: the experiment's question is whether the
*concept* of a distributional neighborhood outperforms the metric neighborhood
as d grows, not whether any particular PP optimiser reaches that bound. A GMM
fit is the oracle upper bound of PP-NN on Gaussian-mixture data and lets us
isolate the dimension-scaling question from the optimiser-quality question
(Failure mode 3 in our paper).

Hypothesis:
  - kNN degrades as d grows (concentration of distances)
  - GMM-based methods robust to d because they exploit the 2 informative axes
  - Crossover where GMMkNN beats kNN expected around d = 20-50

Usage:
    python run_dim_scaling.py [--fast]

Runtime estimate (full): ~15-30 min on a laptop
Runtime estimate (--fast): ~3 min
Author : Touboul Jacques
"""

import sys, os, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# make local modules importable
sys.path.insert(0, str(Path(__file__).parent))
from fast_gmm import GMMHard, GMMkNN

FAST = '--fast' in sys.argv
OUTDIR = Path(__file__).parent / 'results_dim'
OUTDIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------------
if FAST:
    DIMS      = [5, 10, 20, 50]
    N_TRAIN   = 500
    N_TEST    = 500
    N_REPS    = 3
    EPS_GRID  = [0.0, 0.20]
    METHODS   = ['kNN', 'RF', 'GMMkNN_full', 'GMMkNN_PCA']
else:
    DIMS      = [5, 10, 20, 30, 50, 80, 120, 200]
    N_TRAIN   = 1000
    N_TEST    = 1000
    N_REPS    = 5
    EPS_GRID  = [0.0, 0.10, 0.20]
    METHODS   = ['kNN', 'RF', 'GMMkNN_full', 'GMMkNN_PCA', 'GMMHard_PCA']

K_COMPONENTS = 3
K_NEIGHBORS  = 15
PCA_DIM      = 2    # the TRUE informative dimension; in practice we would select via CV

# -----------------------------------------------------------------------------
# Data generating process
# -----------------------------------------------------------------------------
def generate_data(d, n, K=3, sep=3.0, seed=None):
    """
    Generate K-component Gaussian mixture in R^d where informative variance
    is concentrated on the first 2 axes. Class label = component index.
    
    Params:
      d   : ambient dimension
      n   : total sample size
      K   : number of components
      sep : separation between component means on informative axes
      seed: RNG seed
    
    Returns: (X, y) with X shape (n, d), y shape (n,) in {0, ..., K-1}
    """
    rng = np.random.default_rng(seed)
    n_per = n // K
    
    # K means in R^2 on unit circle, scaled by sep
    angles = np.linspace(0, 2*np.pi, K, endpoint=False)
    means_2d = sep * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    
    # Embed into R^d: means live in the first 2 dims, remaining dims = 0
    means_d = np.zeros((K, d))
    means_d[:, :2] = means_2d
    
    # Random orthogonal rotation: mixes the 2 informative directions into R^d
    # This makes the problem realistic (informative directions unknown a priori)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    means_d = means_d @ Q.T  # rotate all means
    
    # Sample from each component
    X = np.zeros((K * n_per, d))
    y = np.zeros(K * n_per, dtype=int)
    # Covariance: isotropic in R^d but inflated in "noise" dims to reflect
    # the fact that non-informative dims carry variance too
    # Informative dims: std=1; noise dims: std=1 (same) - so informative signal
    # is *solely* in the means
    for k in range(K):
        Xk = rng.standard_normal((n_per, d)) + means_d[k]
        X[k*n_per:(k+1)*n_per] = Xk
        y[k*n_per:(k+1)*n_per] = k
    
    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx], Q  # Q kept for diagnostics

def contaminate(X, y, eps, seed):
    """Add eps*n/(1-eps) adversarial outliers to training set."""
    if eps == 0:
        return X, y
    rng = np.random.default_rng(seed)
    n, d = X.shape
    n_out = int(eps * n / (1 - eps))
    # Uniform outliers at 5 sigma
    X_out = rng.uniform(-5, 5, size=(n_out, d)) * X.std(0)
    # Random labels
    y_out = rng.integers(0, int(y.max())+1, size=n_out)
    return np.vstack([X, X_out]), np.concatenate([y, y_out])

# -----------------------------------------------------------------------------
# Method factory
# -----------------------------------------------------------------------------
def get_model(name):
    if name == 'kNN':
        return KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
    if name == 'RF':
        return RandomForestClassifier(n_estimators=100, random_state=0)
    if name == 'GMMHard_full':
        return GMMHard(K=K_COMPONENTS, covariance_type='full')
    if name == 'GMMHard_PCA':
        return GMMHard(K=K_COMPONENTS, covariance_type='full', pca_dim=PCA_DIM)
    if name == 'GMMkNN_full':
        return GMMkNN(K=K_COMPONENTS, k=K_NEIGHBORS, covariance_type='full')
    if name == 'GMMkNN_PCA':
        return GMMkNN(K=K_COMPONENTS, k=K_NEIGHBORS, covariance_type='full', pca_dim=PCA_DIM)
    raise ValueError(name)

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
def run():
    """Run the full sweep. Checkpointed: each completed (d, eps, method, rep)
    cell is written to a CSV; on restart, we skip completed cells."""
    csv_path = OUTDIR / 'dim_scaling.csv'
    
    # Load existing checkpoint
    if csv_path.exists():
        df_done = pd.read_csv(csv_path)
        done_keys = set(zip(df_done['d'], df_done['eps'], df_done['method'], df_done['rep']))
        rows = df_done.to_dict('records')
        print(f"[resume] {len(done_keys)} cells already done")
    else:
        done_keys = set()
        rows = []
    
    total = len(DIMS) * len(EPS_GRID) * len(METHODS) * N_REPS
    done = len(done_keys)
    print(f"Total cells: {total}, already done: {done}")
    t0 = time.time()
    
    for d in DIMS:
        for eps in EPS_GRID:
            for rep in range(N_REPS):
                seed = 1000 * d + 10 * int(eps*100) + rep
                # generate fresh data
                X, y, _ = generate_data(d=d, n=N_TRAIN+N_TEST, seed=seed)
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=N_TEST, stratify=y, random_state=seed)
                X_tr, y_tr = contaminate(X_tr, y_tr, eps, seed=seed+1)
                
                for method_name in METHODS:
                    # GMMkNN_full uses full covariance — statistically
                    # infeasible when d^2 > n_component ~ n/K. Skip when d > 80.
                    if method_name in ('GMMkNN_full', 'GMMHard_full') and d > 80:
                        continue
                    key = (d, eps, method_name, rep)
                    if key in done_keys:
                        continue
                    
                    model = get_model(method_name)
                    t_fit = time.time()
                    try:
                        model.fit(X_tr, y_tr)
                        y_pred = model.predict(X_te)
                        err = float(np.mean(y_pred != y_te))
                    except Exception as e:
                        print(f"  ERROR {method_name} d={d} eps={eps} rep={rep}: {e}")
                        err = float('nan')
                    dt = time.time() - t_fit
                    
                    rows.append({
                        'd': d, 'eps': eps, 'method': method_name,
                        'rep': rep, 'error': err, 'time_s': dt
                    })
                    done += 1
                    
                    if done % 5 == 0:
                        elapsed = time.time() - t0
                        eta = elapsed * (total - done) / max(done - len(done_keys), 1)
                        print(f"  [{done}/{total}] d={d} eps={eps} {method_name} rep={rep} "
                              f"err={err:.3f} ({dt:.1f}s)  ETA ~{eta/60:.0f} min")
                        # checkpoint
                        pd.DataFrame(rows).to_csv(csv_path, index=False)
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\nDone. Results written to {csv_path}")
    return df

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_results(df):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Aggregate by (d, eps, method)
    agg = df.groupby(['d', 'eps', 'method'])['error'].agg(['mean', 'std']).reset_index()
    
    # One figure per eps
    for eps in sorted(df['eps'].unique()):
        sub = agg[agg['eps']==eps]
        fig, ax = plt.subplots(figsize=(8,5))
        methods_present = sub['method'].unique()
        colors = {'kNN':'C0', 'RF':'C1',
                  'GMMHard_full':'C3', 'GMMHard_PCA':'C4',
                  'GMMkNN_full':'C2', 'GMMkNN_PCA':'C5'}
        for m in methods_present:
            row = sub[sub['method']==m].sort_values('d')
            ax.errorbar(row['d'], row['mean'], yerr=row['std'],
                        marker='o', capsize=3, label=m, color=colors.get(m))
        ax.set_xscale('log')
        ax.set_xlabel('Dimension $d$')
        ax.set_ylabel('Classification error')
        ax.set_title(f'Error vs dimension (3-class Gaussian mixture, contamination $\\varepsilon$={eps})')
        ax.legend()
        ax.grid(alpha=0.3)
        fname = OUTDIR / f'dim_scaling_eps{int(eps*100):03d}.png'
        plt.tight_layout()
        plt.savefig(fname, dpi=120)
        plt.close()
        print(f"Wrote {fname}")

def latex_table(df):
    """Produce a compact LaTeX table of mean error per (d, method) at eps=0."""
    agg = df[df['eps']==0].groupby(['d','method'])['error'].mean().reset_index()
    pivot = agg.pivot(index='d', columns='method', values='error')
    methods = ['kNN', 'RF', 'GMMkNN_full', 'GMMkNN_PCA', 'GMMHard_PCA']
    methods = [m for m in methods if m in pivot.columns]
    pivot = pivot[methods]
    latex = pivot.to_latex(float_format='%.3f', bold_rows=False)
    (OUTDIR / 'table_dim.tex').write_text(latex)
    print(f"Wrote {OUTDIR/'table_dim.tex'}")

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    df = run()
    plot_results(df)
    latex_table(df)
    print("\nSummary of mean error by method (all d pooled, eps=0):")
    sub = df[df['eps']==0]
    print(sub.groupby('method')['error'].agg(['mean','std']).round(3).to_string())
