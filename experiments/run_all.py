#!/usr/bin/env python3
r"""
run_all.py --- Single self-contained script for PP-NN empirical evaluation.

Does everything in one go:
  Phase 1: Full benchmark (k-NN, RF, GMM+QDA, MDA, MoE, PPNN_Hard, PPkNN,
           PPNNDispatch) on 10 datasets with 5-fold CV.
  Phase 2: Robustness study (k-NN, RF, PPkNN with KL / Hellinger / chi^2)
           on 4 datasets with contamination epsilon in {0, 5%, 10%, 20%},
           5 replications.
  Phase 3: Generate figures + LaTeX tables + LaTeX section ready to \input{}.

KEY FEATURES:
  - Automatic installation of the `ppnn` library from a tar.gz archive
    or from a ppnn_lib/ directory, with multiple fallbacks including
    sys.path injection if pip install fails (externally-managed envs).
  - Full checkpoint/resume: interrupt with Ctrl-C, rerun the same command,
    and it picks up where it left off. Rerunning a completed run is a no-op.
  - All output in one results directory, with one LaTeX section ready to
    include in your article.

USAGE (simplest case):
    Put ppnn-0.1.0.tar.gz (or the extracted ppnn_lib/ dir) next to this
    script and run:
        python run_all.py

    If the archive is somewhere else:
        python run_all.py --ppnn-path /path/to/ppnn-0.1.0.tar.gz
        python run_all.py --ppnn-path /path/to/ppnn_lib

    To skip slow parts (e.g. no internet):
        python run_all.py --skip-financial --skip-openml

    To just run the robustness study (skip Phase 1):
        python run_all.py --only-robustness

DEPENDENCIES:
    Required: numpy, pandas, scipy, scikit-learn, matplotlib
    Optional: yfinance (financial datasets), xgboost (extra baseline)
Author : Touboul Jacques
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
import warnings
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# BOOTSTRAP PPNN: ensure `import ppnn` works
# =============================================================================

def _try_import_ppnn():
    try:
        import ppnn  # noqa
        return getattr(ppnn, "__version__", "unknown")
    except ImportError:
        return None


def _find_ppnn_source(hint=None):
    """Locate ppnn_lib directory or ppnn*.tar.gz. Returns Path or None."""
    if hint:
        p = Path(hint).resolve()
        if p.exists():
            return p
    here = Path(__file__).resolve().parent
    cwd = Path.cwd()
    for root in [here, cwd]:
        if (root / "ppnn_lib").is_dir():
            return root / "ppnn_lib"
        hits = sorted(glob.glob(str(root / "ppnn*.tar.gz")))
        if hits:
            return Path(hits[0])
    return None


def ensure_ppnn(ppnn_path_hint=None, logger=print):
    """Make `import ppnn` work by any means necessary.

    Strategies, in order:
      1. Already importable => done.
      2. Locate ppnn_lib directory or ppnn*.tar.gz (hint or search).
      3. If archive, extract it.
      4. pip install -e ppnn_lib  (with --break-system-packages fallback).
      5. Inject sys.path so `import ppnn` works directly from source.
    """
    v = _try_import_ppnn()
    if v is not None:
        logger(f"[ppnn] Already importable (version {v})")
        return True

    src = _find_ppnn_source(ppnn_path_hint)
    if src is None:
        logger("[ppnn] ERROR: no ppnn_lib/ directory and no ppnn*.tar.gz found.")
        logger("[ppnn] Place the archive next to this script or use --ppnn-path.")
        return False

    logger(f"[ppnn] Found source: {src}")

    if src.is_file() and str(src).endswith(".tar.gz"):
        extract_dir = src.parent
        logger(f"[ppnn] Extracting {src.name} to {extract_dir}")
        try:
            with tarfile.open(src, "r:gz") as tf:
                tf.extractall(extract_dir)
        except Exception as e:
            logger(f"[ppnn] Extraction failed: {e}")
            return False
        src = extract_dir / "ppnn_lib"
        if not src.is_dir():
            logger(f"[ppnn] ERROR: expected ppnn_lib/ after extraction, not found")
            return False
        logger(f"[ppnn] Extracted to {src}")

    for extra in [[], ["--break-system-packages"]]:
        cmd = [sys.executable, "-m", "pip", "install", "-e", str(src),
               "--quiet", "--disable-pip-version-check", "--no-build-isolation"] + extra
        try:
            logger(f"[ppnn] pip install (args: {extra or 'none'})...")
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if r.returncode == 0:
                v = _try_import_ppnn()
                if v is not None:
                    logger(f"[ppnn] pip install OK (version {v})")
                    return True
                logger("[ppnn] pip install returned 0 but import still fails")
            else:
                tail = "\n".join(r.stderr.splitlines()[-3:])
                logger(f"[ppnn] pip install failed (exit {r.returncode}): {tail}")
        except Exception as e:
            logger(f"[ppnn] pip install exception: {e}")

    logger(f"[ppnn] Falling back to sys.path injection at {src}")
    sys.path.insert(0, str(src))
    for mod_name in list(sys.modules):
        if mod_name == "ppnn" or mod_name.startswith("ppnn."):
            del sys.modules[mod_name]
    v = _try_import_ppnn()
    if v is not None:
        logger(f"[ppnn] sys.path injection OK (version {v})")
        return True
    logger("[ppnn] FATAL: all strategies failed")
    return False


# =============================================================================
# INLINED CHECKPOINT (self-contained)
# =============================================================================

_interrupted_flag = {"value": False}


def install_interrupt_handler(logger=print):
    def handler(signum, frame):
        _interrupted_flag["value"] = True
        logger("\n[ckpt] Ctrl-C received; stopping after current cell. "
               "Press again to force abort.")
        signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGINT, handler)


def interrupted():
    return _interrupted_flag["value"]


class CheckpointState:
    """Incremental CSV-based checkpoint. Rows are appended; set of completed
    keys is kept in memory."""

    def __init__(self, csv_path, keys):
        self.csv_path = Path(csv_path)
        self.keys = list(keys)
        self._done = set()
        self._rows = []
        self._header = None

    def load(self):
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            return 0
        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            print(f"[ckpt] Could not read {self.csv_path}: {e}")
            return 0
        if not all(k in df.columns for k in self.keys):
            print(f"[ckpt] {self.csv_path} lacks keys {self.keys}")
            return 0
        self._header = list(df.columns)
        for _, row in df.iterrows():
            k = tuple(self._canon(row[x]) for x in self.keys)
            self._done.add(k)
            self._rows.append(row.to_dict())
        return len(self._done)

    @staticmethod
    def _canon(v):
        if isinstance(v, (int, str)):
            return v
        if isinstance(v, float):
            return int(v) if v.is_integer() else v
        return str(v)

    def is_done(self, key):
        return tuple(self._canon(k) for k in key) in self._done

    def n_done(self):
        return len(self._done)

    def mark_done(self, key, row):
        self._done.add(tuple(self._canon(k) for k in key))
        self._rows.append(dict(row))
        new_file = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        if self._header is None and new_file:
            self._header = list(row.keys())
        elif self._header is None:
            try:
                self._header = list(pd.read_csv(self.csv_path, nrows=0).columns)
            except Exception:
                self._header = list(row.keys())

        extra = [k for k in row if k not in self._header]
        if extra:
            self._header += extra
            self._safe_rewrite()
            return

        full = {k: row.get(k, "") for k in self._header}
        self._safe_append(full, new_file)

    def _safe_append(self, full_row, new_file, attempts=10, base_delay=0.1):
        """Append one row; retry if Windows/antivirus/OneDrive holds a lock."""
        import time as _time
        for attempt in range(attempts):
            try:
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=self._header)
                    if new_file:
                        w.writeheader()
                    w.writerow(full_row)
                return
            except PermissionError as e:
                if attempt == attempts - 1:
                    # Final attempt failed; try full-rewrite as last resort
                    print(f"[ckpt] append failed after {attempts} retries: {e}. "
                          f"Falling back to full rewrite.")
                    self._safe_rewrite()
                    return
                _time.sleep(base_delay * (2 ** attempt))  # exponential backoff
            except Exception as e:
                print(f"[ckpt] non-retryable error: {e}")
                raise

    def _safe_rewrite(self, attempts=10, base_delay=0.1):
        """Atomic rewrite via temp file + os.replace; retry on lock."""
        import time as _time
        import tempfile
        for attempt in range(attempts):
            try:
                # Write to a uniquely-named temp in the same directory
                dir_ = self.csv_path.parent
                fd, tmp = tempfile.mkstemp(
                    prefix=self.csv_path.stem + ".", suffix=".csv.tmp",
                    dir=dir_)
                try:
                    with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=self._header)
                        w.writeheader()
                        for r in self._rows:
                            w.writerow({k: r.get(k, "") for k in self._header})
                    os.replace(tmp, self.csv_path)
                    return
                except Exception:
                    # Clean up the tmp if the replace failed
                    try:
                        os.unlink(tmp)
                    except Exception:
                        pass
                    raise
            except PermissionError as e:
                if attempt == attempts - 1:
                    print(f"[ckpt] WARNING: rewrite failed after {attempts} "
                          f"retries ({e}). CSV may be stale; in-memory state "
                          f"is intact. Close any program viewing "
                          f"{self.csv_path} (Excel, antivirus, OneDrive).")
                    return
                _time.sleep(base_delay * (2 ** attempt))

    def as_dataframe(self):
        if not self._rows:
            return pd.DataFrame(columns=self._header or self.keys)
        return pd.DataFrame(self._rows)


# =============================================================================
# DATASET LOADERS
# =============================================================================

def load_datasets(logger, skip_financial=False, skip_openml=False,
                  tickers=("SPY", "GLD", "TLT")):
    from sklearn.datasets import (
        load_iris, load_wine, load_breast_cancer, load_diabetes, load_digits,
    )

    out = {}
    out["diabetes"] = (load_diabetes().data, load_diabetes().target, "regression")
    for name, loader in [("iris", load_iris), ("wine", load_wine),
                          ("breast_cancer", load_breast_cancer),
                          ("digits", load_digits)]:
        d = loader()
        out[name] = (d.data, d.target, "classification")

    try:
        from sklearn.datasets import fetch_california_housing
        d = fetch_california_housing()
        out["california_housing"] = (d.data, d.target, "regression")
        logger(f"  california_housing: n={len(d.data)}, d={d.data.shape[1]}")
    except Exception as e:
        logger(f"  california_housing unavailable: {e}")

    if not skip_openml:
        try:
            from sklearn.datasets import fetch_openml
            for name, did in [("boston", 43465), ("segment", 40984)]:
                try:
                    d = fetch_openml(data_id=did, as_frame=False, parser="liac-arff")
                    X = np.asarray(d.data, dtype=float)
                    y = d.target
                    if y.dtype.kind in ("U", "O"):
                        y = pd.Categorical(y).codes.astype(int)
                        task = "classification"
                    else:
                        y = np.asarray(y, dtype=float); task = "regression"
                    out[name] = (X, y, task)
                    logger(f"  {name}: n={len(X)}, d={X.shape[1]}")
                except Exception as e:
                    logger(f"  {name} failed: {e}")
        except Exception as e:
            logger(f"  OpenML unavailable: {e}")

    if not skip_financial:
        try:
            import yfinance as yf
            for label, period in [("3y", "3y"), ("10y", "10y")]:
                try:
                    df = yf.download(list(tickers), period=period,
                                     progress=False, auto_adjust=True)
                    if df.empty:
                        continue
                    close = df["Close"] if "Close" in df.columns.get_level_values(0) else df
                    close = close.dropna()
                    rets = np.log(close / close.shift(1)).dropna()
                    X_today = rets.values
                    X_yest = np.roll(X_today, 1, axis=0)
                    X = np.hstack([X_today[1:], X_yest[1:]])
                    y_tomorrow = np.sign(X_today[1:, 0])
                    X = X[:-1]
                    y = (y_tomorrow[1:] > 0).astype(int)
                    name = f"fin_{label}_{'_'.join(tickers)}"
                    out[name] = (X, y, "classification")
                    logger(f"  {name}: n={len(X)}, d={X.shape[1]}")
                except Exception as e:
                    logger(f"  financial {label} failed: {e}")
        except ImportError:
            logger("  yfinance not installed; skipping financial datasets")

    return out


# =============================================================================
# COMPETITOR IMPLEMENTATIONS
# =============================================================================

class GMM_QDA:
    def __init__(self, n_components=2, task="classification", random_state=0):
        self.K = n_components; self.task = task; self.rs = random_state

    def fit(self, X, y):
        from sklearn.mixture import GaussianMixture
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        self.gmm_ = GaussianMixture(n_components=self.K, covariance_type="full",
                                      random_state=self.rs, reg_covar=1e-4).fit(X)
        self.labels_ = self.gmm_.predict(X)
        self.X_, self.y_ = X, y
        if self.task == "classification":
            self.classes_ = np.unique(y)
            self.per_ = {}
            for k in range(self.K):
                m = self.labels_ == k
                if m.sum() < 2 or len(np.unique(y[m])) < 2:
                    self.per_[k] = None; continue
                try:
                    self.per_[k] = QuadraticDiscriminantAnalysis(reg_param=0.01).fit(X[m], y[m])
                except Exception:
                    self.per_[k] = None
        else:
            self.comp_mean_ = np.zeros(self.K)
            for k in range(self.K):
                m = self.labels_ == k
                self.comp_mean_[k] = y[m].mean() if m.sum() > 0 else y.mean()
        return self

    def predict(self, X):
        a = self.gmm_.predict(X)
        if self.task == "classification":
            out = np.empty(len(X), dtype=self.classes_.dtype)
            for k in range(self.K):
                m = a == k
                if not m.any(): continue
                qda = self.per_.get(k)
                if qda is None:
                    km = self.labels_ == k
                    if km.sum():
                        vals, cnt = np.unique(self.y_[km], return_counts=True)
                        out[m] = vals[cnt.argmax()]
                    else:
                        out[m] = self.classes_[0]
                else:
                    out[m] = qda.predict(X[m])
            return out
        return self.comp_mean_[a]


class MDA:
    def __init__(self, n_subclasses=2, task="classification", random_state=0):
        self.R = n_subclasses; self.task = task; self.rs = random_state

    def fit(self, X, y):
        from sklearn.mixture import GaussianMixture
        if self.task != "classification":
            raise ValueError("MDA is classification only")
        self.classes_ = np.unique(y)
        self.priors_ = np.array([(y == c).mean() for c in self.classes_])
        self.per_ = {}
        for c in self.classes_:
            m = y == c; nc = m.sum()
            R_eff = min(self.R, max(1, nc // 4))
            try:
                self.per_[c] = GaussianMixture(n_components=R_eff,
                                                 covariance_type="full",
                                                 random_state=self.rs,
                                                 reg_covar=1e-3).fit(X[m])
            except Exception:
                self.per_[c] = None
        return self

    def predict(self, X):
        lp = np.full((len(X), len(self.classes_)), -np.inf)
        for j, c in enumerate(self.classes_):
            g = self.per_[c]
            if g is None: continue
            lp[:, j] = g.score_samples(X) + np.log(self.priors_[j] + 1e-12)
        return self.classes_[np.argmax(lp, axis=1)]


class MixtureOfExperts:
    def __init__(self, n_experts=3, task="regression", max_iter=15,
                 tol=1e-4, random_state=0):
        self.K = n_experts; self.task = task
        self.max_iter = max_iter; self.tol = tol; self.rs = random_state

    def fit(self, X, y):
        from sklearn.cluster import KMeans
        from sklearn.linear_model import LinearRegression, LogisticRegression
        n = len(X)
        if self.task == "classification":
            self.classes_ = np.unique(y)

        km = KMeans(n_clusters=self.K, n_init=5, random_state=self.rs).fit(X)
        labels = km.labels_
        resp = np.zeros((n, self.K))
        resp[np.arange(n), labels] = 1.0

        prev = np.inf
        for it in range(self.max_iter):
            self.experts_ = []
            for k in range(self.K):
                w = resp[:, k]
                if w.sum() < 1e-6:
                    self.experts_.append(None); continue
                if self.task == "regression":
                    e = LinearRegression(); e.fit(X, y, sample_weight=w)
                else:
                    e = LogisticRegression(max_iter=100, random_state=self.rs,
                                            solver="lbfgs")
                    try:
                        e.fit(X, y, sample_weight=w)
                    except Exception:
                        e = None
                self.experts_.append(e)

            hard = resp.argmax(axis=1)
            if len(np.unique(hard)) < 2:
                self.gating_ = None
            else:
                try:
                    self.gating_ = LogisticRegression(max_iter=100,
                                                       random_state=self.rs,
                                                       solver="lbfgs").fit(X, hard)
                except Exception:
                    self.gating_ = None

            new_resp = self._resp(X)
            loss = -np.sum(np.log(np.clip(new_resp.max(axis=1), 1e-12, None)))
            resp = new_resp
            if abs(prev - loss) < self.tol * abs(prev + 1e-12):
                break
            prev = loss
        return self

    def _resp(self, X):
        n = len(X)
        if self.gating_ is not None:
            p = self.gating_.predict_proba(X)
            full = np.zeros((n, self.K))
            for j, k in enumerate(self.gating_.classes_):
                full[:, k] = p[:, j]
            zero = full.sum(axis=1) < 1e-12
            full[zero] = 1.0 / self.K
            return full
        return np.full((n, self.K), 1.0 / self.K)

    def predict(self, X):
        pi = self._resp(X)
        if self.task == "regression":
            preds = np.zeros((len(X), self.K))
            for k in range(self.K):
                e = self.experts_[k]
                if e is None: continue
                preds[:, k] = e.predict(X)
            return np.sum(pi * preds, axis=1)
        probs = np.zeros((len(X), len(self.classes_)))
        for k in range(self.K):
            e = self.experts_[k]
            if e is None: continue
            p = e.predict_proba(X)
            aligned = np.zeros((len(X), len(self.classes_)))
            for j, c in enumerate(e.classes_):
                if c in self.classes_:
                    aligned[:, list(self.classes_).index(c)] = p[:, j]
            probs += pi[:, k:k+1] * aligned
        return self.classes_[probs.argmax(axis=1)]


# =============================================================================
# METHOD FACTORIES
# =============================================================================

def build_bench_methods(task, k=10, ppnn_available=True, rs=0):
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    methods = {}
    if task == "regression":
        methods["kNN"] = lambda: KNeighborsRegressor(n_neighbors=k)
        methods["RF"] = lambda: RandomForestRegressor(n_estimators=100,
                                                       random_state=rs, n_jobs=1)
    else:
        methods["kNN"] = lambda: KNeighborsClassifier(n_neighbors=k)
        methods["RF"] = lambda: RandomForestClassifier(n_estimators=100,
                                                        random_state=rs, n_jobs=1)
    methods["GMM_QDA"] = lambda: GMM_QDA(n_components=2, task=task, random_state=rs)
    if task == "classification":
        methods["MDA"] = lambda: MDA(n_subclasses=2, task=task, random_state=rs)
    methods["MoE"] = lambda: MixtureOfExperts(n_experts=3, task=task,
                                                max_iter=15, random_state=rs)
    if ppnn_available:
        from ppnn import PPNNHard, PPkNN, PPNNDispatch
        methods["PPNN_Hard"] = lambda: PPNNHard(
            K=2, divergence="Hellinger", task=task,
            pp_n_restarts=2, max_pp_steps=1, random_state=rs)
        methods["PPkNN"] = lambda: PPkNN(
            k=k, K=2, divergence="Hellinger", task=task,
            pp_n_restarts=2, max_pp_steps=1, random_state=rs)
        methods["PPNNDispatch"] = lambda: PPNNDispatch(
            K=2, k=k, divergence="Hellinger", task=task,
            pp_n_restarts=2, max_pp_steps=1, random_state=rs)
    return methods


def build_robust_methods(task, k=10, ppnn_available=True, rs=0):
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    methods = {}
    if task == "regression":
        methods["kNN"] = lambda: KNeighborsRegressor(n_neighbors=k)
        methods["RF"] = lambda: RandomForestRegressor(n_estimators=100,
                                                       random_state=rs, n_jobs=1)
    else:
        methods["kNN"] = lambda: KNeighborsClassifier(n_neighbors=k)
        methods["RF"] = lambda: RandomForestClassifier(n_estimators=100,
                                                        random_state=rs, n_jobs=1)
    if ppnn_available:
        from ppnn import PPkNN
        for div in ["KL", "Hellinger", "ChiSquared"]:
            def _f(d=div):
                return PPkNN(k=k, K=2, divergence=d, task=task,
                             pp_n_restarts=2, max_pp_steps=1, random_state=rs)
            methods[f"PPkNN_{div}"] = _f
    return methods


# =============================================================================
# PHASE 1: BENCHMARK
# =============================================================================

def run_bench(datasets, k, ppnn_available, state, logger):
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.preprocessing import StandardScaler

    for name, (X, y, task) in datasets.items():
        if interrupted(): break
        logger(f"\n  -- {name} (n={len(X)}, d={X.shape[1]}, {task}) --")
        if task == "regression":
            splitter = KFold(n_splits=5, shuffle=True, random_state=0)
            score_fn = mean_squared_error
        else:
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            score_fn = lambda a, b: 1 - accuracy_score(a, b)
        methods = build_bench_methods(task, k=k, ppnn_available=ppnn_available)

        for fold, (tr, te) in enumerate(splitter.split(X, y)):
            if interrupted(): break
            Xtr, Xte = X[tr], X[te]; ytr, yte = y[tr], y[te]
            sc = StandardScaler().fit(Xtr)
            Xtr = sc.transform(Xtr); Xte = sc.transform(Xte)

            msgs = []
            for m, factory in methods.items():
                if state.is_done((name, m, fold)):
                    continue
                t0 = time.time()
                try:
                    est = factory().fit(Xtr, ytr)
                    yhat = est.predict(Xte)
                    s = float(score_fn(yte, yhat))
                except Exception as e:
                    logger(f"     fold {fold} {m} ERROR: {e}")
                    s = float("nan")
                elapsed = time.time() - t0
                state.mark_done((name, m, fold), {
                    "dataset": name, "n": len(X), "d": X.shape[1],
                    "task": task, "method": m, "fold": fold,
                    "score": s, "time_s": elapsed,
                })
                msgs.append(f"{m}={s:.4f}")
            if msgs:
                logger(f"     fold {fold+1}: " + ", ".join(msgs))


# =============================================================================
# PHASE 2: ROBUSTNESS
# =============================================================================

def contaminate(X, y, eps, task, seed):
    rng = np.random.default_rng(seed)
    n = len(X)
    if eps <= 0:
        return X.copy(), y.copy()
    n_out = int(np.ceil(eps * n / (1 - eps)))
    sd = X.std(axis=0) + 1e-8; mu = X.mean(axis=0)
    X_out = mu + (2 * rng.random((n_out, X.shape[1])) - 1) * 5 * sd
    if task == "regression":
        y_sd = float(y.std() + 1e-8); y_mu = float(y.mean())
        y_out = y_mu + (2 * rng.random(n_out) - 1) * 5 * y_sd
    else:
        y_out = rng.choice(np.unique(y), size=n_out)
    X_new = np.vstack([X, X_out])
    y_new = np.concatenate([y, y_out])
    p = rng.permutation(len(X_new))
    return X_new[p], y_new[p]


def run_robust(subset, k, eps_list, n_reps, ppnn_available, state, logger):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.preprocessing import StandardScaler

    for name, (X, y, task) in subset.items():
        if interrupted(): break
        logger(f"\n  -- {name} (n={len(X)}, {task}) --")
        methods = build_robust_methods(task, k=k, ppnn_available=ppnn_available)
        score_fn = (mean_squared_error if task == "regression"
                    else lambda a, b: 1 - accuracy_score(a, b))

        for rep in range(n_reps):
            if interrupted(): break
            pending = {m: [e for e in eps_list
                           if not state.is_done((name, m, e, rep))]
                       for m in methods}
            if all(not v for v in pending.values()):
                logger(f"     rep {rep+1}/{n_reps}: all done (resumed)")
                continue

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.3, random_state=rep)
            sc = StandardScaler().fit(X_tr)
            X_tr_s = sc.transform(X_tr); X_te_s = sc.transform(X_te)

            clean = {}
            for m, factory in methods.items():
                if not any(e > 0 for e in pending[m]):
                    clean[m] = None; continue
                try:
                    est = factory().fit(X_tr_s, y_tr)
                    clean[m] = est.predict(X_te_s)
                except Exception:
                    clean[m] = None

            n_new = 0
            for eps in eps_list:
                if interrupted(): break
                Xc, yc = contaminate(X_tr, y_tr, eps, task, seed=rep)
                sc_c = StandardScaler().fit(Xc)
                Xc_s = sc_c.transform(Xc); Xte_c = sc_c.transform(X_te)
                for m, factory in methods.items():
                    if state.is_done((name, m, float(eps), rep)):
                        continue
                    t0 = time.time()
                    try:
                        est = factory().fit(Xc_s, yc)
                        yhat = est.predict(Xte_c)
                        s = float(score_fn(y_te, yhat))
                        cy = clean.get(m)
                        if cy is not None:
                            stab = float(np.mean(np.abs(yhat - cy))
                                          if task == "regression"
                                          else np.mean(yhat != cy))
                        else:
                            stab = float("nan")
                    except Exception as e:
                        logger(f"     {m} eps={eps} ERROR: {e}")
                        s = stab = float("nan")
                    state.mark_done((name, m, float(eps), rep), {
                        "dataset": name, "method": m, "eps": float(eps),
                        "rep": rep, "score": s, "stability": stab,
                        "time_s": time.time() - t0,
                    })
                    n_new += 1
            logger(f"     rep {rep+1}/{n_reps}: {n_new} cells computed")


# =============================================================================
# PHASE 3: OUTPUTS
# =============================================================================

LATEX_ESC = {"_": r"\_", "&": r"\&", "%": r"\%", "#": r"\#"}

def latex_escape(s):
    r = str(s)
    for k, v in LATEX_ESC.items():
        r = r.replace(k, v)
    return r


def make_bench_table(df, output_path):
    method_order = ["kNN", "RF", "GMM_QDA", "MDA", "MoE",
                    "PPNN_Hard", "PPkNN", "PPNNDispatch"]
    present = [m for m in method_order if m in df["method"].unique()]
    lines = [r"\begin{table}[ht]", r"\centering", r"\tiny",
             r"\begin{tabular}{lrrl" + "c" * len(present) + "}",
             r"\toprule",
             " & ".join(["Dataset", "$n$", "$d$", "Task"]
                        + [latex_escape(m) for m in present]) + r" \\",
             r"\midrule"]
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        info = sub.iloc[0]
        means = {}
        for m in present:
            arr = sub[sub["method"] == m]["score"].dropna().values
            if len(arr):
                means[m] = (arr.mean(), arr.std(ddof=1) if len(arr) > 1 else 0)
        if not means:
            continue
        best = min(means, key=lambda k: means[k][0])
        row = [latex_escape(ds), str(int(info["n"])), str(int(info["d"])), info["task"]]
        for m in present:
            if m not in means:
                row.append("---")
            else:
                mu, sd = means[m]
                cell = f"${mu:.4f}$ {{\\tiny$\\pm{sd:.4f}$}}"
                if m == best:
                    cell = r"\textbf{" + cell + "}"
                row.append(cell)
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Real-data evaluation: $5$-fold CV mean $\pm$ std, "
              r"best in \textbf{bold}. MSE for regression, error rate for "
              r"classification.}",
              r"\label{tab:all_methods}", r"\end{table}"]
    Path(output_path).write_text("\n".join(lines))


def plot_heatmap_relative(df, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_order = ["kNN", "RF", "GMM_QDA", "MDA", "MoE",
                    "PPNN_Hard", "PPkNN", "PPNNDispatch"]
    present = [m for m in method_order if m in df["method"].unique()]
    datasets, mat = [], []
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        means = {}
        for m in present:
            arr = sub[sub["method"] == m]["score"].dropna().values
            if len(arr):
                means[m] = arr.mean()
        if not means:
            continue
        best = min(means.values())
        row = [100 * (means[m] - best) / max(abs(best), 1e-12) if m in means else np.nan
               for m in present]
        datasets.append(ds); mat.append(row)
    if not mat:
        return
    M = np.array(mat)
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(present)),
                                     max(4, 0.4 * len(datasets))))
    vmax = np.nanpercentile(np.abs(M), 95) if np.any(~np.isnan(M)) else 100
    im = ax.imshow(M, cmap="RdYlGn_r", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=30, ha="right")
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)
    ax.set_title("Relative degradation from best (%) per dataset")
    plt.colorbar(im, ax=ax, label="% worse than best")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isnan(v):
                c = "white" if v > vmax * 0.6 else "black"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=8, color=c)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_rank_boxplot(df, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_order = ["kNN", "RF", "GMM_QDA", "MDA", "MoE",
                    "PPNN_Hard", "PPkNN", "PPNNDispatch"]
    present = [m for m in method_order if m in df["method"].unique()]
    ranks = {m: [] for m in present}
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        means = {}
        for m in present:
            arr = sub[sub["method"] == m]["score"].dropna().values
            if len(arr):
                means[m] = arr.mean()
        if not means:
            continue
        srt = sorted(means.items(), key=lambda kv: kv[1])
        for r, (m, _) in enumerate(srt, 1):
            ranks[m].append(r)
    data = [ranks[m] for m in present if ranks[m]]
    labels = [m for m in present if ranks[m]]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4472c4"); patch.set_alpha(0.6)
    for i, m in enumerate(labels):
        xs = np.random.normal(i + 1, 0.06, size=len(ranks[m]))
        ax.scatter(xs, ranks[m], alpha=0.7, s=26, edgecolor="k", linewidth=0.4)
    ax.set_ylabel("Rank on dataset (1 = best)")
    ax.set_title("Distribution of method ranks across datasets")
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_divergence_comparison(df, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = df[df["method"].str.startswith("PPkNN_")].copy()
    if sub.empty:
        return
    divs = ["PPkNN_KL", "PPkNN_Hellinger", "PPkNN_ChiSquared"]
    colors = {"PPkNN_KL": "#c0504d", "PPkNN_Hellinger": "#4472c4",
              "PPkNN_ChiSquared": "#f39c12"}
    labels = {"PPkNN_KL": "KL", "PPkNN_Hellinger": "Hellinger",
              "PPkNN_ChiSquared": r"$\chi^2$"}
    agg = sub.groupby(["dataset", "method", "eps"])["stability"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in divs:
        ms = agg[agg["method"] == m]
        if ms.empty: continue
        piv = ms.pivot_table(values="stability", index="eps", columns="dataset")
        mean = piv.mean(axis=1); std = piv.std(axis=1)
        ci = 1.96 * std / np.sqrt(max(piv.shape[1], 1))
        ax.errorbar(mean.index, mean.values, yerr=ci.values,
                    color=colors[m], marker="o", label=labels[m],
                    capsize=5, linewidth=2.5, markersize=8)
    ax.set_xlabel(r"Contamination fraction $\varepsilon$")
    ax.set_ylabel("Mean prediction change (across datasets)")
    ax.set_title(r"Divergence robustness: KL vs Hellinger vs $\chi^2$")
    ax.grid(alpha=0.3); ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def make_robust_table(df, output_path):
    if df.empty:
        return
    eps_list = sorted(df["eps"].unique())
    lines = [r"\begin{table}[ht]", r"\centering", r"\scriptsize",
             r"\begin{tabular}{ll" + "c" * len(eps_list) + "}",
             r"\toprule",
             r"Dataset & Method & "
             + " & ".join(f"$\\varepsilon={e:.2f}$" for e in eps_list)
             + r" \\",
             r"\midrule"]
    datasets = df["dataset"].unique()
    for i, ds in enumerate(datasets):
        sdf = df[df["dataset"] == ds]
        for m in sdf["method"].unique():
            row = [latex_escape(ds), latex_escape(m)]
            for e in eps_list:
                vals = sdf[(sdf["method"] == m) & (sdf["eps"] == e)]["stability"].dropna()
                row.append(f"${vals.mean():.4f}$" if len(vals) else "---")
            lines.append(" & ".join(row) + r" \\")
        if i < len(datasets) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Prediction stability: mean $|\hat y_{\text{contam}} - "
              r"\hat y_{\text{clean}}|$ at each contamination level. Lower is better.}",
              r"\label{tab:robustness}", r"\end{table}"]
    Path(output_path).write_text("\n".join(lines))


def make_combined_section(bench_df, robust_df, figs_dir_rel, output_path,
                           ppnn_available, out_dir_name):
    parts = [
        r"\section{Complete Empirical Evaluation}",
        r"\label{sec:full_eval}",
        "",
        r"This section reports the complete empirical evaluation produced by "
        r"\texttt{run\_all.py}: a real-data benchmark against competitors "
        r"(Phase 1) and a contamination-based robustness study (Phase 2).",
        "",
        r"\subsection{Phase 1: benchmark against competitors}",
        r"\label{sec:bench_full}",
        "",
        r"We evaluate the full method spectrum: $k$-NN baseline, Random Forest "
        r"ensemble, generative methods (GMM+QDA, MDA), Mixture-of-Experts, and "
        r"the three PP-NN variants (hard, PP-$k$NN hybrid, dispatch). "
        r"Methodology: $5$-fold CV, features standardised per fold, default "
        r"hyper-parameters.",
        "",
    ]
    if not bench_df.empty:
        parts += [
            rf"\input{{{out_dir_name}/table_bench.tex}}",
            "",
            r"\begin{figure}[ht]", r"\centering",
            rf"\includegraphics[width=0.9\textwidth]{{{figs_dir_rel}/heatmap_relative.png}}",
            r"\caption{Relative degradation (\%) per (dataset, method) pair.}",
            r"\label{fig:full_heatmap}", r"\end{figure}",
            "",
            r"\begin{figure}[ht]", r"\centering",
            rf"\includegraphics[width=0.85\textwidth]{{{figs_dir_rel}/rank_distribution.png}}",
            r"\caption{Distribution of per-dataset method ranks.}",
            r"\label{fig:full_ranks}", r"\end{figure}",
        ]
    parts += [
        "",
        r"\subsection{Phase 2: empirical robustness under contamination}",
        r"\label{sec:robust_full}",
        "",
        r"Contamination protocol: for each $\varepsilon \in \{0, 0.05, 0.10, 0.20\}$, "
        r"adversarial outliers (features at $\pm 5\sigma$, response chosen far from "
        r"the bulk) are added to the training set; methods are evaluated on the clean "
        r"test set. Five replications are averaged. The stability metric is the mean "
        r"absolute change in predictions between clean and contaminated fits, the "
        r"direct empirical counterpart of the influence-function gross-error "
        r"sensitivity.",
        "",
    ]
    if not robust_df.empty:
        parts += [
            rf"\input{{{out_dir_name}/table_robustness.tex}}",
            "",
        ]
        if ppnn_available:
            parts += [
                r"\begin{figure}[ht]", r"\centering",
                rf"\includegraphics[width=0.85\textwidth]{{{figs_dir_rel}/divergence_comparison.png}}",
                r"\caption{Mean prediction stability vs contamination level, averaged "
                r"across datasets. Hellinger is predicted by the theory to be the most "
                r"robust; KL and $\chi^2$ are predicted to be less robust.}",
                r"\label{fig:divs}", r"\end{figure}",
            ]
    Path(output_path).write_text("\n".join(parts))


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="results_all")
    p.add_argument("--ppnn-path", default=None,
                   help="Path to ppnn_lib/ or ppnn*.tar.gz")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-reps", type=int, default=5)
    p.add_argument("--eps", nargs="+", type=float,
                   default=[0.0, 0.05, 0.10, 0.20])
    p.add_argument("--skip-financial", action="store_true")
    p.add_argument("--skip-openml", action="store_true")
    p.add_argument("--skip-bench", action="store_true",
                   help="Skip Phase 1 (benchmark)")
    p.add_argument("--skip-robustness", action="store_true",
                   help="Skip Phase 2 (robustness)")
    p.add_argument("--only-robustness", action="store_true",
                   help="Run only Phase 2")
    p.add_argument("--skip-slow", action="store_true",
                   help="Skip slow datasets (digits, california_housing, "
                        "segment) to finish faster")
    p.add_argument("--tickers", nargs="+", default=["SPY", "GLD", "TLT"])
    args = p.parse_args()

    out = Path(args.output_dir).resolve()
    (out / "figs").mkdir(parents=True, exist_ok=True)
    log_lines = []

    def logger(msg):
        t = datetime.now().strftime("%H:%M:%S")
        line = f"[{t}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    logger("=" * 60)
    logger("PP-NN Complete Evaluation Pipeline (run_all.py)")
    logger("=" * 60)
    logger(f"Output: {out}")
    logger(f"Python: {sys.executable}")

    logger("\n--- Phase 0: ensuring ppnn is importable ---")
    ppnn_available = ensure_ppnn(args.ppnn_path, logger=logger)
    if not ppnn_available:
        logger("WARNING: Running without ppnn. The PP-NN columns will be "
               "missing from all tables and figures.")

    logger("\n--- Loading datasets ---")
    datasets = load_datasets(logger,
                              skip_financial=args.skip_financial,
                              skip_openml=args.skip_openml,
                              tickers=tuple(args.tickers))
    if args.skip_slow:
        SLOW = {"digits", "california_housing", "segment"}
        dropped = [n for n in list(datasets.keys()) if n in SLOW]
        for n in dropped:
            del datasets[n]
        if dropped:
            logger(f"  --skip-slow: dropped {dropped}")
    logger(f"Loaded {len(datasets)} datasets: {list(datasets.keys())}")

    # Time estimate based on heuristic (PP-NN ~60s/fold at d<=30, 300s/fold at d=64)
    if ppnn_available and not args.skip_bench and not args.only_robustness:
        est_total = 0
        for name, (X, y, task) in datasets.items():
            d = X.shape[1]; n = len(X)
            # PP-NN cost per fold scales roughly with d * sqrt(n) in our setup
            per_fold = 5 + 1.2 * d * (n ** 0.5) / 30
            est_total += 5 * per_fold * 3  # 5 folds x 3 PP-NN variants
        logger(f"[estimate] Phase 1 PP-NN work ~= {est_total/60:.0f} min "
               f"(plus ~1 min baselines). Checkpointing: you can Ctrl-C at any "
               f"time and resume by re-running the command.")

    install_interrupt_handler(logger)

    if args.only_robustness or args.skip_bench:
        logger("\n(Phase 1 skipped)")
        bench_df = pd.DataFrame()
    else:
        logger("\n" + "=" * 60)
        logger("PHASE 1: Benchmark")
        logger("=" * 60)
        state1 = CheckpointState(out / "bench_results.csv",
                                  keys=["dataset", "method", "fold"])
        n_prior = state1.load()
        if n_prior:
            logger(f"[ckpt] Resuming bench: {n_prior} cells already done")
        run_bench(datasets, args.k, ppnn_available, state1, logger)
        bench_df = state1.as_dataframe()
        logger(f"Phase 1 complete: {len(bench_df)} rows")

    if args.skip_robustness:
        logger("\n(Phase 2 skipped)")
        robust_df = pd.DataFrame()
    elif interrupted():
        logger("\n[ckpt] Interrupted after Phase 1; skipping Phase 2.")
        robust_df = pd.DataFrame()
    else:
        logger("\n" + "=" * 60)
        logger("PHASE 2: Robustness")
        logger("=" * 60)
        robust_names = [n for n in ["diabetes", "iris", "wine", "breast_cancer"]
                        if n in datasets]
        subset = {n: datasets[n] for n in robust_names}
        logger(f"Robustness on {len(subset)} datasets: {robust_names}")
        state2 = CheckpointState(out / "robust_results.csv",
                                  keys=["dataset", "method", "eps", "rep"])
        n_prior = state2.load()
        if n_prior:
            logger(f"[ckpt] Resuming robustness: {n_prior} cells already done")
        run_robust(subset, args.k, args.eps, args.n_reps,
                    ppnn_available, state2, logger)
        robust_df = state2.as_dataframe()
        logger(f"Phase 2 complete: {len(robust_df)} rows")

    logger("\n" + "=" * 60)
    logger("PHASE 3: Outputs")
    logger("=" * 60)
    figs_dir = out / "figs"
    if not bench_df.empty:
        make_bench_table(bench_df, out / "table_bench.tex")
        logger("Wrote table_bench.tex")
        plot_heatmap_relative(bench_df, figs_dir / "heatmap_relative.png")
        plot_rank_boxplot(bench_df, figs_dir / "rank_distribution.png")
        logger("Wrote bench figures")
    if not robust_df.empty:
        make_robust_table(robust_df, out / "table_robustness.tex")
        logger("Wrote table_robustness.tex")
        if ppnn_available:
            plot_divergence_comparison(robust_df, figs_dir / "divergence_comparison.png")
            logger("Wrote divergence_comparison.png")
    make_combined_section(bench_df, robust_df, figs_dir_rel="figs",
                           output_path=out / "section_full_eval.tex",
                           ppnn_available=ppnn_available,
                           out_dir_name=out.name)
    logger("Wrote section_full_eval.tex")

    (out / "run_log.txt").write_text("\n".join(log_lines))

    logger("\n" + "=" * 60)
    logger("DONE")
    logger("=" * 60)
    logger(f"To include in your LaTeX article:")
    logger(rf"  \input{{{out.name}/section_full_eval.tex}}")
    logger(f"  (and copy {out.name}/figs/ next to your .tex file)")
    if not ppnn_available:
        logger("")
        logger("WARNING: ppnn was not importable. Rerun after placing "
               "ppnn-0.1.0.tar.gz or ppnn_lib/ next to this script to get "
               "the PP-NN columns.")


if __name__ == "__main__":
    main()
