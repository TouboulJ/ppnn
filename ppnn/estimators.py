"""Projection-Pursuit Nearest Neighbors estimators.

Three scikit-learn-compatible estimators built on top of :class:`TouboulPP`:

  * :class:`PPNNHard`    - predicts the aggregated response of the component to
                           which the query is hard-assigned (largest
                           responsibility).
  * :class:`PPNNSoft`    - predicts the responsibility-weighted mixture of
                           per-component means.
  * :class:`PPkNN`       - hybrid: restricts a standard k-NN search to the
                           component assigned by PP.

All three support both regression and classification via a ``task`` argument
and share a common border-handling mechanism through ``margin_threshold``:
queries whose top-1 responsibility is below this threshold are dispatched to a
configurable fallback (plain k-NN by default).

A wrapper :class:`PPNNDispatch` automates the p-value-gated selection between
:class:`PPNNHard`, :class:`PPkNN`, and a plain :class:`sklearn.neighbors.KNeighborsRegressor`
/ :class:`sklearn.neighbors.KNeighborsClassifier`.
Author : Touboul Jacques
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .core import TouboulPP, DIVERGENCES, PhiDivergence


# =============================================================================
# Internal helper: partition training data into PP components
# =============================================================================

def _pp_components(X, pp: TouboulPP, K: int, rng):
    """Use the fitted TouboulPP directions to partition X into K components.

    Touboul's algorithm yields a sequence of directions (a_1, a_2, ...). We use
    them as features for a Gaussian mixture model fit in the low-dimensional
    projected space; the GMM responsibilities then define soft component
    membership, and the argmax gives hard labels.

    Parameters
    ----------
    X : ndarray of shape (n, d)
    pp : fitted TouboulPP
    K : int, number of components (mixture order)
    rng : np.random.Generator

    Returns
    -------
    hard_labels : ndarray of shape (n,)
    responsibilities : ndarray of shape (n, K)
    gmm : fitted GaussianMixture (used to predict responsibilities on new X)
    directions : ndarray of shape (d, n_directions) stacking a_k's
    """
    if not pp.steps_:
        # fallback: fit GMM directly on X
        gmm = GaussianMixture(
            n_components=K, random_state=int(rng.integers(0, 2**31 - 1))
        ).fit(X)
        resp = gmm.predict_proba(X)
        return resp.argmax(axis=1), resp, gmm, np.empty((X.shape[1], 0))

    A = np.column_stack([s.a for s in pp.steps_])  # shape (d, n_steps)
    Z = X @ A                                      # projected features
    gmm = GaussianMixture(
        n_components=K,
        covariance_type="full",
        random_state=int(rng.integers(0, 2**31 - 1)),
        reg_covar=1e-4,
    ).fit(Z)
    resp = gmm.predict_proba(Z)
    return resp.argmax(axis=1), resp, gmm, A


def _responsibilities_new(X_new, gmm, A):
    """Compute responsibilities for new queries using the cached GMM."""
    if A.shape[1] == 0:
        return gmm.predict_proba(X_new)
    return gmm.predict_proba(X_new @ A)


# =============================================================================
# Base class with shared fit-machinery
# =============================================================================

class _PPNNBase(BaseEstimator):
    """Common fit code for the PP-NN family."""

    def __init__(
        self,
        K: int = 2,
        divergence: Union[str, PhiDivergence] = "Hellinger",
        task: Literal["regression", "classification"] = "regression",
        margin_threshold: float = 0.0,
        fallback: Literal["knn", "prior", "none"] = "knn",
        n_neighbors_fallback: int = 15,
        max_pp_steps: int = 3,
        pp_n_restarts: int = 15,
        pp_alpha: float = 0.05,
        random_state: Optional[int] = 0,
    ):
        self.K = K
        self.divergence = divergence
        self.task = task
        self.margin_threshold = margin_threshold
        self.fallback = fallback
        self.n_neighbors_fallback = n_neighbors_fallback
        self.max_pp_steps = max_pp_steps
        self.pp_n_restarts = pp_n_restarts
        self.pp_alpha = pp_alpha
        self.random_state = random_state

    # sklearn: get_params / set_params inherited from BaseEstimator

    def _make_rng(self):
        return np.random.default_rng(self.random_state)

    def _fit_pp(self, X, y):
        rng = self._make_rng()
        # Fit PP on covariates only
        self.pp_ = TouboulPP(
            divergence=self.divergence,
            max_steps=self.max_pp_steps,
            n_restarts=self.pp_n_restarts,
            alpha=self.pp_alpha,
            rng=rng,
        ).fit(X)
        self.labels_, self.resp_, self.gmm_, self.A_ = _pp_components(
            X, self.pp_, self.K, rng
        )
        self.X_train_ = X
        self.y_train_ = y
        # Fit fallback estimator once for border queries
        if self.fallback == "knn":
            if self.task == "regression":
                self.fallback_est_ = KNeighborsRegressor(
                    n_neighbors=min(self.n_neighbors_fallback, len(X))
                ).fit(X, y)
            else:
                self.fallback_est_ = KNeighborsClassifier(
                    n_neighbors=min(self.n_neighbors_fallback, len(X))
                ).fit(X, y)
        else:
            self.fallback_est_ = None
        self.n_features_in_ = X.shape[1]
        if self.task == "classification":
            self.classes_ = np.unique(y)

    def _prior_prediction(self):
        if self.task == "regression":
            return float(self.y_train_.mean())
        else:
            vals, counts = np.unique(self.y_train_, return_counts=True)
            return vals[counts.argmax()]

    def _below_margin(self, resp_new):
        """Boolean mask of queries with top-responsibility below threshold."""
        if self.margin_threshold <= 0:
            return np.zeros(len(resp_new), dtype=bool)
        top = np.max(resp_new, axis=1)
        return top < self.margin_threshold

    def _apply_fallback(self, X_new, mask, base_predictions):
        """In-place patch of base_predictions on the mask rows."""
        if not np.any(mask):
            return base_predictions
        if self.fallback == "knn" and self.fallback_est_ is not None:
            base_predictions[mask] = self.fallback_est_.predict(X_new[mask])
        elif self.fallback == "prior":
            base_predictions[mask] = self._prior_prediction()
        # "none" -> keep base predictions unchanged
        return base_predictions


# =============================================================================
# PPNNHard
# =============================================================================

class PPNNHard(_PPNNBase, RegressorMixin, ClassifierMixin):
    """Hard-assignment Projection-Pursuit Nearest Neighbors.

    At training time, fits the Touboul phi-divergence projection pursuit to the
    covariates X and partitions the cloud into ``K`` components. For each
    component ``l``, precomputes a constant predictor (mean of y for regression,
    majority class for classification).

    At prediction time, the query is assigned to the component with the largest
    responsibility; the component's precomputed predictor is returned.

    Parameters
    ----------
    K : int, default=2
        Number of mixture components.
    divergence : str or PhiDivergence, default="Hellinger"
        Which phi-divergence to use inside the PP optimization. Supported
        strings: ``"KL"``, ``"Hellinger"``, ``"ChiSquared"``.
    task : {"regression", "classification"}, default="regression"
    margin_threshold : float in [0, 1], default=0.0
        Queries whose top responsibility is below this threshold are routed to
        the fallback estimator (see ``fallback``). 0 disables the mechanism.
    fallback : {"knn", "prior", "none"}, default="knn"
        How to handle low-margin queries when ``margin_threshold > 0``.
    n_neighbors_fallback : int, default=15
        Number of neighbors used by the k-NN fallback.
    max_pp_steps : int, default=3
        Maximum number of PP iterations (directions extracted).
    pp_n_restarts : int, default=15
        Number of random restarts per PP step (non-convex optimization).
    pp_alpha : float, default=0.05
        Significance level of the per-step factorization test.
    random_state : int or None, default=0

    Attributes
    ----------
    pp_ : TouboulPP
        The fitted projection-pursuit backbone.
    labels_ : ndarray of shape (n_train,)
        Hard component labels of the training points.
    resp_ : ndarray of shape (n_train, K)
        Training responsibilities.
    component_prediction_ : dict
        Mapping from component index to its constant predictor.
    """

    def fit(self, X: ArrayLike, y: ArrayLike) -> "PPNNHard":
        X, y = check_X_y(X, y, ensure_min_samples=max(10, 2 * self.K))
        self._fit_pp(X, y)
        self.component_prediction_ = {}
        for lab in range(self.K):
            idx = self.labels_ == lab
            if idx.sum() == 0:
                self.component_prediction_[lab] = self._prior_prediction()
            elif self.task == "regression":
                self.component_prediction_[lab] = float(y[idx].mean())
            else:
                vals, cnt = np.unique(y[idx], return_counts=True)
                self.component_prediction_[lab] = vals[cnt.argmax()]
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X)
        resp = _responsibilities_new(X, self.gmm_, self.A_)
        assign = resp.argmax(axis=1)
        out = np.array([self.component_prediction_[a] for a in assign])
        mask = self._below_margin(resp)
        out = self._apply_fallback(X, mask, out)
        return out

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Classification probabilities via responsibility-weighted class posteriors.

        For classification tasks only. The probability of class j at query x is
        computed as sum_l pi_l(x) * p_l(j), where p_l(j) is the empirical class
        frequency in component l.
        """
        if self.task != "classification":
            raise AttributeError("predict_proba available only in classification task")
        check_is_fitted(self)
        X = check_array(X)
        resp = _responsibilities_new(X, self.gmm_, self.A_)

        # Per-component class frequencies
        cls_probs = np.zeros((self.K, len(self.classes_)))
        for lab in range(self.K):
            idx = self.labels_ == lab
            if idx.sum() == 0:
                cls_probs[lab] = 1.0 / len(self.classes_)
                continue
            for j, c in enumerate(self.classes_):
                cls_probs[lab, j] = np.mean(self.y_train_[idx] == c)
        return resp @ cls_probs


# =============================================================================
# PPNNSoft
# =============================================================================

class PPNNSoft(_PPNNBase, RegressorMixin, ClassifierMixin):
    """Soft, responsibility-weighted Projection-Pursuit Nearest Neighbors.

    Instead of hard assignment, every component contributes to the prediction
    in proportion to the query's responsibility for that component.

    For regression:
        y_hat(x) = sum_l  pi_l(x) * bar_y_l
    where bar_y_l is the per-component mean of y.

    For classification, the component-wise class probabilities are aggregated
    by responsibility and the most probable class is returned.

    Parameters are identical to :class:`PPNNHard`.
    """

    def fit(self, X: ArrayLike, y: ArrayLike) -> "PPNNSoft":
        X, y = check_X_y(X, y, ensure_min_samples=max(10, 2 * self.K))
        self._fit_pp(X, y)
        # Precompute per-component mean (for regression) or class proportions
        if self.task == "regression":
            self.component_mean_ = np.zeros(self.K)
            for lab in range(self.K):
                idx = self.labels_ == lab
                self.component_mean_[lab] = (
                    float(y[idx].mean())
                    if idx.sum() > 0
                    else self._prior_prediction()
                )
        else:
            self.component_class_probs_ = np.zeros((self.K, len(self.classes_)))
            for lab in range(self.K):
                idx = self.labels_ == lab
                if idx.sum() == 0:
                    self.component_class_probs_[lab] = 1.0 / len(self.classes_)
                else:
                    for j, c in enumerate(self.classes_):
                        self.component_class_probs_[lab, j] = np.mean(y[idx] == c)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X)
        resp = _responsibilities_new(X, self.gmm_, self.A_)
        if self.task == "regression":
            out = resp @ self.component_mean_
        else:
            class_probs = resp @ self.component_class_probs_
            out = self.classes_[class_probs.argmax(axis=1)]
        mask = self._below_margin(resp)
        out = self._apply_fallback(X, mask, out)
        return out

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        if self.task != "classification":
            raise AttributeError("predict_proba available only in classification task")
        check_is_fitted(self)
        X = check_array(X)
        resp = _responsibilities_new(X, self.gmm_, self.A_)
        return resp @ self.component_class_probs_


# =============================================================================
# PPkNN (hybrid)
# =============================================================================

class PPkNN(_PPNNBase, RegressorMixin, ClassifierMixin):
    """Hybrid Projection-Pursuit + local k-NN.

    A query is assigned (hard) to its top-responsibility component; a k-NN
    restricted to the training points of that component is then used for
    prediction. This hedges against misspecification of the mixture: within
    each component, the k-NN local smoother remains in force.

    Additional parameter
    --------------------
    k : int, default=10
        Number of neighbors within the component. Capped at component size.
    """

    def __init__(self, k: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def fit(self, X: ArrayLike, y: ArrayLike) -> "PPkNN":
        X, y = check_X_y(X, y, ensure_min_samples=max(10, 2 * self.K))
        self._fit_pp(X, y)
        self.component_knns_ = {}
        for lab in range(self.K):
            idx = self.labels_ == lab
            if idx.sum() < 1:
                self.component_knns_[lab] = None
                continue
            k_eff = min(self.k, idx.sum())
            if self.task == "regression":
                knn = KNeighborsRegressor(n_neighbors=k_eff)
            else:
                knn = KNeighborsClassifier(n_neighbors=k_eff)
            knn.fit(X[idx], y[idx])
            self.component_knns_[lab] = knn
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X)
        resp = _responsibilities_new(X, self.gmm_, self.A_)
        assign = resp.argmax(axis=1)
        out = np.empty(
            len(X), dtype=float if self.task == "regression" else object
        )
        for lab in range(self.K):
            mask = assign == lab
            if not mask.any():
                continue
            knn = self.component_knns_[lab]
            if knn is None:
                if self.fallback == "knn" and self.fallback_est_ is not None:
                    out[mask] = self.fallback_est_.predict(X[mask])
                else:
                    out[mask] = self._prior_prediction()
            else:
                out[mask] = knn.predict(X[mask])
        border_mask = self._below_margin(resp)
        out = self._apply_fallback(X, border_mask, out)
        if self.task == "classification":
            out = out.astype(self.classes_.dtype)
        else:
            out = out.astype(float)
        return out

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        if self.task != "classification":
            raise AttributeError("predict_proba available only in classification task")
        check_is_fitted(self)
        X = check_array(X)
        resp = _responsibilities_new(X, self.gmm_, self.A_)
        assign = resp.argmax(axis=1)
        probs = np.zeros((len(X), len(self.classes_)))
        for lab in range(self.K):
            mask = assign == lab
            if not mask.any():
                continue
            knn = self.component_knns_[lab]
            if knn is None:
                probs[mask] = 1.0 / len(self.classes_)
            else:
                p = knn.predict_proba(X[mask])
                # Align knn's class order with self.classes_
                aligned = np.zeros((mask.sum(), len(self.classes_)))
                for j, c in enumerate(knn.classes_):
                    if c in self.classes_:
                        aligned[:, list(self.classes_).index(c)] = p[:, j]
                probs[mask] = aligned
        return probs


# =============================================================================
# PPNNDispatch: p-value gated selection
# =============================================================================

class PPNNDispatch(BaseEstimator):
    """Dispatcher that selects between k-NN, PP-NN, and PP-kNN based on p-value.

    The factorization p-value reported by :class:`TouboulPP` after fitting on X
    drives a three-tier decision rule:

    - p > accept_threshold  -> use :class:`PPNNHard`  (or :class:`PPNNSoft`)
    - reject < p <= accept  -> use :class:`PPkNN`     (safe hybrid)
    - p <= reject_threshold -> fall back to plain k-NN

    Parameters
    ----------
    accept_threshold : float, default=0.10
    reject_threshold : float, default=0.01
    soft : bool, default=False
        If True, when the test accepts we use :class:`PPNNSoft` instead of
        :class:`PPNNHard`.
    k : int, default=10
        Number of neighbors for PPkNN / plain k-NN.
    task : {"regression", "classification"}, default="regression"
    Other parameters are passed through to the selected estimator.
    """

    def __init__(
        self,
        accept_threshold: float = 0.10,
        reject_threshold: float = 0.01,
        soft: bool = False,
        K: int = 2,
        k: int = 10,
        divergence: Union[str, PhiDivergence] = "Hellinger",
        task: Literal["regression", "classification"] = "regression",
        margin_threshold: float = 0.0,
        max_pp_steps: int = 3,
        pp_n_restarts: int = 15,
        random_state: Optional[int] = 0,
    ):
        self.accept_threshold = accept_threshold
        self.reject_threshold = reject_threshold
        self.soft = soft
        self.K = K
        self.k = k
        self.divergence = divergence
        self.task = task
        self.margin_threshold = margin_threshold
        self.max_pp_steps = max_pp_steps
        self.pp_n_restarts = pp_n_restarts
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> "PPNNDispatch":
        X, y = check_X_y(X, y, ensure_min_samples=20)
        rng = np.random.default_rng(self.random_state)
        # Run one PP fit to obtain the factorization p-value
        probe = TouboulPP(
            divergence=self.divergence,
            max_steps=1,                 # only need step-1 p-value
            n_restarts=self.pp_n_restarts,
            alpha=1.0,                   # never stop; we just want the p-value
            rng=rng,
        ).fit(X)
        self.pvalue_ = probe.steps_[0].p_value if probe.steps_ else 0.0

        # Choose estimator
        common_kwargs = dict(
            K=self.K, divergence=self.divergence, task=self.task,
            margin_threshold=self.margin_threshold,
            max_pp_steps=self.max_pp_steps,
            pp_n_restarts=self.pp_n_restarts,
            random_state=self.random_state,
        )
        if self.pvalue_ > self.accept_threshold:
            self.selected_ = "PPNNSoft" if self.soft else "PPNNHard"
            cls = PPNNSoft if self.soft else PPNNHard
            self.estimator_ = cls(**common_kwargs).fit(X, y)
        elif self.pvalue_ > self.reject_threshold:
            self.selected_ = "PPkNN"
            self.estimator_ = PPkNN(k=self.k, **common_kwargs).fit(X, y)
        else:
            self.selected_ = "kNN"
            Cls = (KNeighborsRegressor if self.task == "regression"
                   else KNeighborsClassifier)
            self.estimator_ = Cls(n_neighbors=min(self.k, len(X))).fit(X, y)
            if self.task == "classification":
                self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.estimator_.predict(check_array(X))

    def predict_proba(self, X):
        if self.task != "classification":
            raise AttributeError("predict_proba available only in classification task")
        check_is_fitted(self)
        return self.estimator_.predict_proba(check_array(X))
