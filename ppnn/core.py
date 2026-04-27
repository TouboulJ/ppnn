"""
Faithful implementation of Touboul (2010) Projection Pursuit through phi-divergence.

Reference: Touboul, J. (2010). "Projection Pursuit Through phi-Divergence
Minimisation." Entropy 12(6):1581-1611.

Core algorithm (eq. 1.3, Touboul 2010):
    At step k:
        a_k = argmin_a D_phi( g^{(k-1)} * f_a/g_a^{(k-1)} ,  f )
        g^{(k)}(x) = g^{(k-1)}(x) * f_{a_k}(a_k'x) / g_{a_k}^{(k-1)}(a_k'x)
    Stop when D_phi(g^{(k)}, f) = 0 (tested via Corollary 3.2).

Empirical estimator (Prop. 3.2, using dual form):
    a_hat = argmin_a M_n(a, a)
    where M_n(b, a) uses the Fenchel dual of phi.

This module provides:
  - PhiDivergence: KL, Hellinger, ChiSquared, Cressie-Read
  - TouboulPP: the full projection pursuit estimator
  - factorisation_test: implementation of Corollary 3.2 for the stopping rule
  - Utilities: elliptical density class (Gaussian here), KDE helpers

Design notes:
  - We use kernel density estimation for f and f_a (Touboul's eq. after 2.2)
  - The instrumental g is a Gaussian with same mean and variance as f
    (Touboul, section 2.1 "Choice of g")
  - Optimization: the min_a requires a smooth non-convex optimization over
    the unit sphere in R^d; we use random multi-start + BFGS refinement
  - The minimax (a, c) is expensive; in the scalar case it simplifies
    because M(b, a, .) becomes tractable
Author : Touboul Jacques
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy import stats
from scipy.optimize import minimize
from typing import Callable, List, Tuple, Optional
from numpy.typing import ArrayLike


# =============================================================================
# PHI-DIVERGENCE DEFINITIONS
# =============================================================================

@dataclass
class PhiDivergence:
    """A phi-divergence defined by phi, phi', and phi* (Fenchel conjugate-related).

    We encode:
      phi(t):      strictly convex, phi(1) = 0, phi''(1) > 0
      phi_prime(t): derivative
      phi_star(s):  phi^#(s) = s*(phi')^{-1}(s) - phi((phi')^{-1}(s))
                    this is the transform appearing in Keziou's dual form
    """
    name: str
    phi: Callable[[np.ndarray], np.ndarray]
    phi_prime: Callable[[np.ndarray], np.ndarray]
    phi_star: Callable[[np.ndarray], np.ndarray]
    phi_double_prime_at_1: float  # used for test statistic normalisation


def _safe(x, eps=1e-12):
    return np.clip(x, eps, None)


# Kullback-Leibler: phi(t) = t log t - t + 1
def _kl():
    return PhiDivergence(
        name="KL",
        phi=lambda t: _safe(t) * np.log(_safe(t)) - t + 1,
        phi_prime=lambda t: np.log(_safe(t)),
        # phi'^{-1}(s) = exp(s). phi^#(s) = s*exp(s) - (exp(s)*s - exp(s) + 1) = exp(s) - 1
        phi_star=lambda s: np.exp(s) - 1,
        phi_double_prime_at_1=1.0,
    )


# Hellinger: phi(t) = (sqrt(t) - 1)^2 = t - 2*sqrt(t) + 1
def _hellinger():
    # phi'(t) = 1 - 1/sqrt(t). phi'^{-1}(s) = 1/(1-s)^2 for s<1.
    # phi^#(s) = s/(1-s)^2 - [(1/(1-s)^2) - 2/(1-s) + 1]
    #         = s/(1-s)^2 - 1/(1-s)^2 + 2/(1-s) - 1
    #         = (s-1)/(1-s)^2 + 2/(1-s) - 1
    #         = -1/(1-s) + 2/(1-s) - 1 = 1/(1-s) - 1 = s/(1-s)
    return PhiDivergence(
        name="Hellinger",
        phi=lambda t: (np.sqrt(_safe(t)) - 1) ** 2,
        phi_prime=lambda t: 1 - 1 / np.sqrt(_safe(t)),
        phi_star=lambda s: np.clip(s, None, 0.999) / (1 - np.clip(s, None, 0.999)),
        phi_double_prime_at_1=0.5,
    )


# Pearson chi-squared: phi(t) = 0.5*(t-1)^2
def _chi2():
    # phi'(t) = t - 1. phi'^{-1}(s) = s + 1.
    # phi^#(s) = s(s+1) - 0.5*s^2 = 0.5*s^2 + s
    return PhiDivergence(
        name="ChiSquared",
        phi=lambda t: 0.5 * (t - 1) ** 2,
        phi_prime=lambda t: t - 1,
        phi_star=lambda s: 0.5 * s ** 2 + s,
        phi_double_prime_at_1=1.0,
    )


# Cressie-Read family: phi_lambda(t) = (t^(lam+1) - t - lam*(t-1)) / (lam*(lam+1))
def cressie_read(lam: float) -> PhiDivergence:
    assert lam not in (0, -1), "use KL for lam=0, Hellinger is lam=-1/2 special case"

    def phi(t):
        t = _safe(t)
        return (t ** (lam + 1) - t - lam * (t - 1)) / (lam * (lam + 1))

    def phi_prime(t):
        t = _safe(t)
        return (t ** lam - 1) / lam

    def phi_star(s):
        # (phi')^{-1}(s) = (1 + lam*s)^{1/lam}
        base = 1 + lam * s
        base = np.where(base <= 0, 1e-12, base)
        t_inv = base ** (1 / lam)
        return s * t_inv - phi(t_inv)

    return PhiDivergence(
        name=f"CressieRead({lam})",
        phi=phi, phi_prime=phi_prime, phi_star=phi_star,
        phi_double_prime_at_1=1.0,
    )


DIVERGENCES = {
    "KL": _kl(),
    "Hellinger": _hellinger(),
    "ChiSquared": _chi2(),
}


# =============================================================================
# GAUSSIAN INSTRUMENTAL DENSITY g (Touboul section 2.1)
# =============================================================================

@dataclass
class GaussianInstrumental:
    """Instrumental g: Gaussian with same mean and variance as f."""
    mu: np.ndarray          # shape (d,)
    Sigma: np.ndarray       # shape (d, d)
    _L: np.ndarray = field(init=False)
    _log_det: float = field(init=False)

    def __post_init__(self):
        self._L = np.linalg.cholesky(self.Sigma + 1e-8 * np.eye(len(self.mu)))
        self._log_det = 2 * np.sum(np.log(np.diag(self._L)))

    @classmethod
    def from_data(cls, X):
        return cls(mu=X.mean(axis=0), Sigma=np.cov(X.T) + 1e-6 * np.eye(X.shape[1]))

    def log_pdf(self, x):
        x = np.atleast_2d(x)
        d = x.shape[1]
        diff = x - self.mu
        sol = np.linalg.solve(self._L, diff.T).T
        quad = np.sum(sol ** 2, axis=1)
        return -0.5 * (d * np.log(2 * np.pi) + self._log_det + quad)

    def pdf(self, x):
        return np.exp(self.log_pdf(x))

    def sample(self, n, rng=None):
        rng = rng or np.random.default_rng()
        return self.mu + rng.standard_normal((n, len(self.mu))) @ self._L.T

    def marginal_pdf_along(self, a, z):
        """pdf of a'X when X ~ g. a in R^d unit vector, z scalar array.

        a'X ~ N(a'mu, a'Sigma a).
        """
        a = np.asarray(a).flatten()
        mu_a = float(a @ self.mu)
        var_a = float(a @ self.Sigma @ a)
        sig_a = np.sqrt(max(var_a, 1e-12))
        return stats.norm.pdf(z, loc=mu_a, scale=sig_a)


# =============================================================================
# KDE ESTIMATORS FOR f AND f_a
# =============================================================================

class KDEMultivariate:
    """Gaussian KDE for multivariate f, used by TouboulPP for evaluating f(x)."""
    def __init__(self, X, bandwidth=None):
        self.X = np.asarray(X)
        self.n, self.d = self.X.shape
        if bandwidth is None:
            # Silverman rule of thumb per-dimension
            stds = np.std(self.X, axis=0) + 1e-8
            bandwidth = stds * self.n ** (-1.0 / (self.d + 4))
        self.h = np.asarray(bandwidth)
        self._log_norm = -0.5 * self.d * np.log(2 * np.pi) - np.sum(np.log(self.h))

    def pdf(self, x):
        x = np.atleast_2d(x)
        # compute log kernel for each pair
        diff = (x[:, None, :] - self.X[None, :, :]) / self.h
        log_k = -0.5 * np.sum(diff ** 2, axis=2) + self._log_norm
        # logsumexp over training points, divide by n
        m = log_k.max(axis=1, keepdims=True)
        return np.exp(m.squeeze(-1)) * np.mean(np.exp(log_k - m), axis=1)


class KDE1D:
    """Univariate KDE for marginal f_a (projection a'X)."""
    def __init__(self, z, bandwidth=None):
        self.z = np.asarray(z).flatten()
        self.n = len(self.z)
        if bandwidth is None:
            s = np.std(self.z) + 1e-8
            bandwidth = 1.06 * s * self.n ** (-1.0 / 5)
        self.h = float(bandwidth)

    def pdf(self, z):
        z = np.atleast_1d(z)
        diff = (z[:, None] - self.z[None, :]) / self.h
        k = np.exp(-0.5 * diff ** 2) / np.sqrt(2 * np.pi)
        return np.mean(k, axis=1) / self.h


# =============================================================================
# TOUBOUL PP: FAITHFUL ALGORITHM
# =============================================================================

@dataclass
class TouboulStep:
    """One iteration of Touboul's algorithm: direction a_k and updated g^(k) info."""
    a: np.ndarray                  # projection direction (unit vector)
    divergence_value: float        # D_phi at this step
    test_statistic: float          # Corollary 3.2 statistic
    p_value: float                 # asymptotic p-value from N(0,1)
    converged: bool


class TouboulPP:
    """
    Faithful Touboul (2010) projection pursuit via phi-divergence minimization.

    Algorithm (Touboul 2010, eq 1.3):
      g^(0) = Gaussian with same mean/variance as f (eq. after 2.1)
      For k = 1, 2, ...:
        a_k = argmin_a D_phi( g^(k-1) * f_a / g_a^(k-1) , f )
        g^(k) = g^(k-1) * f_{a_k} / g_{a_k}^(k-1)
        test H_0 : D_phi(g^(k), f) = 0 via Corollary 3.2
        if accepted: stop
      end

    We parametrise directions as a(theta) on the unit sphere S^{d-1}.
    """

    def __init__(self, divergence: str | PhiDivergence = "KL",
                 max_steps: int = 5,
                 n_restarts: int = 20,
                 alpha: float = 0.05,
                 rng=None):
        self.phi = DIVERGENCES[divergence] if isinstance(divergence, str) else divergence
        self.max_steps = max_steps
        self.n_restarts = n_restarts
        self.alpha = alpha
        self.rng = rng or np.random.default_rng(0)

        # Fitted state
        self.f_kde_: Optional[KDEMultivariate] = None
        self.g_: Optional[GaussianInstrumental] = None
        self.steps_: List[TouboulStep] = []
        self.n_: int = 0
        self.d_: int = 0

    # -------------------------------------------------------------------------
    # Core quantity: empirical D_phi at direction a, given g^(k-1)
    # -------------------------------------------------------------------------
    def _empirical_divergence(self, a, X, f_a_kde, g_prev_marg, g_prev_full):
        """
        Compute D_phi_hat ( g^(k-1) * f_a / g_a^(k-1) ,  f ) evaluated via
        the integral representation with empirical f (i.e. sample-average
        against X_i drawn from f).

        Using the primal formula:
          D_phi(q, p) = E_{X~p}[ phi(q(X)/p(X)) ]
        where here p = f (empirical) and q = g^(k-1) * f_a / g_a^(k-1).

        So D_phi_hat = (1/n) sum_i phi( q(X_i) / f_hat(X_i) ).
        """
        a = np.asarray(a).flatten()
        n = len(X)
        z = X @ a                                    # projections of X onto a
        f_a_vals = f_a_kde.pdf(z)                    # f_a(a' X_i), from marg KDE
        g_a_vals = g_prev_marg(a, z)                 # g_a^(k-1)(a' X_i)
        g_full_vals = g_prev_full(X)                 # g^(k-1)(X_i)
        f_vals = self.f_kde_.pdf(X)                  # f_hat(X_i)

        q = g_full_vals * f_a_vals / _safe(g_a_vals)
        ratio = q / _safe(f_vals)
        return float(np.mean(self.phi.phi(ratio)))

    # -------------------------------------------------------------------------
    # Minimax M function (Touboul eq. after H4, Broniatowski-Keziou dual form)
    # This is needed for the rigorous estimator in Prop 3.2 (minimax)
    # -------------------------------------------------------------------------
    def _M(self, b, a, X, f_a_kde, g_prev_marg, g_prev_full):
        """
        M(b, a, x) = phi'( g(x) f_b(b'x) / (f(x) g_b(b'x)) ) * g(x) f_a(a'x)/g_a(a'x)
                     - phi^#( phi'( g(x) f_b(b'x) / (f(x) g_b(b'x)) ) )
        and we return its empirical mean P_n M(b, a).
        """
        b = np.asarray(b).flatten()
        a = np.asarray(a).flatten()
        z_b = X @ b
        z_a = X @ a
        f_b_vals = f_a_kde(b).pdf(z_b) if callable(f_a_kde) else f_a_kde.pdf(z_b)
        # For minimax we need f_a for direction b and for direction a separately.
        # We lazily reconstruct both since KDE is cheap.
        f_b_vals = KDE1D(X @ b).pdf(z_b)
        f_a_vals = KDE1D(X @ a).pdf(z_a)
        g_b_vals = g_prev_marg(b, z_b)
        g_a_vals = g_prev_marg(a, z_a)
        g_full = g_prev_full(X)
        f_full = self.f_kde_.pdf(X)

        u = g_full * f_b_vals / (_safe(f_full) * _safe(g_b_vals))
        term1 = self.phi.phi_prime(u) * g_full * f_a_vals / _safe(g_a_vals)
        term2 = self.phi.phi_star(self.phi.phi_prime(u))
        return float(np.mean(term1 - term2))

    # -------------------------------------------------------------------------
    # Direction optimization: find a_k minimizing the primal D_phi
    # (simpler than full minimax and equivalent at the optimum by Prop 3.1)
    # -------------------------------------------------------------------------
    @staticmethod
    def _unit_from_theta(theta: np.ndarray) -> np.ndarray:
        """Map free params theta to a unit vector on S^{d-1} via spherical coords.

        For d dimensions, theta has d-1 angles. Standard parameterisation:
          a_1 = cos(theta_1)
          a_2 = sin(theta_1) cos(theta_2)
          ...
          a_{d-1} = sin(theta_1)...sin(theta_{d-2}) cos(theta_{d-1})
          a_d     = sin(theta_1)...sin(theta_{d-1})
        """
        theta = np.atleast_1d(theta)
        d = len(theta) + 1
        a = np.zeros(d)
        s = 1.0
        for i in range(d - 1):
            a[i] = s * np.cos(theta[i])
            s *= np.sin(theta[i])
        a[-1] = s
        return a

    def _optimize_direction(self, X, f_a_kde_factory, g_prev_marg, g_prev_full):
        """Find a_k by multi-start minimization of empirical D_phi."""
        d = X.shape[1]
        best_a = None
        best_val = np.inf
        for _ in range(self.n_restarts):
            theta0 = self.rng.uniform(0, np.pi, size=d - 1)
            theta0[-1] = self.rng.uniform(0, 2 * np.pi)  # last angle in [0, 2pi]

            def obj(theta):
                a = self._unit_from_theta(theta)
                f_a_kde = f_a_kde_factory(a)
                return self._empirical_divergence(
                    a, X, f_a_kde, g_prev_marg, g_prev_full
                )

            try:
                res = minimize(obj, theta0, method="Nelder-Mead",
                               options={"maxiter": 200, "xatol": 1e-4})
                val = res.fun
                if val < best_val:
                    best_val = val
                    best_a = self._unit_from_theta(res.x)
            except Exception:
                continue

        return best_a, best_val

    # -------------------------------------------------------------------------
    # Factorisation test (Corollary 3.2, Touboul 2010)
    # -------------------------------------------------------------------------
    def _factorisation_test(self, a, X, f_a_kde, g_prev_marg, g_prev_full):
        """
        Corollary 3.2: under H_0 : D_phi(g^(k), f) = 0,
            sqrt(n) * Var_P(M)^{-1/2} * P_n M(a, a)  ->  N(0, 1) in law.

        Returns: (test_statistic, two-sided p-value).
        """
        n = len(X)
        # point-wise M(a, a, x_i) values
        a = np.asarray(a).flatten()
        z = X @ a
        f_a_vals = KDE1D(z).pdf(z)
        g_a_vals = g_prev_marg(a, z)
        g_full = g_prev_full(X)
        f_full = self.f_kde_.pdf(X)

        u = g_full * f_a_vals / (_safe(f_full) * _safe(g_a_vals))
        M_pointwise = self.phi.phi_prime(u) * g_full * f_a_vals / _safe(g_a_vals) \
                      - self.phi.phi_star(self.phi.phi_prime(u))

        mean_M = float(np.mean(M_pointwise))
        var_M = float(np.var(M_pointwise, ddof=1)) + 1e-12

        # Standardised statistic ~ N(0,1) under H_0
        T = np.sqrt(n) * mean_M / np.sqrt(var_M)
        # two-sided p-value
        pval = 2 * (1 - stats.norm.cdf(abs(T)))
        return T, pval

    # -------------------------------------------------------------------------
    # Main fit loop
    # -------------------------------------------------------------------------
    def fit(self, X):
        X = np.asarray(X)
        self.n_, self.d_ = X.shape

        # Estimate f by multivariate KDE
        self.f_kde_ = KDEMultivariate(X)

        # Initial instrumental g^(0) = Gaussian with same mean/variance as f
        self.g_ = GaussianInstrumental.from_data(X)

        # g^(0) marginal and full pdf as callables
        g_prev_marg = lambda a, z: self.g_.marginal_pdf_along(a, z)
        g_prev_full = lambda x: self.g_.pdf(x)

        self.steps_ = []
        # cached factory: builds KDE1D of projection onto a
        def f_a_kde_factory(a):
            return KDE1D(X @ a)

        for k in range(self.max_steps):
            # 1. Find best direction
            a_k, div_k = self._optimize_direction(
                X, f_a_kde_factory, g_prev_marg, g_prev_full
            )
            if a_k is None:
                break

            # 2. Test factorization (Corollary 3.2)
            f_a_kde = f_a_kde_factory(a_k)
            T, pval = self._factorisation_test(
                a_k, X, f_a_kde, g_prev_marg, g_prev_full
            )

            converged = pval > self.alpha
            self.steps_.append(TouboulStep(
                a=a_k, divergence_value=div_k,
                test_statistic=T, p_value=pval,
                converged=converged,
            ))

            if converged:
                break

            # 3. Update g^(k) = g^(k-1) * f_{a_k} / g_{a_k}^(k-1)
            a_k_copy = a_k.copy()
            f_ak_kde = f_a_kde
            prev_marg = g_prev_marg
            prev_full = g_prev_full

            def new_full(x, a=a_k_copy, prev=prev_full, pm=prev_marg,
                         fka=f_ak_kde):
                z = x @ a
                return prev(x) * fka.pdf(z) / _safe(pm(a, z))

            def new_marg(b, z, a=a_k_copy, prev=prev_marg, fka=f_ak_kde):
                # Approximation: marginal along b of the updated density.
                # This is intractable in closed form; we use the same g^(0)
                # marginal as approximation since the update is along a_k
                # specifically. For directions orthogonal to a_k, the
                # marginal is unchanged.
                # For theoretical fidelity we track the full product form.
                return prev(b, z)  # approximation

            g_prev_full = new_full
            g_prev_marg = new_marg  # approximation noted above

        return self

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------
    def summary(self):
        print(f"Touboul PP with {self.phi.name} divergence")
        print(f"  n = {self.n_},  d = {self.d_}")
        print(f"  Steps taken: {len(self.steps_)}")
        for k, s in enumerate(self.steps_, 1):
            print(f"  Step {k}: a = {np.round(s.a, 3)},  "
                  f"D_phi = {s.divergence_value:.5f},  "
                  f"T = {s.test_statistic:.3f},  "
                  f"p = {s.p_value:.4f}  "
                  f"{'[accept H0, stop]' if s.converged else '[reject H0, continue]'}")


# =============================================================================
# SELF-TEST: reproduce Touboul 2010 simulation setup
# =============================================================================

def _selftest():
    """
    Reproduction check inspired by Touboul 2010 Simulation 4.3-4.4:

    "We generate in R^2 a sample of size 500 with density
         f(x1, x2) = phi(x1) * h(x2)
     where phi is standard Gaussian and h is non-Gaussian (Gumbel or Exp).
     Then g is Gaussian with same mean/variance as f.
     Theoretical answer: a_1 = (0, 1), i.e., the non-Gaussian direction
     is recovered. Testing H_0: a_1 = (0, 1) versus (1, -1) should accept."
    """
    rng = np.random.default_rng(42)
    n = 500

    # Simulation 4 of Touboul 2010: x1 standard Gaussian, x2 Gumbel(0,1)
    x1 = rng.standard_normal(n)
    x2 = rng.gumbel(loc=0.0, scale=1.0, size=n)
    X = np.column_stack([x1, x2])

    print("\n" + "=" * 70)
    print("TOUBOUL 2010 SELF-TEST")
    print("=" * 70)
    print("Setup: f(x1, x2) = N(0,1)(x1) * Gumbel(0,1)(x2)")
    print(f"       n = {n}, d = 2")
    print("Expected: a_1 ~ (0, +/-1) (the non-Gaussian direction)")
    print()

    for div in ["KL", "Hellinger", "ChiSquared"]:
        print(f"--- Divergence: {div} ---")
        pp = TouboulPP(divergence=div, max_steps=2,
                       n_restarts=15, alpha=0.05,
                       rng=np.random.default_rng(0))
        pp.fit(X)
        pp.summary()
        print()

    return pp


if __name__ == "__main__":
    _selftest()
