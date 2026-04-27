"""
Fast GMM-based substitute estimators for the dimension-scaling experiment.
Author : Touboul Jacques
Because the controlled design is a Gaussian mixture, fitting a GMM and using
responsibilities is equivalent to (or better than) running projection pursuit
on the same design. This gives us a fair upper bound on what PP-NN could
achieve with a perfect optimiser — letting us isolate the dimension-scaling
question from the optimiser-quality question.

Exposes sklearn-compatible fit/predict:
  - GMMkNN  : analog of PP-kNN (component-conditioned k-NN)
  - GMMHard : analog of PP-NN Hard (predict via highest-responsibility component)
"""
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class GMMHard(BaseEstimator, ClassifierMixin):
    """Hard-assign each query to GMM component, predict via majority label."""
    def __init__(self, K=3, covariance_type='full', pca_dim=None):
        self.K = K
        self.covariance_type = covariance_type
        self.pca_dim = pca_dim   # if not None, project to pca_dim first

    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        if self.pca_dim is not None and self.pca_dim < X.shape[1]:
            self.pca_ = PCA(n_components=self.pca_dim, random_state=0).fit(Xs)
            Xs = self.pca_.transform(Xs)
        else:
            self.pca_ = None
        self.gmm_ = GaussianMixture(n_components=self.K,
                                    covariance_type=self.covariance_type,
                                    random_state=0, n_init=3, reg_covar=1e-4)
        self.gmm_.fit(Xs)
        comp_train = self.gmm_.predict(Xs)
        self.component_label_ = np.zeros(self.K, dtype=int)
        for k in range(self.K):
            mask = comp_train == k
            if mask.any():
                vals, cnts = np.unique(y[mask], return_counts=True)
                self.component_label_[k] = vals[np.argmax(cnts)]
        self.classes_ = np.unique(y)
        return self

    def _transform(self, X):
        X = np.asarray(X)
        Xs = self.scaler_.transform(X)
        if self.pca_ is not None:
            Xs = self.pca_.transform(Xs)
        return Xs

    def predict(self, X):
        comp = self.gmm_.predict(self._transform(X))
        return self.component_label_[comp]


class GMMkNN(BaseEstimator, ClassifierMixin):
    """k-NN restricted to the GMM component assigned to the query."""
    def __init__(self, K=3, k=15, covariance_type='full', pca_dim=None):
        self.K = K
        self.k = k
        self.covariance_type = covariance_type
        self.pca_dim = pca_dim

    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        if self.pca_dim is not None and self.pca_dim < X.shape[1]:
            self.pca_ = PCA(n_components=self.pca_dim, random_state=0).fit(Xs)
            Xs_proj = self.pca_.transform(Xs)
        else:
            self.pca_ = None
            Xs_proj = Xs
        self.gmm_ = GaussianMixture(n_components=self.K,
                                    covariance_type=self.covariance_type,
                                    random_state=0, n_init=3, reg_covar=1e-4)
        self.gmm_.fit(Xs_proj)
        comp_train = self.gmm_.predict(Xs_proj)
        # Note: k-NN is fit in the ORIGINAL scaled space (not PCA-projected),
        # so the gating uses PCA+GMM but within-component prediction uses full d.
        # This is the natural PP-NN analog.
        self.knn_by_comp_ = {}
        for k in range(self.K):
            mask = comp_train == k
            if mask.sum() >= self.k:
                knn = KNeighborsClassifier(n_neighbors=min(self.k, mask.sum()-1))
                knn.fit(Xs[mask], y[mask])
                self.knn_by_comp_[k] = knn
            elif mask.sum() >= 1:
                vals, cnts = np.unique(y[mask], return_counts=True)
                self.knn_by_comp_[k] = ('majority', vals[np.argmax(cnts)])
        self.global_knn_ = KNeighborsClassifier(n_neighbors=self.k).fit(Xs, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        Xs = self.scaler_.transform(X)
        if self.pca_ is not None:
            Xs_proj = self.pca_.transform(Xs)
        else:
            Xs_proj = Xs
        comp = self.gmm_.predict(Xs_proj)
        out = np.zeros(len(X), dtype=int)
        for k in range(self.K):
            mask = comp == k
            if not mask.any():
                continue
            model_k = self.knn_by_comp_.get(k, self.global_knn_)
            if isinstance(model_k, tuple):
                out[mask] = model_k[1]
            else:
                out[mask] = model_k.predict(Xs[mask])
        return out
