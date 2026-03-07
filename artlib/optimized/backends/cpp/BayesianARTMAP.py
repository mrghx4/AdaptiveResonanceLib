"""Bayesian ARTMAP (C++ backend)."""

import numpy as np
from typing import Literal, Tuple

from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

from artlib.elementary.BayesianART import BayesianART
from artlib.supervised.SimpleARTMAP import SimpleARTMAP
from artlib.optimized.backends.cpp.cppBayesianARTMAP import (
    FitBayesianARTMAP,
    PredictBayesianARTMAP,
)


class BayesianARTMAP(SimpleARTMAP):
    """BayesianARTMAP for Classification. optimized with C++"""

    def __init__(self, rho: float, cov_init: np.ndarray):
        cov_init = np.asarray(cov_init, dtype=float)
        if cov_init.ndim != 2 or cov_init.shape[0] != cov_init.shape[1]:
            raise ValueError("'cov_init' must be a square 2-D array.")

        module_a = BayesianART(rho=rho, cov_init=cov_init)
        super().__init__(module_a)
        self._cov_init = cov_init

    def _synchronize_cpp_results(
        self,
        labels_a_out: np.ndarray,
        weights_arrays: list[np.ndarray],
        cluster_labels_out: np.ndarray,
        incremental: bool = False,
    ):
        if not incremental:
            self.map: dict[int, int] = {}
            self.module_a.labels_ = np.array((), dtype=int)
            self.module_a.weight_sample_counter_ = []

        self.module_a.labels_ = np.concatenate(
            [self.module_a.labels_, labels_a_out.astype(int)]
        )

        new_counts = np.bincount(labels_a_out, minlength=len(weights_arrays))
        if len(self.module_a.weight_sample_counter_) < len(new_counts):
            self.module_a.weight_sample_counter_.extend(
                [0] * (len(new_counts) - len(self.module_a.weight_sample_counter_))
            )
        for k, c in enumerate(new_counts):
            self.module_a.weight_sample_counter_[k] += int(c)

        self.module_a.W = [w for w in weights_arrays]

        for c_a, c_b in enumerate(cluster_labels_out):
            if c_a in self.map:
                assert self.map[c_a] == c_b, "Incremental fit changed cluster map."
            else:
                self.map[c_a] = int(c_b)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_iter: int = 1,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 1e-10,
        verbose: bool = False,
        leave_progress_bar: bool = True,
    ):
        X_ = np.ascontiguousarray(X, dtype=np.float64)
        y_ = np.ascontiguousarray(y, dtype=np.int32)
        SimpleARTMAP.validate_data(self, X_, y_)
        self.classes_ = unique_labels(y_)
        self.labels_ = y_
        self.module_a.W = []
        self.module_a.labels_ = np.zeros((X_.shape[0],), dtype=int)

        la, W, cl = FitBayesianARTMAP(
            X_,
            y_,
            rho=self.module_a.params["rho"],
            cov_init=self._cov_init,
            MT=match_tracking,
            epsilon=epsilon,
            weights=None,
            cluster_labels=None,
        )
        self._synchronize_cpp_results(la, W, cl)
        self.module_a.is_fitted_ = True
        return self

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 1e-10,
    ):
        X_ = np.ascontiguousarray(X, dtype=np.float64)
        y_ = np.ascontiguousarray(y, dtype=np.int32)
        SimpleARTMAP.validate_data(self, X_, y_)

        if not hasattr(self, "labels_"):
            self.labels_ = y_
            existing_W = None
            existing_map = None
        else:
            j = len(self.labels_)
            self.labels_ = np.pad(self.labels_, (0, len(y_)))
            self.labels_[j:] = y_
            existing_W = np.ascontiguousarray(self.module_a.W, dtype=float)
            existing_map = np.ascontiguousarray(
                [self.map[c] for c in range(self.module_a.n_clusters)]
            )

        la, W, cl = FitBayesianARTMAP(
            X_,
            y_,
            rho=self.module_a.params["rho"],
            cov_init=self._cov_init,
            MT=match_tracking,
            epsilon=epsilon,
            weights=existing_W,
            cluster_labels=existing_map,
        )
        self._synchronize_cpp_results(la, W, cl, incremental=True)
        self.module_a.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray, clip: bool = False) -> np.ndarray:
        check_is_fitted(self)
        X_ = np.ascontiguousarray(X, dtype=np.float64)
        if clip:
            X_ = np.clip(X_, self.module_a.d_min_, self.module_a.d_max_)
        self.module_a.validate_data(X_)
        self.module_a.check_dimensions(X_)

        W = np.ascontiguousarray(self.module_a.W, dtype=float)
        cl = np.ascontiguousarray([self.map[c] for c in range(self.module_a.n_clusters)])
        _, y_b = PredictBayesianARTMAP(
            X_,
            rho=self.module_a.params["rho"],
            cov_init=self._cov_init,
            MT="",
            epsilon=0.0,
            weights=W,
            cluster_labels=cl,
        )
        return y_b

    def predict_ab(self, X: np.ndarray, clip: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self)
        X_ = np.ascontiguousarray(X, dtype=np.float64)
        if clip:
            X_ = np.clip(X_, self.module_a.d_min_, self.module_a.d_max_)
        self.module_a.validate_data(X_)
        self.module_a.check_dimensions(X_)

        W = np.ascontiguousarray(self.module_a.W, dtype=float)
        cl = np.ascontiguousarray([self.map[c] for c in range(self.module_a.n_clusters)])
        y_a, y_b = PredictBayesianARTMAP(
            X_,
            rho=self.module_a.params["rho"],
            cov_init=self._cov_init,
            MT="",
            epsilon=0.0,
            weights=W,
            cluster_labels=cl,
        )
        return y_a, y_b
