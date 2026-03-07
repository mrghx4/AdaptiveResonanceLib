"""Bayesian ART :cite:`vigdor2007bayesian`."""

import numpy as np
from typing import Literal, Optional, Callable

from sklearn.utils.validation import check_is_fitted

from artlib.elementary.BayesianART import BayesianART as pyBayesianART
from artlib.optimized.backends.cpp.cppBayesianART import FitBayesianART, PredictBayesianART


class BayesianART(pyBayesianART):
    """BayesianART for Clustering. optimized with C++"""

    def __init__(self, rho: float, cov_init: np.ndarray):
        cov_init = np.asarray(cov_init, dtype=float)
        if cov_init.ndim != 2 or cov_init.shape[0] != cov_init.shape[1]:
            raise ValueError("'cov_init' must be a square 2-D array.")
        super().__init__(rho=rho, cov_init=cov_init)

    def _synchronize_cpp_results(
        self,
        labels_out: np.ndarray,
        weights_arrays: list[np.ndarray],
        incremental: bool = False,
    ):
        if not incremental:
            self.labels_ = np.array((), dtype=int)
            self.weight_sample_counter_ = []

        self.labels_ = np.concatenate([self.labels_, labels_out.astype(int)])

        new_counts = np.bincount(labels_out, minlength=len(weights_arrays))
        if len(self.weight_sample_counter_) < len(new_counts):
            self.weight_sample_counter_.extend(
                [0] * (len(new_counts) - len(self.weight_sample_counter_))
            )
        for k, c in enumerate(new_counts):
            self.weight_sample_counter_[k] += int(c)

        self.W = [w for w in weights_arrays]

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        match_reset_func: Optional[Callable] = None,
        max_iter=1,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 0.0,
        verbose: bool = False,
        leave_progress_bar: bool = True,
    ):
        X_ = np.ascontiguousarray(X, dtype=np.float64)
        self.validate_data(X_)
        self.W = []
        self.labels_ = np.zeros((X_.shape[0],), dtype=int)

        la, W = FitBayesianART(
            X_,
            rho=self.params["rho"],
            cov_init=np.asarray(self.params["cov_init"], dtype=np.float64),
            weights=None,
        )
        self._synchronize_cpp_results(la, W)
        self.is_fitted_ = True
        return self

    def partial_fit(
        self,
        X: np.ndarray,
        match_reset_func: Optional[Callable] = None,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 0.0,
    ):
        X_ = np.ascontiguousarray(X, dtype=np.float64)
        self.validate_data(X_)

        if not hasattr(self, "labels_"):
            self.labels_ = np.zeros((X_.shape[0],), dtype=int)
            existing_W = None
        else:
            existing_W = np.ascontiguousarray(self.W, dtype=float)

        la, W = FitBayesianART(
            X_,
            rho=self.params["rho"],
            cov_init=np.asarray(self.params["cov_init"], dtype=np.float64),
            weights=existing_W,
        )
        self._synchronize_cpp_results(la, W, incremental=True)
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray, clip: bool = False) -> np.ndarray:
        check_is_fitted(self)
        X_ = np.ascontiguousarray(X, dtype=np.float64)
        if clip:
            X_ = np.clip(X_, self.d_min_, self.d_max_)
        self.validate_data(X_)

        W = np.ascontiguousarray(self.W, dtype=float)
        return PredictBayesianART(
            X_,
            rho=self.params["rho"],
            cov_init=np.asarray(self.params["cov_init"], dtype=np.float64),
            weights=W,
        )
