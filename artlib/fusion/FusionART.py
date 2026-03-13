"""Fusion ART :cite:`tan2007intelligence`."""
# Tan, A.-H., Carpenter, G. A., & Grossberg, S. (2007).
# Intelligence Through Interaction: Towards a Unified Theory for Learning.
# In D. Liu, S. Fei, Z.-G. Hou, H. Zhang, & C. Sun (Eds.),
# Advances in Neural Networks – ISNN 2007 (pp. 1094–1103).
# Berlin, Heidelberg: Springer Berlin Heidelberg.
# doi:10.1007/ 978-3-540-72383-7_128.

import numpy as np
from typing import Optional, Union, Callable, List, Literal, Tuple, Dict, Sequence
from copy import deepcopy
from artlib.common.BaseART import BaseART
from sklearn.utils.validation import check_is_fitted
import operator

try:
    from artlib.optimized.backends.cpp.cppFusionUtils import (
        ArgmaxWeightedActivations,
        ArgmaxWeightedChannelActivations,
        ExtractPresentChannels,
        JoinChannelsWithFill,
    )
except ImportError:  # pragma: no cover - optional acceleration module
    ArgmaxWeightedActivations = None
    ArgmaxWeightedChannelActivations = None
    ExtractPresentChannels = None
    JoinChannelsWithFill = None

try:
    from artlib.optimized.backends.cpp.cppSimpleARTMAP import GatherClusterCenters
except ImportError:  # pragma: no cover - optional acceleration module
    GatherClusterCenters = None


def get_channel_position_tuples(
    channel_dims: List[int],
) -> List[Tuple[int, int]]:
    """Generate the start and end positions for each channel in the input data.

    Parameters
    ----------
    channel_dims : list of int
        A list representing the number of dimensions for each channel.

    Returns
    -------
    list of tuple of int
        A list of tuples where each tuple represents the start and end index for a
        channel.

    """
    positions = []
    start = 0
    for length in channel_dims:
        end = start + length
        positions.append((start, end))
        start = end
    return positions


class FusionART(BaseART):
    """Fusion ART for Data Fusion and Regression.

    This module implements Fusion ART as first described in:
    :cite:`tan2007intelligence`.

    .. # Tan, A.-H., Carpenter, G. A., & Grossberg, S. (2007).
    .. # Intelligence Through Interaction: Towards a Unified Theory for Learning.
    .. # In D. Liu, S. Fei, Z.-G. Hou, H. Zhang, & C. Sun (Eds.),
    .. # Advances in Neural Networks – ISNN 2007 (pp. 1094–1103).
    .. # Berlin, Heidelberg: Springer Berlin Heidelberg.
    .. # doi:10.1007/ 978-3-540-72383-7_128.

    Fusion ART accepts an arbitrary number of ART modules, each assigned a different
    data channel. The activation and match functions for all ART modules are then fused
    such that all modules must be simultaneously active and resonant in order for a
    match to occur. This provides fine-grain control when clustering multi-channel or
    multi-modal data and allows for different geometries of clusters to be used for
    each channel. Fusion ART also allows for fitting regression models and specific
    functions have been implemented to allow this.

    """

    def __init__(
        self,
        modules: List[BaseART],
        gamma_values: Union[List[float], np.ndarray],
        channel_dims: Union[List[int], np.ndarray],
    ):
        """Initialize the FusionART instance.

        Parameters
        ----------
        modules : List[BaseART]
            A list of ART modules corresponding to each data channel.
        gamma_values : Union[List[float], np.ndarray]
            The activation ratio for each channel.
        channel_dims : Union[List[int], np.ndarray]
            The number of dimensions for each channel.

        """
        assert len(modules) == len(gamma_values) == len(channel_dims)
        params = {"gamma_values": gamma_values}
        super().__init__(params)
        self.modules = modules
        self.n = len(self.modules)
        self.channel_dims = channel_dims
        self._channel_indices = get_channel_position_tuples(self.channel_dims)
        self._gamma_values = np.asarray(self.params["gamma_values"], dtype=float)
        self._cpp_fusion_argmax_threshold = 64
        self._channel_centers_cache: dict[int, np.ndarray] = {}
        self.dim_ = sum(channel_dims)

    def _invalidate_channel_centers_cache(self):
        self._channel_centers_cache.clear()

    def get_params(self, deep: bool = True) -> Dict:
        """Get the parameters of the FusionART model.

        Parameters
        ----------
        deep : bool, optional
            If True, will return parameters for this class and the contained sub-objects
            that are estimators (default is True).

        Returns
        -------
        dict
            Parameter names mapped to their values.

        """
        out = self.params
        for i, module in enumerate(self.modules):
            deep_items = module.get_params().items()
            out.update((f"module_{i}" + "__" + k, val) for k, val in deep_items)
            out[f"module_{i}"] = module
        return out

    @property
    def n_clusters(self) -> int:
        """Return the number of clusters in the first ART module.

        Returns
        -------
        int
            The number of clusters.

        """
        return self.modules[0].n_clusters

    @property
    def W(self):
        """Get the weights of all modules as a single array.

        Returns
        -------
        list
            Concatenated weights of all channels from the ART modules.

        """
        W = [
            [self.modules[k].W[i] for k in range(self.n)]
            for i in range(self.modules[0].n_clusters)
        ]
        return W

    @W.setter
    def W(self, new_W: list):
        """Set the weights for each module from a list of cluster-wise composite
        weights.

        Parameters
        ----------
        new_W : list
            List of cluster weights. Each element is a list of weights, one per module.

        """
        for k in range(self.n):
            self.modules[k].W = [new_W[i][k] for i in range(len(new_W))]
        self._invalidate_channel_centers_cache()

    @staticmethod
    def validate_params(params: Dict):
        """Validate clustering parameters.

        Parameters
        ----------
        params : dict
            The parameters for the FusionART model.

        """
        assert "gamma_values" in params
        assert all([1.0 >= g >= 0.0 for g in params["gamma_values"]])
        assert sum(params["gamma_values"]) == 1.0
        assert isinstance(params["gamma_values"], (np.ndarray, list))

    def validate_data(self, X: np.ndarray):
        """Validate the input data for clustering.

        Parameters
        ----------
        X : np.ndarray
            The input dataset.

        """
        self.check_dimensions(X)
        for k in range(self.n):
            X_k = X[:, self._channel_indices[k][0] : self._channel_indices[k][1]]
            self.modules[k].validate_data(X_k)

    def _n_categories(self) -> int:
        return self.modules[0].n_clusters

    def _cluster_weight(self, c_idx: int) -> list:
        return [self.modules[k].W[c_idx] for k in range(self.n)]

    def _split_sample_channels(self, i: np.ndarray) -> list[np.ndarray]:
        idxs = self._channel_indices
        return [i[s:e] for (s, e) in idxs]

    def _category_choice_idx(
        self,
        i: np.ndarray,
        c_idx: int,
        skip_channels: Optional[List[int]] = None,
        i_parts: Optional[list[np.ndarray]] = None,
    ) -> Tuple[float, Dict]:
        skip = self._normalize_skip_channels(skip_channels)
        modules = self.modules
        parts = i_parts if i_parts is not None else self._split_sample_channels(i)
        activation = 0.0
        caches: Dict[int, Dict] = {}
        for k in range(self.n):
            if k in skip:
                activation += self._gamma_values[k]
                caches[k] = {}
                continue
            a_k, c_k = modules[k].category_choice(
                parts[k],
                modules[k].W[c_idx],
                modules[k].params,
            )
            activation += a_k * self._gamma_values[k]
            caches[k] = c_k
        return float(activation), caches

    def _category_choice_value_idx(
        self,
        i: np.ndarray,
        c_idx: int,
        skip: set[int],
        i_parts: Optional[list[np.ndarray]] = None,
    ) -> float:
        modules = self.modules
        parts = i_parts if i_parts is not None else self._split_sample_channels(i)
        activation = 0.0
        for k in range(self.n):
            if k in skip:
                activation += self._gamma_values[k]
                continue
            a_k, _ = modules[k].category_choice(
                parts[k],
                modules[k].W[c_idx],
                modules[k].params,
            )
            activation += a_k * self._gamma_values[k]
        return float(activation)

    def _match_criterion_bin_idx(
        self,
        i: np.ndarray,
        c_idx: int,
        cache: Dict,
        op: Callable = operator.ge,
        skip_channels: Optional[List[int]] = None,
        i_parts: Optional[list[np.ndarray]] = None,
    ) -> Tuple[bool, Dict]:
        skip = self._normalize_skip_channels(skip_channels)
        modules = self.modules
        parts = i_parts if i_parts is not None else self._split_sample_channels(i)
        caches: Dict[int, Dict] = {}
        all_match = True
        for k in range(self.n):
            if k in skip:
                caches[k] = {"match_criterion": np.inf}
                continue
            mb_k, c_k = modules[k].match_criterion_bin(
                parts[k],
                modules[k].W[c_idx],
                modules[k].params,
                cache[k],
                op,
            )
            caches[k] = c_k
            if not mb_k:
                all_match = False
        return all_match, caches

    def _update_idx(
        self,
        i: np.ndarray,
        c_idx: int,
        cache: Dict,
        i_parts: Optional[list[np.ndarray]] = None,
    ) -> list:
        modules = self.modules
        parts = i_parts if i_parts is not None else self._split_sample_channels(i)
        return [
            modules[k].update(
                parts[k],
                modules[k].W[c_idx],
                modules[k].params,
                cache[k],
            )
            for k in range(self.n)
        ]

    def _normalize_skip_channels(self, skip_channels: Optional[List[int]]) -> set[int]:
        if not skip_channels:
            return set()
        normalized: set[int] = set()
        for k in skip_channels:
            idx = self.n + k if k < 0 else k
            if idx < 0 or idx >= self.n:
                raise ValueError(
                    f"skip channel index {k} (normalized to {idx}) is out of range "
                    f"for {self.n} channels"
                )
            normalized.add(idx)
        return normalized

    def check_dimensions(self, X: np.ndarray):
        """Ensure that the input data has the correct dimensions.

        Parameters
        ----------
        X : np.ndarray
            The input dataset.

        """
        assert X.shape[1] == self.dim_, "Invalid data shape"

    def prepare_data(
        self, channel_data: List[np.ndarray], skip_channels: Optional[List[int]] = None
    ) -> np.ndarray:
        """Prepare the input data by processing each channel's data through its
        respective ART module.

        Parameters
        ----------
        channel_data : list of np.ndarray
            List of arrays, one for each channel.
        skip_channels : list of int, optional
            Channels to be skipped (default is None).

        Returns
        -------
        np.ndarray
            Processed and concatenated data.

        """
        skip_channels = sorted(self._normalize_skip_channels(skip_channels))
        prepared_channel_data = [
            self.modules[i].prepare_data(channel_data[i])
            for i in range(self.n)
            if i not in skip_channels
        ]

        return self.join_channel_data(
            prepared_channel_data, skip_channels=skip_channels
        )

    def restore_data(
        self, X: np.ndarray, skip_channels: Optional[List[int]] = None
    ) -> List[np.ndarray]:
        """Restore data to its original state before preparation.

        Parameters
        ----------
        X : np.ndarray
            The prepared data.
        skip_channels : list of int, optional
            Channels to be skipped (default is None).
        Returns
        -------
        np.ndarray
            Restored data for each channel.

        """
        skip_channels = sorted(self._normalize_skip_channels(skip_channels))
        channel_data = self.split_channel_data(X, skip_channels=skip_channels)
        restored_channel_data = [
            self.modules[i].restore_data(channel_data[i])
            for i in range(self.n)
            if i not in skip_channels
        ]
        return restored_channel_data

    def category_choice(
        self,
        i: np.ndarray,
        w: list,
        params: Dict,
        skip_channels: Optional[List[int]] = None,
    ) -> Tuple[float, Optional[Dict]]:
        """Get the activation of the cluster.

        Parameters
        ----------
        i : np.ndarray
            The data sample.
        w : np.ndarray
            The cluster weight information.
        params : dict
            Parameters for the ART algorithm.
        skip_channels : list of int, optional
            Channels to be skipped (default is []).

        Returns
        -------
        tuple
            Cluster activation and cache for further processing.

        """
        skip = self._normalize_skip_channels(skip_channels)
        modules = self.modules
        idxs = self._channel_indices
        activation = 0.0
        caches = {}
        for k in range(self.n):
            if k in skip:
                activation += self._gamma_values[k]
                caches[k] = {}
                continue
            a_k, c_k = modules[k].category_choice(
                i[idxs[k][0] : idxs[k][1]],
                w[k],
                modules[k].params,
            )
            activation += a_k * self._gamma_values[k]
            caches[k] = c_k
        cache = caches
        return activation, cache

    def match_criterion(
        self,
        i: np.ndarray,
        w: list,
        params: Dict,
        cache: Optional[Dict] = None,
        skip_channels: Optional[List[int]] = None,
    ) -> Tuple[float, Optional[Dict]]:
        """Get the match criterion for the cluster.

        Parameters
        ----------
        i : np.ndarray
            The data sample.
        w : np.ndarray
            The cluster weight information.
        params : dict
            Parameters for the ART algorithm.
        cache : dict, optional
            Cache for previous calculations (default is None).
        skip_channels : list of int, optional
            Channels to be skipped (default is []).

        Returns
        -------
        tuple
            max match_criterion across channels and the updated cache.

        """
        if cache is None:
            raise ValueError("No cache provided")
        skip = self._normalize_skip_channels(skip_channels)
        modules = self.modules
        idxs = self._channel_indices
        matches = np.empty((self.n,), dtype=float)
        caches = {}
        for k in range(self.n):
            if k in skip:
                matches[k] = np.nan
                caches[k] = {"match_criterion": np.inf}
                continue
            m_k, c_k = modules[k].match_criterion(
                i[idxs[k][0] : idxs[k][1]],
                w[k],
                modules[k].params,
                cache[k],
            )
            matches[k] = m_k
            caches[k] = c_k
        return float(np.nanmax(matches)), caches

    def match_criterion_bin(
        self,
        i: np.ndarray,
        w: list,
        params: Dict,
        cache: Optional[Dict] = None,
        op: Callable = operator.ge,
        skip_channels: Optional[List[int]] = None,
    ) -> Tuple[bool, Dict]:
        """Get the binary match criterion for the cluster.

        Parameters
        ----------
        i : np.ndarray
            The data sample.
        w : np.ndarray
            The cluster weight information.
        params : dict
            Parameters for the ART algorithm.
        cache : dict, optional
            Cache for previous calculations (default is None).
        op : Callable, optional
            Operator for comparison (default is operator.ge).
        skip_channels : list of int, optional
            Channels to be skipped (default is []).

        Returns
        -------
        tuple
            Binary match criterion and cache for further processing.

        """
        if cache is None:
            raise ValueError("No cache provided")
        skip = self._normalize_skip_channels(skip_channels)
        modules = self.modules
        idxs = self._channel_indices
        caches = {}
        all_match = True
        for k in range(self.n):
            if k in skip:
                caches[k] = {"match_criterion": np.inf}
                continue
            mb_k, c_k = modules[k].match_criterion_bin(
                i[idxs[k][0] : idxs[k][1]],
                w[k],
                modules[k].params,
                cache[k],
                op,
            )
            caches[k] = c_k
            if not mb_k:
                all_match = False
        return all_match, caches

    def _match_tracking(
        self,
        cache: Union[List[Dict], Dict],
        epsilon: float,
        params: Union[List[Dict], Dict],
        method: Literal["MT+", "MT-", "MT0", "MT1", "MT~"],
    ) -> bool:
        """Perform match tracking for all channels using the specified method.

        Parameters
        ----------
        cache : list of dict
            Cached match criterion values for each channel.
        epsilon : float
            Small adjustment factor for match tracking.
        params : list of dict
            Parameters for each channel module.
        method : Literal["MT+", "MT-", "MT0", "MT1", "MT~"]
            Match tracking method to apply.

        Returns
        -------
        bool
            Whether to continue searching for a match across all channels.

        """
        for i in range(self.n):
            if cache[i]["match_criterion_bin"]:
                keep_searching_i = self.modules[i]._match_tracking(
                    cache[i], epsilon, params[i], method
                )
                if not keep_searching_i:
                    return False
        return True

    def _set_params(self, new_params: Sequence[Dict]):
        """Set the parameters for each module in FusionART.

        Parameters
        ----------
        new_params : list of dict
            A list of parameters for each module.

        """
        for i in range(self.n):
            self.modules[i].params = new_params[i]

    def _deep_copy_params(self) -> list[Dict]:
        """Create a deep copy of the parameters for each module.

        Returns
        -------
        dict
            A dictionary with module indices as keys and their deep-copied parameters
            as values.

        """
        return [deepcopy(module.params) for module in self.modules]

    def step_fit(
        self,
        x: np.ndarray,
        match_reset_func: Optional[Callable] = None,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 0.0,
    ) -> int:
        """Fit the model to a single sample.

        Parameters
        ----------
        x : np.ndarray
            Data sample.
        match_reset_func : callable, optional
            A callable that influences cluster creation.
        match_tracking : {"MT+", "MT-", "MT0", "MT1", "MT~"}, default="MT+"
            Method for resetting match criterion.
        epsilon : float, default=0.0
            Epsilon value used for adjusting match criterion.

        Returns
        -------
        int
            Cluster label of the input sample.

        """
        self.sample_counter_ += 1
        base_params = self._deep_copy_params()
        mt_operator = self._match_tracking_operator(match_tracking)
        n_categories = self._n_categories()
        x_parts = self._split_sample_channels(x)
        if n_categories == 0:
            w_new = self.new_weight(x, self.params)
            self.add_weight(w_new)
            return 0
        else:
            T_values = np.full((n_categories,), np.nan, dtype=float)
            T_cache: List[Optional[Dict]] = [None] * n_categories
            if match_tracking in ["MT~"] and match_reset_func is not None:
                for c_ in range(n_categories):
                    w = self._cluster_weight(c_)
                    if match_reset_func(x, w, c_, params=self.params, cache=None):
                        t, c = self._category_choice_idx(x, c_, i_parts=x_parts)
                        T_values[c_] = t
                        T_cache[c_] = c
            else:
                for c_ in range(n_categories):
                    t, c = self._category_choice_idx(x, c_, i_parts=x_parts)
                    T_values[c_] = t
                    T_cache[c_] = c

            # Sort candidates once:
            # primary = -T (descending T), secondary = index (ascending)
            valid = ~np.isnan(T_values)
            if np.any(valid):
                idx = np.arange(T_values.shape[0])[valid]
                T_valid = T_values[valid]
                order = idx[np.lexsort((idx, -T_valid))]  # last key is primary
            else:
                order = np.array([], dtype=int)
            params = [module.params for module in self.modules]

            for c_idx in order:
                c_ = int(c_idx)
                cache = T_cache[c_]
                assert cache is not None
                m, cache = self._match_criterion_bin_idx(
                    x, c_, cache=cache, op=mt_operator, i_parts=x_parts
                )

                if match_tracking in ["MT~"] and match_reset_func is not None:
                    no_match_reset = True
                else:
                    w = self._cluster_weight(c_)
                    no_match_reset = match_reset_func is None or match_reset_func(
                        x, w, c_, params=self.params, cache=cache
                    )

                if m and no_match_reset:
                    self.set_weight(c_, self._update_idx(x, c_, cache, i_parts=x_parts))
                    self._set_params(base_params)
                    return c_
                keep_searching = self._match_tracking(
                    cache, epsilon, params, match_tracking
                )
                if not keep_searching:
                    break

            c_new = n_categories
            w_new = self.new_weight(x, self.params)
            self.add_weight(w_new)
            self._set_params(base_params)
            return c_new

    def partial_fit(
        self,
        X: np.ndarray,
        match_reset_func: Optional[Callable] = None,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 0.0,
    ):
        """Iteratively fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            Input dataset.
        match_reset_func : callable, optional
            Function to reset the match criteria based on external factors.
        match_tracking : Literal["MT+", "MT-", "MT0", "MT1", "MT~"], optional
            Method for resetting match criteria (default is "MT+").
        epsilon : float, optional
            Value to adjust the vigilance parameter (default is 0.0).

        """
        self.validate_data(X)
        self.check_dimensions(X)
        self.is_fitted_ = True

        if not hasattr(self.modules[0], "W"):
            self.W: List[np.ndarray] = []
            self.labels_ = np.zeros((X.shape[0],), dtype=int)
            j = 0
        else:
            j = len(self.labels_)
            self.labels_ = np.pad(self.labels_, [(0, X.shape[0])], mode="constant")
        for i, x in enumerate(X):
            c = self.step_fit(
                x,
                match_reset_func=match_reset_func,
                match_tracking=match_tracking,
                epsilon=epsilon,
            )
            self.labels_[i + j] = c
        return self

    def step_pred(self, x, skip_channels: Optional[List[int]] = None) -> int:
        """Predict the label for a single sample.

        Parameters
        ----------
        x : np.ndarray
            Input sample.
        skip_channels : list of int, optional
            Channels to skip (default is []).

        Returns
        -------
        int
            Predicted cluster label for the input sample.

        """
        skip = self._normalize_skip_channels(skip_channels)
        return self._step_pred_with_skip_set(x, skip)

    def _step_pred_with_skip_set(self, x: np.ndarray, skip: set[int]) -> int:
        n_categories = self._n_categories()
        assert n_categories > 0, "ART module is not fit."
        x_parts = self._split_sample_channels(x)
        if (
            ArgmaxWeightedChannelActivations is not None
            and n_categories >= self._cpp_fusion_argmax_threshold
        ):
            channel_activ = np.empty((self.n, n_categories), dtype=np.float64)
            for k in range(self.n):
                if k in skip:
                    channel_activ[k, :] = 1.0
                    continue
                x_k = x_parts[k]
                mod = self.modules[k]
                for c_ in range(n_categories):
                    a_k, _ = mod.category_choice(x_k, mod.W[c_], mod.params)
                    channel_activ[k, c_] = a_k
            skip_mask = np.zeros((self.n,), dtype=np.uint8)
            if skip:
                skip_mask[list(skip)] = 1
            return int(
                ArgmaxWeightedChannelActivations(
                    channel_activ,
                    np.ascontiguousarray(self._gamma_values, dtype=np.float64),
                    skip_mask,
                )
            )

        best_idx = 0
        best_t = -np.inf
        for c_ in range(n_categories):
            t = self._category_choice_value_idx(x, c_, skip, i_parts=x_parts)
            if t > best_t:
                best_t = t
                best_idx = c_
        return int(best_idx)

    def predict(
        self, X: np.ndarray, clip: bool = False, skip_channels: Optional[List[int]] = None
    ) -> np.ndarray:
        """Predict labels for the input data.

        Parameters
        ----------
        X : np.ndarray
            Input dataset.
        clip : bool
            clip the input values to be between the previously seen data limits
        skip_channels : list of int, optional
            Channels to skip (default is []).

        Returns
        -------
        np.ndarray
            Predicted labels for the input data.

        """
        check_is_fitted(self)
        if clip:
            X = np.clip(X, self.d_min_, self.d_max_)
        self.validate_data(X)
        self.check_dimensions(X)

        y = np.empty((X.shape[0],), dtype=int)
        skip = self._normalize_skip_channels(skip_channels)
        step_pred = self._step_pred_with_skip_set
        for i, x in enumerate(X):
            y[i] = step_pred(x, skip)
        return y

    def update(
        self,
        i: np.ndarray,
        w: list,
        params: Dict,
        cache: Optional[Dict] = None,
    ) -> list:
        """Update the cluster weight.

        Parameters
        ----------
        i : np.ndarray
            Input data sample.
        w : np.ndarray
            Cluster weight information.
        params : dict
            Parameters for the ART algorithm.
        cache : dict, optional
            Cache for previous calculations (default is None).

        Returns
        -------
        list
            Updated cluster weight.

        """
        assert cache is not None
        W = [
            self.modules[k].update(
                i[self._channel_indices[k][0] : self._channel_indices[k][1]],
                w[k],
                self.modules[k].params,
                cache[k],
            )
            for k in range(self.n)
        ]
        return W

    def new_weight(self, i: np.ndarray, params: Dict) -> list:
        """Generate a new cluster weight.

        Parameters
        ----------
        i : np.ndarray
            Input data sample.
        params : dict
            Parameters for the ART algorithm.

        Returns
        -------
        list
            New cluster weight.

        """
        W = [
            self.modules[k].new_weight(
                i[self._channel_indices[k][0] : self._channel_indices[k][1]],
                self.modules[k].params,
            )
            for k in range(self.n)
        ]
        return W

    def add_weight(self, new_w: list):
        """Add a new cluster weight.

        Parameters:
        - new_w: new cluster weight to add

        """
        for k in range(self.n):
            self.modules[k].add_weight(new_w[k])
        self._invalidate_channel_centers_cache()

    def set_weight(self, idx: int, new_w: list):
        """Set the value of a cluster weight.

        Parameters:
        - idx: index of cluster to update
        - new_w: new cluster weight

        """
        for k in range(self.n):
            self.modules[k].set_weight(idx, new_w[k])
        self._invalidate_channel_centers_cache()

    def get_cluster_centers(self) -> List[np.ndarray]:
        """Get the center points for each cluster.

        Returns
        -------
        list of np.ndarray
            Center points of the clusters.

        """
        centers_ = [module.get_cluster_centers() for module in self.modules]
        centers = [
            np.concatenate([centers_[k][i] for k in range(self.n)])
            for i in range(self.n_clusters)
        ]
        return centers

    def get_channel_centers(self, channel: int) -> List[np.ndarray]:
        """Get the center points of clusters for a specific channel.

        Parameters
        ----------
        channel : int
            The channel index.

        Returns
        -------
        list of np.ndarray
            Cluster centers for the specified channel.

        """
        return self.modules[channel].get_cluster_centers()

    def _get_channel_centers_array_cached(self, channel: int) -> Optional[np.ndarray]:
        if channel in self._channel_centers_cache:
            return self._channel_centers_cache[channel]
        centers = self.get_channel_centers(channel)
        centers_arr = np.asarray(centers)
        if centers_arr.dtype != object and centers_arr.ndim >= 2:
            cached = np.ascontiguousarray(centers_arr, dtype=np.float64)
            self._channel_centers_cache[channel] = cached
            return cached
        return None

    def predict_regression(
        self, X: np.ndarray, clip: bool = False, target_channels: Optional[List[int]] = None
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Predict regression values for the input data using the target channels.

        Parameters
        ----------
        X : np.ndarray
            Input dataset.
        clip : bool
            clip the input values to be between the previously seen data limits
        target_channels : list of int, optional
            List of target channels to use for regression. If negative values are used,
            they are considered as channels counting backward from the last channel.
            By default, it uses the last channel (-1).

        Returns
        -------
        Union[np.ndarray, list of np.ndarray]
            Predicted regression values. If only one target channel is used, returns a
            single np.ndarray. If multiple target channels are used, returns a list of
            np.ndarray, one for each channel.

        """
        if target_channels is None:
            target_channels = [-1]
        target_channels = [self.n + k if k < 0 else k for k in target_channels]
        C = self.predict(X, clip=clip, skip_channels=target_channels)
        c_i32 = np.ascontiguousarray(C, dtype=np.int32)
        predictions = []
        for channel in target_channels:
            centers_arr = self._get_channel_centers_array_cached(channel)
            if centers_arr is not None:
                if GatherClusterCenters is not None:
                    pred = GatherClusterCenters(c_i32, centers_arr)
                else:
                    pred = centers_arr[c_i32]
            else:
                centers = self.get_channel_centers(channel)
                pred = np.array([centers[c] for c in C])
            predictions.append(pred)
        if len(predictions) == 1:
            return predictions[0]
        return predictions

    def join_channel_data(
        self, channel_data: List[np.ndarray], skip_channels: Optional[List[int]] = None
    ) -> np.ndarray:
        """Concatenate data from different channels into a single array.

        Parameters
        ----------
        channel_data : list of np.ndarray
            Data from each channel.
        skip_channels : list of int, optional
            Channels to skip (default is []).

        Returns
        -------
        np.ndarray
            Concatenated data.

        """
        skip = self._normalize_skip_channels(skip_channels)
        n_present = self.n - len(skip)
        if len(channel_data) != n_present:
            raise ValueError(
                f"expected {n_present} present channels, got {len(channel_data)}"
            )
        if n_present == 0:
            raise ValueError("at least one non-skipped channel is required")

        formatted_channel_data = []
        n_samples: Optional[int] = None
        input_idx = 0
        channel_widths = np.array(self.channel_dims, dtype=np.int64)
        present_mask = np.array([k not in skip for k in range(self.n)], dtype=np.uint8)

        for k in range(self.n):
            width = self._channel_indices[k][1] - self._channel_indices[k][0]
            if k in skip:
                continue
            channel = np.asarray(channel_data[input_idx], dtype=np.float64)
            if channel.ndim != 2:
                raise ValueError(f"channel {k} must be 2-D")
            if channel.shape[1] != width:
                raise ValueError(
                    f"channel {k} width {channel.shape[1]} does not match expected {width}"
                )
            if n_samples is None:
                n_samples = channel.shape[0]
                if n_samples == 0:
                    raise ValueError("channels must have at least one row")
            elif channel.shape[0] != n_samples:
                raise ValueError("all channels must have the same number of rows")
            formatted_channel_data.append(channel)
            input_idx += 1

        if JoinChannelsWithFill is not None:
            return JoinChannelsWithFill(
                formatted_channel_data,
                channel_widths,
                present_mask,
                0.5,
            )

        filled_channel_data = []
        input_idx = 0
        for k in range(self.n):
            if k not in skip:
                filled_channel_data.append(formatted_channel_data[input_idx])
                input_idx += 1
            else:
                filled_channel_data.append(
                    np.full((n_samples, self.channel_dims[k]), 0.5, dtype=np.float64)
                )
        return np.hstack(filled_channel_data)

    def split_channel_data(
        self, joined_data: np.ndarray, skip_channels: Optional[List[int]] = None
    ) -> List[np.ndarray]:
        """Split the concatenated data into its original channels.

        Parameters
        ----------
        joined_data : np.ndarray
            Concatenated data from multiple channels.
        skip_channels : list of int, optional
            Channels to skip (default is []).

        Returns
        -------
        list of np.ndarray
            Split data, one array for each channel.

        """
        skip_channels = self._normalize_skip_channels(skip_channels)
        expected_dim = sum(self.channel_dims)
        if joined_data.ndim != 2:
            raise ValueError("joined_data must be 2-D")
        if joined_data.shape[1] != expected_dim:
            raise ValueError(
                f"joined_data width {joined_data.shape[1]} does not match expected {expected_dim}"
            )

        if ExtractPresentChannels is not None:
            present_mask = np.array(
                [k not in skip_channels for k in range(self.n)], dtype=np.uint8
            )
            channel_widths = np.array(self.channel_dims, dtype=np.int64)
            return list(
                ExtractPresentChannels(joined_data, channel_widths, present_mask)
            )

        channel_data = []
        current_col = 0

        for k in range(self.n):
            start_idx, end_idx = self._channel_indices[k]
            channel_width = end_idx - start_idx

            if k not in skip_channels:
                # Extract the original channel data
                channel_data.append(
                    joined_data[:, current_col : current_col + channel_width]
                )
                current_col += channel_width
            else:
                # If this channel was skipped, we know it was filled with 0.5,
                # so we skip those columns
                current_col += channel_width

        return channel_data
