import numpy as np
import pytest

cpp_simple_artmap = pytest.importorskip(
    "artlib.optimized.backends.cpp.cppSimpleARTMAP"
)


def test_map_simple_artmap_labels_dense_map():
    labels_a = np.array([0, 2, 1, 2, 0], dtype=np.int32)
    cluster_labels = np.array([10, 20, 30], dtype=np.int32)

    y_b = cpp_simple_artmap.MapSimpleARTMAPLabels(labels_a, cluster_labels)
    assert np.array_equal(y_b, np.array([10, 30, 20, 30, 10], dtype=np.int32))


def test_map_simple_artmap_labels_out_of_range_raises():
    labels_a = np.array([0, 3], dtype=np.int32)
    cluster_labels = np.array([4, 5, 6], dtype=np.int32)

    with pytest.raises(Exception):
        cpp_simple_artmap.MapSimpleARTMAPLabels(labels_a, cluster_labels)
