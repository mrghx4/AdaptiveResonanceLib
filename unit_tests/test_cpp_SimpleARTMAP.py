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


def test_gather_cluster_centers():
    labels = np.array([2, 0, 1], dtype=np.int32)
    centers = np.array(
        [
            [10.0, 11.0],
            [20.0, 21.0],
            [30.0, 31.0],
        ],
        dtype=np.float64,
    )
    out = cpp_simple_artmap.GatherClusterCenters(labels, centers)
    exp = np.array([[30.0, 31.0], [10.0, 11.0], [20.0, 21.0]], dtype=np.float64)
    np.testing.assert_allclose(out, exp, rtol=0.0, atol=0.0)


def test_gather_cluster_centers_out_of_range_raises():
    labels = np.array([0, 3], dtype=np.int32)
    centers = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    with pytest.raises(Exception):
        cpp_simple_artmap.GatherClusterCenters(labels, centers)


def test_map_simple_artmap_labels_chain():
    labels = np.array([0, 1, 2, 1], dtype=np.int32)
    map_chain = [
        np.array([2, 0, 1], dtype=np.int32),
        np.array([1, 0, 2], dtype=np.int32),
    ]
    out = cpp_simple_artmap.MapSimpleARTMAPLabelsChain(labels, map_chain)

    expected = labels.copy()
    for m in map_chain:
        expected = m[expected]
    assert np.array_equal(out, expected)
