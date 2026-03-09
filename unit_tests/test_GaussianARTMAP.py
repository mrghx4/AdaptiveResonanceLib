import numpy as np
from sklearn.datasets import make_blobs
from artlib.elementary.GaussianART import GaussianART
from artlib.supervised.SimpleARTMAP import SimpleARTMAP
from artlib.optimized.backends.cpp.GaussianARTMAP import GaussianARTMAP


def test_prepare_data():
    data, target = make_blobs(
        n_samples=150,
        centers=3,
        cluster_std=0.50,
        random_state=0,
        shuffle=False,
    )
    params = {
        "rho": 0.05,
        "alpha": 1e-10,
        "sigma_init": np.array([0.5, 0.5]),
    }
    A = SimpleARTMAP(GaussianART(**params))
    B = GaussianARTMAP(**params)

    X_A = A.prepare_data(data)
    X_B = B.prepare_data(data)
    assert np.array_equal(X_A, X_B)



def test_consistency():
    data, target = make_blobs(
            n_samples=1500,
            centers=3,
            cluster_std=0.50,
            random_state=0,
            shuffle=False,
        )

    params = {
        "rho": 0.05,
        "alpha": 1e-10,
        "sigma_init": np.array([0.5, 0.5]),
    }
    A = SimpleARTMAP(GaussianART(**params))
    B = GaussianARTMAP(**params)

    X = A.prepare_data(data)

    A = A.fit(X, target)
    B = B.fit(X, target)

    assert len(A.module_a.W) == len(B.module_a.W)
    for w_a, w_b in zip(A.module_a.W, B.module_a.W):
        assert np.allclose(w_a, w_b, rtol=0.0, atol=1e-12)

    y_A = A.labels_
    y_B = B.labels_

    assert np.array_equal(y_A, y_B)


def test_predict_cache_reuse_and_refresh():
    data, target = make_blobs(
        n_samples=320,
        centers=3,
        cluster_std=0.55,
        random_state=7,
        shuffle=False,
    )

    params = {
        "rho": 0.05,
        "alpha": 1e-10,
        "sigma_init": np.array([0.5, 0.5]),
    }
    model = GaussianARTMAP(**params)
    X = model.prepare_data(data)
    model.fit(X, target)

    y1 = model.predict(X[:30])
    w_id_1 = id(model._cpp_predict_weights)
    cl_id_1 = id(model._cpp_predict_cluster_labels)

    y2 = model.predict(X[:30])
    w_id_2 = id(model._cpp_predict_weights)
    cl_id_2 = id(model._cpp_predict_cluster_labels)

    assert np.array_equal(y1, y2)
    assert w_id_1 == w_id_2
    assert cl_id_1 == cl_id_2

    model.partial_fit(X[:40], target[:40])
    _ = model.predict(X[:30])
    w_id_3 = id(model._cpp_predict_weights)
    cl_id_3 = id(model._cpp_predict_cluster_labels)

    assert (w_id_3 != w_id_2) or (cl_id_3 != cl_id_2)
