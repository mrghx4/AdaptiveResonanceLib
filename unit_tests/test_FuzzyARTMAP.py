import numpy as np
from sklearn.datasets import make_blobs
from artlib.elementary.FuzzyART import FuzzyART
from artlib.supervised.SimpleARTMAP import SimpleARTMAP
from artlib.optimized.backends.cpp.FuzzyARTMAP import FuzzyARTMAP


def test_prepare_data():
    data, target = make_blobs(
        n_samples=150,
        centers=3,
        cluster_std=0.50,
        random_state=0,
        shuffle=False,
    )
    params = {"rho": 0.8, "alpha": 1e-10, "beta": 1.0}
    A = SimpleARTMAP(FuzzyART(**params))
    B = FuzzyARTMAP(**params)

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

    params = {"rho":0.8, "alpha":1e-10, "beta":1.0}
    A = SimpleARTMAP(FuzzyART(**params))
    B = FuzzyARTMAP(**params)

    X = A.prepare_data(data)

    A = A.fit(X, target)
    B = B.fit(X, target)

    assert np.array_equal(A.module_a.W, B.module_a.W)

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

    params = {"rho": 0.8, "alpha": 1e-10, "beta": 1.0}
    model = FuzzyARTMAP(**params)
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
