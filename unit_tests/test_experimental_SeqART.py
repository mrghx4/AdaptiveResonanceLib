import numpy as np

from artlib.experimental.SeqART import (
    SeqART,
    arr2seq,
    compress_dashes,
    needleman_wunsch,
)


def test_compress_dashes_collapses_runs():
    assert compress_dashes("A---B--C") == "A-B-C"


def test_arr2seq_converts_integer_array_to_string():
    x = np.array([1, 0, 2, 3])
    assert arr2seq(x) == "1023"


def test_needleman_wunsch_returns_alignment_and_normalized_score():
    alignment, score = needleman_wunsch("123", "103")
    assert alignment == "1-3"
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_seqart_fit_and_predict_smoke():
    X = np.array(
        [
            [1, 2, 3],
            [1, 2, 3],
            [9, 9, 9],
        ],
        dtype=int,
    )
    model = SeqART(rho=0.5)
    model.fit(X, max_iter=1)

    pred = model.predict(X)
    assert pred.shape == (3,)
    assert pred[0] == pred[1]
