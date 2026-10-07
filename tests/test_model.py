import numpy as np
import torch

from anomaly.model import (
    N_FEATURES,
    WINDOW,
    LastStepReconstruction,
    LSTMAutoencoder,
    make_windows,
    preprocess,
)


def test_make_windows_shape_and_contents():
    x = np.arange(60 * N_FEATURES, dtype=np.float32).reshape(60, N_FEATURES)
    windows = make_windows(x)
    assert windows.shape == (60 - WINDOW + 1, WINDOW, N_FEATURES)
    # Window i covers readings i .. i+WINDOW-1, in order.
    assert np.array_equal(windows[0], x[:WINDOW])
    assert np.array_equal(windows[-1], x[-WINDOW:])


def test_preprocess_clips_out_of_range_values():
    x = np.array([[-5.0, 0.5, 9.0]])
    assert preprocess(x).tolist() == [[-1.0, 0.5, 2.0]]


def test_autoencoder_rebuilds_the_input_shape():
    model = LSTMAutoencoder()
    batch = torch.rand(4, WINDOW, N_FEATURES)
    assert model(batch).shape == batch.shape
    assert LastStepReconstruction(model)(batch).shape == (4, N_FEATURES)
