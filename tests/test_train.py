# Copyright 2021-2025 The DADApy Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Module for testing methods of the DiffImbalance class."""

import os
import sys

import numpy as np
import pytest
from jax import config

config.update("jax_platform_name", "cpu")
filename = os.path.join(os.path.split(__file__)[0], "3d_gauss_small_z_var.npy")


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python>=3.9")
def test_DiffImbalance_train1():
    """Test DII train function."""
    from dii import DiffImbalance  # noqa: E402

    # generate test data
    weights_ground_truth = np.array([10, 3, 100])
    data_A = np.load(filename)
    data_B = weights_ground_truth[np.newaxis, :] * data_A

    expected_weights = [0.13924, 0.04242, 0.09387]
    expected_imb = 0.05478
    expected_imb_final = 0.05478

    # train the DII to recover ground-truth metric
    dii = DiffImbalance(
        data_A,  # matrix of shape (N,D_A)
        data_B,  # matrix of shape (N,D_B)
        distances_B=None,
        periods_A=None,
        periods_B=None,
        seed=0,
        num_epochs=10,
        batches_per_epoch=1,
        l1_strength=0.0,
        point_adapt_lambda=False,
        k=10,
        lambda_factor=1e-1,
        params_init=None,
        params_groups=None,
        optimizer_name="sgd",
        learning_rate=1e-1,
        learning_rate_decay="cos",
        num_points_rows=None,
    )
    weights, imbs = dii.train()

    # compute final DII
    imb_final, _ = dii.return_final_dii(
        compute_error=False, ratio_rows_columns=None, seed=0, discard_close_ind=0
    )

    assert weights[-1] == pytest.approx(expected_weights, abs=0.001)
    assert imbs[-1] == pytest.approx(expected_imb, abs=0.001)
    assert imb_final == pytest.approx(expected_imb_final, abs=0.001)


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python>=3.9")
def test_DiffImbalance_train2():
    """Test DII train function."""
    from dii import DiffImbalance  # noqa: E402

    # generate test data
    weights_ground_truth = np.array([10, 3, 100])
    params_init = np.array([10.0, 10.0, 10.0])
    data_A = np.load(filename)
    data_B = weights_ground_truth[np.newaxis, :] * data_A

    expected_weights = [11.83414, 5.17263, 11.54111]
    expected_imb = 0.14955
    expected_imb_final = 0.04635

    # train the DII
    dii = DiffImbalance(
        data_A,
        data_B,
        distances_B=None,
        periods_A=None,
        periods_B=None,
        seed=0,
        num_epochs=10,
        batches_per_epoch=5,
        l1_strength=1e-4,
        point_adapt_lambda=False,
        k=1,
        lambda_factor=1e-1,
        params_init=params_init,
        params_groups=None,
        optimizer_name="adam",
        learning_rate=1e-1,
        learning_rate_decay=None,
    )
    weights, imbs = dii.train()

    imb_final, _ = dii.return_final_dii(
        compute_error=False, ratio_rows_columns=None, seed=0, discard_close_ind=10
    )

    assert weights[-1] == pytest.approx(expected_weights, abs=0.001)
    assert imbs[-1] == pytest.approx(expected_imb, abs=0.001)
    assert imb_final == pytest.approx(expected_imb_final, abs=0.001)


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python>=3.9")
def test_DiffImbalance_train3():
    """Test DII train function."""
    from dii import DiffImbalance  # noqa: E402

    # generate test data
    weights_ground_truth = np.array([10, 3, 100])
    data_A = np.load(filename)
    data_B = weights_ground_truth[np.newaxis, :] * data_A

    expected_weights = [0.14967, -0.07077, 0.05089]
    expected_imb = 0.43272
    expected_imb_final = 0.59785
    expected_error_final = 0.07628

    # train the DII
    dii = DiffImbalance(
        data_A,
        data_B,
        distances_B=None,
        periods_A=2 * np.pi,
        periods_B=2 * np.pi,
        seed=0,
        num_epochs=10,
        batches_per_epoch=1,
        l1_strength=0.0,
        point_adapt_lambda=True,
        k=1,
        lambda_factor=1e-1,
        params_init=None,
        params_groups=None,
        optimizer_name="sgd",
        learning_rate=1e-1,
        learning_rate_decay="cos",
        num_points_rows=None,
    )
    weights, imbs = dii.train()

    # compute final DII
    imb_final, error_final = dii.return_final_dii(
        compute_error=True, ratio_rows_columns=1, seed=0, discard_close_ind=0
    )

    assert weights[-1] == pytest.approx(expected_weights, abs=0.01)
    assert imbs[-1] == pytest.approx(expected_imb, abs=0.01)
    assert imb_final == pytest.approx(expected_imb_final, abs=0.001)
    assert error_final == pytest.approx(expected_error_final, abs=0.001)


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python>=3.9")
def test_DiffImbalance_train4():
    """Test DII train function."""
    from dii import DiffImbalance  # noqa: E402

    # generate test data
    weights_ground_truth = np.array([10, 3, 100])
    data_A = np.load(filename)
    data_B = weights_ground_truth[np.newaxis, :] * data_A

    expected_weights = [0.12785, 0.05922, 0.10074]
    expected_imb = 0.03108
    expected_imb_final = 0.09842
    expected_error_final = 0.01501

    # train the DII to recover ground-truth metric
    dii = DiffImbalance(
        data_A,  # matrix of shape (N,D_A)
        data_B,  # matrix of shape (N,D_B)
        distances_B=None,
        periods_A=None,
        periods_B=None,
        seed=0,
        num_epochs=10,
        batches_per_epoch=1,
        l1_strength=0.0,
        point_adapt_lambda=False,
        k=1,
        lambda_factor=1e-1,
        params_init=None,
        params_groups=None,
        optimizer_name="sgd",
        learning_rate=1e-1,
        learning_rate_decay="cos",
        num_points_rows=50,
    )
    weights, imbs = dii.train()

    # compute final DII
    imb_final, error_final = dii.return_final_dii(
        compute_error=True, ratio_rows_columns=0.5, seed=0, discard_close_ind=1
    )

    assert weights[-1] == pytest.approx(expected_weights, abs=0.01)
    assert imbs[-1] == pytest.approx(expected_imb, abs=0.01)
    assert imb_final == pytest.approx(expected_imb_final, abs=0.001)
    assert error_final == pytest.approx(expected_error_final, abs=0.001)


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python>=3.9")
def test_DiffImbalance_train5():
    """Test DII train function."""
    from dii import DiffImbalance  # noqa: E402

    # generate test data
    weights_ground_truth = np.array([10, 3, 100])
    data_A = np.load(filename)
    data_B = weights_ground_truth[np.newaxis, :] * data_A
    params_init = [1, 0.1]
    params_groups = [2, 1]

    expected_weights = [1.0, 0.10002]
    expected_imb = 0.03980
    expected_imb_final = 0.14683
    expected_error_final = 0.02029

    # train the DII to recover ground-truth metric
    dii = DiffImbalance(
        data_A,  # matrix of shape (N,D_A)
        data_B,  # matrix of shape (N,D_B)
        distances_B=None,
        periods_A=None,
        periods_B=None,
        seed=0,
        num_epochs=10,
        batches_per_epoch=1,
        l1_strength=0.0,
        point_adapt_lambda=False,
        k=1,
        lambda_factor=1e-1,
        params_init=params_init,
        params_groups=params_groups,
        optimizer_name="sgd",
        learning_rate=1e-1,
        learning_rate_decay="cos",
        num_points_rows=50,
    )
    weights, imbs = dii.train()

    # compute final DII
    imb_final, error_final = dii.return_final_dii(
        compute_error=True, ratio_rows_columns=0.5, seed=0, discard_close_ind=1
    )

    assert weights[-1] == pytest.approx(expected_weights, abs=0.01)
    assert imbs[-1] == pytest.approx(expected_imb, abs=0.01)
    assert imb_final == pytest.approx(expected_imb_final, abs=0.001)
    assert error_final == pytest.approx(expected_error_final, abs=0.001)


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python>=3.9")
def test_DiffImbalance_train6():
    """Test DII train function."""
    from dii import DiffImbalance  # noqa: E402

    # generate test data
    weights_ground_truth = np.array([10, 3, 100])
    data_A = np.load(filename)
    data_B = weights_ground_truth[np.newaxis, :] * data_A
    distances_B = ((data_B[np.newaxis, :, :] - data_B[:, np.newaxis, :]) ** 2).sum(
        axis=-1
    )

    expected_weights = [0.12785, 0.05922, 0.10074]
    expected_imb = 0.03108
    expected_imb_final = 0.09842
    expected_error_final = 0.01501

    # train the DII to recover ground-truth metric
    dii = DiffImbalance(
        data_A,  # matrix of shape (N,D_A)
        data_B=None,  # matrix of shape (N,D_B)
        distances_B=distances_B,
        periods_A=None,
        periods_B=None,
        seed=0,
        num_epochs=10,
        batches_per_epoch=1,
        l1_strength=0.0,
        point_adapt_lambda=False,
        k=1,
        lambda_factor=1e-1,
        params_init=None,
        params_groups=None,
        optimizer_name="sgd",
        learning_rate=1e-1,
        learning_rate_decay="cos",
        num_points_rows=50,
    )
    weights, imbs = dii.train()

    # compute final DII
    imb_final, error_final = dii.return_final_dii(
        compute_error=True, ratio_rows_columns=0.5, seed=0, discard_close_ind=1
    )

    assert weights[-1] == pytest.approx(expected_weights, abs=0.01)
    assert imbs[-1] == pytest.approx(expected_imb, abs=0.01)
    assert imb_final == pytest.approx(expected_imb_final, abs=0.001)
    assert error_final == pytest.approx(expected_error_final, abs=0.001)
