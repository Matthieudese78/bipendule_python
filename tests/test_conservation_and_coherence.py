import numpy as np
from pytest import approx

from double_pendulum.physics import GRAVITY

g = GRAVITY


def test_conformity_euler_forward_backward(
    result_pendulum_euler_forward: np.ndarray, result_pendulum_euler_backward_iterative: np.ndarray
):
    # assert position delta
    assert result_pendulum_euler_backward_iterative[:, 0] == approx(result_pendulum_euler_forward[:, 0], rel=1.0e-2)
    # assert velocity delta
    assert result_pendulum_euler_backward_iterative[:, -1] == approx(result_pendulum_euler_forward[:, -1], rel=1.0e-2)


def test_conservation_euler_forward(result_pendulum_euler_forward: np.ndarray, pendulum_setup: dict):
    J = pendulum_setup["inertia tensor"]
    m1 = pendulum_setup["mass"]
    T = 0.5 * J * result_pendulum_euler_forward[:, 1] ** 2
    K = m1 * g * np.sin(result_pendulum_euler_forward[:, 0])
    H = T + K  # Hamiltonian
    assert H[-1] == approx(H[0], rel=1.0e-2)


def test_conservation_euler_backward_iterative(
    result_pendulum_euler_backward_iterative: np.ndarray, pendulum_setup: dict
):
    J = pendulum_setup["inertia tensor"]
    m1 = pendulum_setup["mass"]
    T = 0.5 * J * result_pendulum_euler_backward_iterative[:, 1] ** 2
    K = m1 * g * np.sin(result_pendulum_euler_backward_iterative[:, 0])
    H = T + K  # Hamiltonian
    assert H[-1] == approx(H[0], rel=1.0e-2)
