import numpy as np
from pytest import approx

tol = 3.0e-2


def test_conformity_euler_forward_solve_ivp(
    result_pendulum_euler_forward: np.ndarray, result_pendulum_solve_ivp: np.ndarray
):
    # assert final position delta
    assert (
        (result_pendulum_solve_ivp[-1, 0] - result_pendulum_euler_forward[-1, 0]) / result_pendulum_solve_ivp[-1, 0]
    ) < tol
    # assert final velocity delta
    assert (
        (result_pendulum_solve_ivp[-1, 1] - result_pendulum_euler_forward[-1, 1]) / result_pendulum_solve_ivp[-1, 1]
    ) < tol


def test_conformity_euler_forward_odeint(result_pendulum_euler_forward: np.ndarray, result_pendulum_odeint: np.ndarray):
    # assert final position delta
    assert (
        (result_pendulum_odeint[-1, 0] - result_pendulum_euler_forward[-1, 0]) / result_pendulum_odeint[-1, 0]
    ) < tol
    # assert final velocity delta
    assert (
        (result_pendulum_odeint[-1, 1] - result_pendulum_euler_forward[-1, 1]) / result_pendulum_odeint[-1, 1]
    ) < tol


# def test_conformity_euler_forward_rk45(result_pendulum_euler_forward: np.ndarray, result_pendulum_rk45: np.ndarray):
#     print(f"test euler_forward_rk45 shape t, sol {type(result_pendulum_rk45)}")
#     # print(f"test euler_forward_rk45 shape t, sol {np.shape(result_pendulum_rk45)}")
#     # assert final position delta
#     assert result_pendulum_rk45[1][:, 0] == approx(result_pendulum_euler_forward[:, 0], rel=tol)
#     # assert final velocity delta
#     assert result_pendulum_rk45[1][:, -1] == approx(result_pendulum_euler_forward[:, -1], rel=tol)


def test_conformity_euler_backward_solve_ivp(
    result_pendulum_euler_backward_iterative: np.ndarray, result_pendulum_solve_ivp: np.ndarray
):
    # assert final position delta
    assert (
        (result_pendulum_solve_ivp[-1, 0] - result_pendulum_euler_backward_iterative[-1, 0])
        / result_pendulum_solve_ivp[-1, 0]
    ) < tol
    # assert final velocity delta
    assert (
        (result_pendulum_solve_ivp[-1, 1] - result_pendulum_euler_backward_iterative[-1, 1])
        / result_pendulum_solve_ivp[-1, 1]
    ) < tol


def test_conformity_euler_backward_odeint(
    result_pendulum_euler_backward_iterative: np.ndarray, result_pendulum_odeint: np.ndarray
):
    # assert final position delta
    assert (
        (result_pendulum_odeint[-1, 0] - result_pendulum_euler_backward_iterative[-1, 0])
        / result_pendulum_odeint[-1, 0]
    ) < tol
    # assert final velocity delta
    assert (
        (result_pendulum_odeint[-1, 1] - result_pendulum_euler_backward_iterative[-1, 1])
        / result_pendulum_odeint[-1, 1]
    ) < tol


# def test_conformity_euler_backward_rk45(result_pendulum_euler_backward_iterative: np.ndarray, result_pendulum_rk45: np.ndarray):
#     # assert final position delta
#     assert result_pendulum_rk45[1][:, 0] == approx(result_pendulum_euler_backward_iterative[:, 0], rel=tol)
#     # assert final velocity delta
#     assert result_pendulum_rk45[1][:, -1] == approx(result_pendulum_euler_backward_iterative[:, -1], rel=tol)
