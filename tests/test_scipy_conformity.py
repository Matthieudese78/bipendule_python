import numpy as np
from pytest import approx

tol = 0.2


def test_conformity_euler_forward_solve_ivp(
    result_pendulum_euler_forward: np.ndarray, result_pendulum_solve_ivp: np.ndarray
):
    # assert position delta
    assert result_pendulum_solve_ivp[:, 0] == approx(result_pendulum_euler_forward[:, 0], rel=tol)
    # assert velocity delta
    assert result_pendulum_solve_ivp[:, -1] == approx(result_pendulum_euler_forward[:, -1], rel=tol)


def test_conformity_euler_forward_odeint(result_pendulum_euler_forward: np.ndarray, result_pendulum_odeint: np.ndarray):
    # assert position delta
    assert result_pendulum_odeint[:, 0] == approx(result_pendulum_euler_forward[:, 0], rel=tol)
    # assert velocity delta
    assert result_pendulum_odeint[:, -1] == approx(result_pendulum_euler_forward[:, -1], rel=tol)


def test_conformity_euler_forward_rk45(result_pendulum_euler_forward: np.ndarray, result_pendulum_rk45: np.ndarray):
    # assert position delta
    assert result_pendulum_rk45[1][:, 0] == approx(result_pendulum_euler_forward[:, 0], rel=tol)
    # assert velocity delta
    assert result_pendulum_rk45[1][:, -1] == approx(result_pendulum_euler_forward[:, -1], rel=tol)


# def test_conformity_euler_backward_solve_ivp(
#     result_pendulum_euler_backward: np.ndarray, result_pendulum_solve_ivp: np.ndarray
# ):
#     # assert position delta
#     assert result_pendulum_solve_ivp[:, 0] == approx(result_pendulum_euler_backward[:, 0], rel=1.0e-2)
#     # assert velocity delta
#     assert result_pendulum_solve_ivp[:, -1] == approx(result_pendulum_euler_backward[:, -1], rel=1.0e-2)


# def test_conformity_euler_backward_odeint(
#     result_pendulum_euler_backward: np.ndarray, result_pendulum_odeint: np.ndarray
# ):
#     # assert position delta
#     assert result_pendulum_odeint[:, 0] == approx(result_pendulum_euler_backward[:, 0], rel=1.0e-2)
#     # assert velocity delta
#     assert result_pendulum_odeint[:, -1] == approx(result_pendulum_euler_backward[:, -1], rel=1.0e-2)


# def test_conformity_euler_backward_rk45(result_pendulum_euler_backward: np.ndarray, result_pendulum_rk45: np.ndarray):
#     # assert position delta
#     assert result_pendulum_rk45[1][:, 0] == approx(result_pendulum_euler_backward[:, 0], rel=1.0e-2)
#     # assert velocity delta
#     assert result_pendulum_rk45[1][:, -1] == approx(result_pendulum_euler_backward[:, -1], rel=1.0e-2)
