"""Elements of solving pendulum dynamical equation."""

import numpy as np

from double_pendulum.physics import GRAVITY


# for custom euler forward and backward :
def right_hand_side_pendulum(y, **fargs) -> np.ndarray:
    m1 = fargs["mass"]
    l1 = fargs["length"]
    inertia = fargs["inertia tensor"]
    gamma = -m1 * l1 * GRAVITY * np.cos(y[0]) / inertia
    return np.array([y[1], gamma])


def residu_euler(y, ypred, h, **fargs):
    return ypred - y - h * right_hand_side_pendulum(ypred, **fargs)


def residu_midpoint(y, ypred, h, **fargs):
    return ypred - y - h * right_hand_side_pendulum((y + ypred) / 2.0, **fargs)


def residu_jacobian_euler(ypred, h, **fargs):
    l1 = fargs["length"]
    return np.array(
        [
            [1.0, -h],
            [-(h * GRAVITY / l1) * np.sin(ypred[0]), 1.0],
        ]
    )


def residu_jacobian_midpoint(y: np.ndarray, ypred: np.ndarray, h: float, **fargs: dict) -> np.ndarray:
    l1 = fargs["length"]
    return np.eye(y.shape[0]) - h * 0.5 * np.array(
        [
            [0, 1.0],
            [(GRAVITY / l1) * np.sin((y[0] + ypred[0]) / 2.0), 0.0],
        ]
    )


# Right hand sides for scipy.integrate. :
#   - solve_ivp
#   - odeint
#   - rk45


def right_hand_side_solve_ivp(t: np.ndarray, y: np.ndarray, *args: tuple) -> np.ndarray:
    """Right hand sides with right signature for scipy.integrate.solve_ivp.

    Parameters:
        t : time array
        y : y array

    Returns:
        RHS
    """
    m1, l1, inertia = args
    theta, dthetadt = y
    gamma = -m1 * GRAVITY * l1 * np.cos(theta) / inertia
    return np.array([dthetadt, gamma])


def right_hand_side_odeint(y: np.ndarray, t: np.ndarray, *args: tuple) -> np.ndarray:
    """Right hand sides with right signature for scipy.integrate.odeint.

    Parameters:
        t : time array
        y : y array

    Returns:
        RHS
    """
    m1, l1, inertia = args
    gamma = -m1 * l1 * GRAVITY * np.cos(y[0]) / inertia
    return np.array([y[1], gamma])
