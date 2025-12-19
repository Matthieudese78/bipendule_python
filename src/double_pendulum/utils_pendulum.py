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


def cq_pendulum(y: np.ndarray, **fargs: dict) -> np.ndarray:
    l1 = fargs["length"]
    theta = y[4]  # y = (x,y,dot(x),dot(y),theta,dot(theta))
    np.array([[1.0, 0.0, 0.0, 0.0, l1 * np.sin(theta), 0.0][0.0, 1.0, 0.0, 0.0, -l1 * np.cos(theta), 0.0]])


def augmented_lhs(y: np.ndarray, **fargs: dict) -> np.ndarray:
    """Creates a left hand side matrix for the 1st order constrained DAE.

    Parameters :
        y : state space coordinates
        fargs : miscellaneous data
        n : number of coordinates
        nc : number of kinematic constraints
    Returns :
        (n+nc) x (n+nc) augmented left hand side.
    """
    n = y.shape[0]
    cq = cq_pendulum(y, **fargs)
    nc = cq.shape[0]
    return np.block([[np.eye(n), cq.T], [cq, np.zeros((nc, nc))]])


def augmented_rhs(y: np.ndarray, nc: int, **fargs: dict) -> np.ndarray:
    m1 = fargs["mass"]
    l1 = fargs["length"]
    # reminder  y = (x,y,dot(x),dot(y),theta,dot(theta), lambda_x, lambda_y)
    f = np.array([y[2], y[3], 0.0, -m1 * GRAVITY, y[5], -(GRAVITY / l1) * np.cos(y[4])])
    # for now, the appended joints forces are null.
    # which means holonomic constraints (independent of velocity) + no motor.
    return np.concatenate(f, np.zeros(nc))


def mulag_rhs(y: np.ndarray, nc: np.ndarray, **fargs: dict) -> np.ndarray:
    return np.linalg.solve(augmented_lhs(y, **fargs), augmented_rhs(y, nc, **fargs))
