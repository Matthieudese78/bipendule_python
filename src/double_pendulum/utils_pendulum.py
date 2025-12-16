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


def residu_pendulum(y, ypred, h, **fargs):
    return ypred - y - h * right_hand_side_pendulum(ypred, **fargs)


def residu_jacobian_pendulum(ypred, h, **fargs):
    l1 = fargs["length"]
    return np.array(
        [
            [1.0, -h],
            [-(h * GRAVITY / l1) * np.sin(ypred[0]), 1.0],
        ]
    )
