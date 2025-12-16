"""Elements of solving pendulum dynamical equation."""

import numpy as np

from double_pendulum.physics import GRAVITY


# for custom euler forward and backward :
def right_hand_side_pendulum(y) -> np.ndarray:
    gamma = -m1 * l1 * GRAVITY * np.cos(y[0]) / J
    return np.array([y[1], gamma])


def residu_pendulum(y, ypred):
    return ypred - y - h * right_hand_side(ypred)


def residu_jacobian_pendulum(ypred):
    return np.array(
        [
            [1.0, -h],
            [-(h * GRAVITY / l1) * np.sin(ypred[0]), 1.0],
        ]
    )
