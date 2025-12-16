import numpy as np
import pytest

from double_pendulum.solvers import (
    euler_backward_iterative,
    euler_forward,
)
from double_pendulum.utils_pendulum import (
    residu_jacobian_pendulum,
    residu_pendulum,
    right_hand_side_pendulum,
)

m1 = 1.0
l1 = 1.0
inertia = m1 * l1**2

theta0 = -45.0 * np.pi / 180.0
dthetadt0 = 0.0
y0 = np.array([theta0, dthetadt0])
num_steps = 10000
t = np.linspace(0.0, 10.0, num_steps)

fargs = {"mass": m1, "length": l1, "inertia tensor": inertia}


@pytest.fixture
def pendulum_setup():
    yield fargs


@pytest.fixture
def result_pendulum_euler_forward():
    yield euler_forward(t, y0, right_hand_side_pendulum, **fargs)


@pytest.fixture
def result_pendulum_euler_backward_iterative():
    yield euler_backward_iterative(t, y0, right_hand_side_pendulum, residu_pendulum, residu_jacobian_pendulum, **fargs)
