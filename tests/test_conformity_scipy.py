import numpy as np

from double_pendulum.physics import GRAVITY
from double_pendulum.solvers import (
    euler_backward_iterative,
    euler_forward,
)
from double_pendulum.utils_pendulum import (
    residu_jacobian_pendulum,
    residu_pendulum,
    right_hand_side_pendulum,
)

g = GRAVITY
m1 = 1.0
l1 = 1.0
J = m1 * l1**2
theta0 = -45.0 * np.pi / 180.0
dthetadt0 = 0.0
y0 = np.array([theta0, dthetadt0])
num_steps = 10000
t = np.linspace(0.0, 10.0, num_steps)
h = t[0] - t[1]

fargs = {"mass": m1, "length": l1, "inertia tensor": J}

result_forward = euler_forward(t, y0, right_hand_side_pendulum, **fargs)

result_backward = euler_backward_iterative(
    t, y0, right_hand_side_pendulum, residu_pendulum, residu_jacobian_pendulum, **fargs
)


def test_conformity_euler_forward():
    assert np.linalg.norm((result_backward[:, -1] - result_forward[:, -1])) < 1.0e-1


def test_conservation_forward():
    T = 0.5 * J * result_forward[:, 1] ** 2
    K = m1 * g * np.sin(result_forward[:, 0])
    H = T + K  # Hamiltonian
    assert np.abs(H[0] - H[-1]) < 0.05 * np.abs(H[0])


def test_conservation_backward():
    T = 0.5 * J * result_backward[:, 1] ** 2
    K = m1 * g * np.sin(result_backward[:, 0])
    H = T + K  # Hamiltonian
    assert np.abs(H[0] - H[-1]) < 0.05 * np.abs(H[0])
