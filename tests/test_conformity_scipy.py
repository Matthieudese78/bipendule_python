import numpy as np

from double_pendulum.pendulum import (
    euler_backward,
    euler_forward,
    f_pendulum,
    jac_pendulum_euler_backward,
    res_pendulum_euler_backward,
)

g = 9.81
m1 = 1.0
l1 = 1.0
J = m1 * l1**2
theta0 = -45.0 * np.pi / 180.0
dthetadt0 = 0.0
y0 = np.array([theta0, dthetadt0])
num_steps = 10000
t = np.linspace(0.0, 10.0, num_steps)
h = t[0] - t[1]

fargs = {"m": m1, "l": l1, "g": g}

result_forward = euler_forward(t, y0, f_pendulum, **fargs)

result_backward = euler_backward(t, y0, f_pendulum, res_pendulum_euler_backward, jac_pendulum_euler_backward, **fargs)


def test_conformity_euler_forward():
    assert np.linalg.norm((result_backward[:, -1] - result_forward[:, -1])) < 1.0e-6


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
