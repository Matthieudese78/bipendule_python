# %%
import numpy as np

from double_pendulum.physics import GRAVITY
from double_pendulum.utils_pendulum import augmented_lhs, augmented_rhs, mulag_rhs

# %%

n = 6
nc = 2

g = GRAVITY

m1 = 1.0
l1 = 1.0

J = m1 * l1**2

fargs = {"mass": m1, "length": l1, "inertia tensor": J}
# initial conditions:
theta0 = 0 * np.pi / 180.0

x0 = l1 * np.cos(theta0)
y0 = l1 * np.sin(theta0)

dthetadt0 = 0.0

vx0 = -dthetadt0 * l1 * np.sin(theta0)
vy0 = dthetadt0 * l1 * np.cos(theta0)

# y0 = np.array([theta0, dthetadt0])

num_steps = 10000
t = np.linspace(0.0, 10.0, num_steps)
h = t[1] - t[0]

# p 128 Shabana (3rd edition 2005):

# %%
# yinit = np.array([x0, y0, vx0, vy0, theta0, dthetadt0, 0.0, 0.0])
yinit = np.array([x0, y0, vx0, vy0, theta0, dthetadt0])


def test_size_augmented_lhs():
    assert augmented_lhs(yinit, **fargs).shape[0] == n + nc
    assert augmented_lhs(yinit, **fargs).shape[1] == n + nc


def test_size_augmented_rhs():
    assert augmented_rhs(yinit, nc, **fargs).shape[0] == n + nc


def test_size_mulag_rhs():
    assert mulag_rhs(yinit, nc, **fargs).shape[0] == n + nc
