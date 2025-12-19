"""Starting point for the double pendulum simulation."""

import numpy as np

from double_pendulum.physics import GRAVITY
# %%

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

y0 = np.array([theta0, dthetadt0])

num_steps = 10000
t = np.linspace(0.0, 10.0, num_steps)
h = t[1] - t[0]
