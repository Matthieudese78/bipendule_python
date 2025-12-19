"""Starting point for the double pendulum simulation."""

# %%
import numpy as np

from double_pendulum.physics import GRAVITY
from double_pendulum.postreatment import post_treatment_pendulum
from double_pendulum.solvers import euler_forward
from double_pendulum.utils_pendulum import mulag_rhs

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

# p 128 Shabana (3rd edition 2005):

# %%
# pour demain : probleme de shape
# dans utils pendulum, certaines foncitons magent du 6, d'autres du 8 --> à voir.
y0 = np.array([x0, y0, vx0, vy0, theta0, dthetadt0])
result = euler_forward(t, y0, mulag_rhs, **fargs)
# %%
post_treatment_pendulum(
    result,
    l1,
    m1,
    t,
)

# %%
