# %%
"""Starting point for the double pendulum simulation."""

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

from double_pendulum.symbolic import pendulum_cq, pendulum_qc

import scipy as sp

# %%

g = 9.81

m1 = 1.0

l1 = 1.0

J = m1 * l1**2

theta = -45.0 * np.pi / 180.0

x0 = l1 * np.cos(theta)
y0 = l1 * np.sin(theta)

dthetadt0 = 0.0

vx0 = -dthetadt0 * l1 * np.sin(theta)
vy0 = dthetadt0 * l1 * np.cos(theta)


def right_hand_side(t, y) -> np.ndarray:
    theta, dthetadt = y
    gamma = -m1 * l1 * np.cos(theta) / J
    return np.array([dthetadt, gamma])


y0 = (theta, dthetadt0)

dt = 0.1

# %%
t = np.linspace(0, 10, 100)
sol = sp.integrate.solve_ivp(right_hand_side, [0.0, 10.0], y0, t_eval=t)

plt.scatter(sol.t, sol.y[0])
plt.scatter(sol.t, sol.y[1])
plt.show()
plt.scatter(sol.t, l1 * np.cos(sol.y[0]))
plt.scatter(sol.t, l1 * np.sin(sol.y[0]))
plt.show()
plt.close("all")

plt.scatter(l1 * np.cos(sol.y[0]), l1 * np.cos(sol.y[1]))
plt.show()
plt.close("all")

# %% Euler forward integration
dt = 0.001
num_steps = 1000
save_discr = 4
isave = 0
# Initialization :
y = np.array([theta, dthetadt0])
gamma = -m1 * g * l1 * np.cos(theta)
dydt = np.array([dthetadt0, gamma / J])

saved_step = np.zeros((int(np.floor(num_steps / save_discr)), y.shape[0]))

for step in range(num_steps):
    y += dydt * dt

    gamma = -m1 * g * l1 * np.cos(y[0])

    dydt[0] = y[1]
    dydt[1] = gamma / J

    # save step:
    print(f"y = {y}")
    if step % save_discr == 0:
        saved_step[isave] = y
        isave += 1

t = np.arange(len(saved_step)) * save_discr * dt
theta = saved_step[:, 0]
dthetadt = saved_step[:, 1]
x = l1 * np.cos(saved_step[:, 0])
y = l1 * np.sin(saved_step[:, 0])
# %%
plt.scatter(t, theta, s=4)
plt.show()
# %%
plt.scatter(t, dthetadt, s=4)
plt.show()
# %%
plt.scatter(t, x, s=4)
plt.scatter(t, y, s=4)
plt.show()
# %%
plt.scatter(x, y, s=4)
plt.show()
# %%
# %%
plt.scatter(t, saved_step[:, 1], s=4)
plt.show()
# %%
plt.scatter(t, saved_step[:, 0] * 180.0 / np.pi, s=4)
plt.show()

# %%
saved_step = np.array(saved_step)
plt.scatter(t, saved_step[:, 1], s=4)
plt.show()


# %% plot the lagrangian
T = 0.5 * J * saved_step[:, 1] ** 2
# T = 0.5 * m1 * (l1 * dthetadt) ** 2
K = m1 * g * np.sin(saved_step[:, 0])
H = T + K
L = T - K
plt.scatter(t, T, s=4, color="r")
plt.scatter(t, K, s=4, color="g")
plt.scatter(t, H, s=4, color="b")
plt.scatter(t, L, s=4, color="orange")
plt.show()


# %%
