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

h = 0.1

# %%
num_steps = 10000
t = np.linspace(0.0, 10.0, num_steps)
h = t[0] - t[1]

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

# Initialization :
y = np.array([theta, dthetadt0])
gamma = -m1 * g * l1 * np.cos(theta)
dydt = np.array([dthetadt0, gamma / J])

result = np.zeros((len(t), y.shape[0]))

for i, step in enumerate(t):
    # # Euler Forward :
    # y += dydt * h

    # gamma = -m1 * g * l1 * np.cos(y[0])

    # dydt[0] = y[1]
    # dydt[1] = gamma / J

    # Euler Backward :
    # initialization newton raphson : one step of euler forward
    gamma = -m1 * g * l1 * np.cos(y[0])
    th = y[0] + h * y[1]
    w = y[1] + h * gamma / J

    res = np.array([th - y[0] - h * w, w - y[1] + h * (g / l1) * np.cos(th)])
    crit0 = la.norm(res)
    crit = crit0
    niter = 0
    print(f"crit initial = {crit}")
    while (crit > crit0 / 10.0) and (niter < 100):
        jac = np.array([[1.0, -h], [-(h * g / l1) * np.sin(th), 1.0]])
        delta_res = -np.linalg.solve(jac, res)
        th += delta_res[0]
        w += delta_res[1]
        res += delta_res
        crit = la.norm(res)
        niter += 1

    print(f"step {i}, nb iterations = {niter}")
    print(f"          crit final = {crit}")

    y[0] = th
    y[1] = w

    # save step:
    # print(f"y = {y}")
    # if step % save_discr == 0:
    result[i] = y
    # isave += 1
# %%
theta = result[:, 0]
dthetadt = result[:, 1]
x = l1 * np.cos(result[:, 0])
y = l1 * np.sin(result[:, 0])

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

# %% plot the lagrangian
T = 0.5 * J * result[:, 1] ** 2
# T = 0.5 * m1 * (l1 * dthetadt) ** 2
K = m1 * g * np.sin(result[:, 0])
H = T + K
L = T - K
# plt.scatter(t, T, s=4, color="r")
# plt.scatter(t, K, s=4, color="g")
plt.scatter(t, H, s=4, color="b")
# plt.scatter(t, L, s=4, color="orange")
plt.show()


# %%
