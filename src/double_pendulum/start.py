# %%
"""Starting point for the double pendulum simulation."""

import matplotlib.pyplot as plt
import numpy as np

g = 9.81

m1 = 1.0

x0 = 0.0
y0 = 0.0
vx0 = 1.0
vy0 = 20.0

y = np.array([x0, y0, vx0, vy0])

F = np.array([0.0, -m1 * g])

dydt = np.array([y[2], y[3], F[0], F[1]])


# Euler forward integration
dt = 0.01
num_steps = 1000
save_discr = 4
isave = 0
saved_step = np.zeros((int(np.floor(num_steps / save_discr)), y.shape[0]))
for step in range(num_steps):
    y += dydt * dt
    print(f"y = {y}")
    dydt = np.array([y[2], y[3], F[0], F[1]])
    if step % save_discr == 0:
        saved_step[isave] = y
        isave += 1

saved_step = np.array(saved_step)
plt.scatter(saved_step[:, 0], saved_step[:, 1], s=4)
plt.show()

# %%
plt.scatter(saved_step[:, 0], saved_step[:, 3], s=4)
plt.show()

# %% plot the lagrangian
T = 0.5 * m1 * (saved_step[:, 3] ** 2)
K = m1 * g * saved_step[:, 1]
H = T + K
L = T - K
t = dt * np.arange(len(saved_step))
plt.scatter(t, T, s=4, color="r")
plt.scatter(t, K, s=4, color="g")
plt.scatter(t, H, s=4, color="b")
plt.scatter(t, L, s=4, color="orange")
plt.show()

# %%
