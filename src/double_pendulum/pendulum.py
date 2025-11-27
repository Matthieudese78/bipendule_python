# %%
"""Starting point for the double pendulum simulation."""

import matplotlib.pyplot as plt
import numpy as np

from double_pendulum.symbolic import pendulum_cq, pendulum_qc

# %%

g = 9.81

m1 = 1.0

l1 = 1.0

J = m1 * l1**2

theta = 10.0 * np.pi / 180.0

x0 = np.cos(theta)
y0 = np.sin(theta)

# initial angular velocity in rad :
dthetadt0 = np.pi / 10.0
vx0 = 1.0
vy0 = 20.0


y = np.array([x0, y0, theta, vx0, vy0, dthetadt0])
# external forces : Fx , Fy, gamma_z :

F = np.array([0.0, -m1 * g, -m1 * g * l1 * np.cos(theta)])

dydt = np.array([y[3], y[4], y[5], F[0], F[1], F[2]])

M = np.array([[m1, 0, 0], [0, m1, 0], [0, 0, J]])

# Constraints jacobian (numpy fction):
cq = pendulum_cq(l1)[1]
# %% Generalized forces associated with constraints (numpy fction):
qc = pendulum_qc(pendulum_cq(l1)[0])[1]

Qc = qc(x0, y0, theta, vx0, vy0, dthetadt0).ravel().ravel()

print(f"Qc = {Qc}")

# %%
M_aug = np.block([[M, cq(x0, y0, theta).T], [cq(x0, y0, theta), np.zeros((2, 2))]])

print(M_aug)
# %%
print(F)
print(Qc)
f_aug = np.concatenate([F, Qc])
print(f_aug)

# %% Euler forward integration
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
