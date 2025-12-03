# %%
"""Starting point for the double pendulum simulation."""

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

from double_pendulum.symbolic import pendulum_cq, pendulum_qc

# %%

g = 9.81

m1 = 1.0

l1 = 1.0

J = m1 * l1**2

theta = -10.0 * np.pi / 180.0

x0 = np.cos(theta)
y0 = np.sin(theta)

# initial angular velocity in rad :
dthetadt0 = -2.0 * np.pi / 10.0
vx0 = -dthetadt0 * l1 * np.sin(theta)
vy0 = dthetadt0 * l1 * np.cos(theta)


# y = np.array([x0, y0, theta, vx0, vy0, dthetadt0])

# Dummy Lambdas :
Lambda_x = 0.0
Lambda_y = 0.0
Lambda_theta = 0.0
y = np.array([x0, y0, theta])

dydt = np.array([vx0, vy0, dthetadt0])
# external forces : Fx , Fy, gamma_z :

F = np.array([0.0, -m1 * g, -m1 * g * l1 * np.cos(theta)])

# dydt = np.array([y[3], y[4], y[5], F[0], F[1], F[2]])

M = np.array([[m1, 0, 0], [0, m1, 0], [0, 0, J]])

# Constraints jacobian (numpy fction):
cq = pendulum_cq(l1)[1]

print(f"Cq = {pendulum_cq(l1)[0]}")
# %% Generalized forces associated with constraints (numpy fction):
qc = pendulum_qc(l1)[1]
print(f"Qc = {pendulum_qc(l1)[0]}")
# qc = pendulum_qc(pendulum_cq(l1)[1])[1]

Qc = qc(theta, dthetadt0).ravel()
# Qc = qc(x0, y0, theta, vx0, vy0, dthetadt0).ravel()

print(f"Qc = {Qc}")

# %%
M_aug = np.block([[M, cq(x0, y0, theta).T], [cq(x0, y0, theta), np.zeros((2, 2))]])

print(M_aug)
# %%
print(F)
print(Qc)
F_aug = np.concatenate([F, Qc])
print(F_aug)

# %% Initializing d2ydt2 :
d2ydt2 = np.dot(la.inv(M_aug), F_aug)
print(d2ydt2)
print(np.shape(d2ydt2[:3]))

# %% vector of unknowns including the lagrange multipliers
# Static computation for initial Lambdas :

y = np.concatenate([y, dydt])
dydt = np.concatenate([dydt, d2ydt2[:3]])
# %%
print(np.shape(y))
print(np.shape(dydt))
# %% Euler forward integration
dt = 0.001
num_steps = 10000
save_discr = 4
isave = 0
saved_step = np.zeros((int(np.floor(num_steps / save_discr)), y.shape[0]))
saved_lambdas = np.zeros((int(np.floor(num_steps / save_discr)), 2))
for step in range(num_steps):
    y += dydt * dt
    # compute M_aug:
    M_aug = np.block([[M, cq(y[0], y[1], y[2]).T], [cq(y[0], y[1], y[2]), np.zeros((2, 2))]])
    # M_aug = np.concatenate([M, cq(y[0], y[1], y[2]).T])
    # compute F_aug:
    Qc = qc(y[2], y[5]).ravel()
    # F_aug = np.concatenate([F, Qc])
    F = np.array([0.0, -m1 * g, -m1 * g * l1 * np.cos(y[2])])
    F_aug = np.concatenate([F, Qc])
    # compute dydt2:
    d2ydt2 = la.inv(M_aug) @ F_aug
    # d2ydt2 = np.dot(la.inv(M_aug), F_aug)
    # update dydt:
    dydt = np.concatenate([y[:3], d2ydt2[:3]])
    # dydt = np.array([y[2], y[3], F[0], F[1]])

    # save step:
    print(f"y = {y}")
    if step % save_discr == 0:
        saved_step[isave] = y
        saved_lambdas[isave] = d2ydt2[3:]
        isave += 1
# %%
saved_step = np.array(saved_step)
plt.scatter(np.arange(len(saved_step)) * dt, saved_step[:, 1], s=4)
plt.show()

# %%
plt.scatter(np.arange(len(saved_step)) * dt, saved_step[:, 2], s=4)
plt.show()
# %%
plt.scatter(np.arange(len(saved_step)) * dt, saved_step[:, 3], s=4)
plt.show()

# %%
plt.scatter(np.arange(len(saved_step)), saved_lambdas[:, 0], s=4, label="lambda_x")
plt.scatter(np.arange(len(saved_step)), saved_lambdas[:, 1], s=4, label="lambda_y")
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
