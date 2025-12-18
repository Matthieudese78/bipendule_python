# %%
"""Starting point for the double pendulum simulation."""

import numpy as np
import scipy as sp
from scipy.integrate import odeint

from double_pendulum.physics import GRAVITY
from double_pendulum.postreatment import post_treatment_pendulum
from double_pendulum.solvers import euler_backward_direct, euler_backward_iterative, euler_forward
from double_pendulum.utils_pendulum import (
    pendulum_euler_backward_rhs_inverse,
    residu_jacobian_pendulum,
    residu_pendulum,
    right_hand_side_odeint,
    right_hand_side_pendulum,
    right_hand_side_solve_ivp,
)

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

# %%

bool_odeint = False
bool_solve_ivp = False
bool_rk45 = True

if bool_solve_ivp:
    print("SOLVE_IVP")
    y0 = (theta0, dthetadt0)
    result = sp.integrate.solve_ivp(
        right_hand_side_solve_ivp,
        [0.0, t[-1]],
        y0,
        t_eval=t,
        args=(fargs["mass"], fargs["length"], fargs["inertia tensor"]),
    )
    sol = np.array([result.y[0], result.y[1]]).T

if bool_odeint:
    print("ODEINT")
    sol = odeint(right_hand_side_odeint, y0, t, args=(fargs["mass"], fargs["length"], fargs["inertia tensor"]))

if bool_rk45:
    print("RK45")

    def wrapper_rhs_rk45(t, y):
        return right_hand_side_solve_ivp(t, y, *(fargs["mass"], fargs["length"], fargs["inertia tensor"]))

    rk = sp.integrate.RK45(
        wrapper_rhs_rk45,
        # right_hand_side_solve_ivp,
        0.0,
        y0,
        t[-1],
        max_step=h,
    )

    times = [rk.t]
    states = [rk.y.copy()]

    while rk.status == "running":
        rk.step()
        times.append(rk.t)
        states.append(rk.y.copy())

    time = np.asarray(times)
    sol = np.asarray(states)

print(np.shape(sol))

post_treatment_pendulum(sol, l1, m1, time)

# %% PROTOTYPES :
# Solver :
choice = "euler_forward"
# choice = "euler_backward"

result = np.zeros((len(t), y0.shape[0]))

h = t[1] - t[0]
# # Euler Forward :
if choice == "euler_forward":
    y = y0.copy()
    for i, step in enumerate(t):
        y += right_hand_side_pendulum(y, **fargs) * h
        result[i] = y

post_treatment_pendulum(result, l1, m1, t)
# %% euler backward direct :
result = euler_backward_direct(t, y0, right_hand_side_pendulum, pendulum_euler_backward_rhs_inverse, **fargs)

post_treatment_pendulum(result, l1, m1, t)
# %% test en fonction :

result = euler_forward(t, y0, right_hand_side_pendulum, **fargs)

post_treatment_pendulum(result, l1, m1, t)

# %%
result = euler_backward_iterative(t, y0, right_hand_side_pendulum, residu_pendulum, residu_jacobian_pendulum, **fargs)

# %%
post_treatment_pendulum(
    result,
    l1,
    m1,
    t,
)

# %%
