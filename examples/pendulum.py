# %%
"""Starting point for the double pendulum simulation."""

import numpy as np
import scipy as sp
from scipy.integrate import odeint

from double_pendulum.physics import GRAVITY
from double_pendulum.postreatment import post_treatment_pendulum
from double_pendulum.solvers import euler_backward_iterative, euler_forward
from double_pendulum.utils_pendulum import residu_jacobian_pendulum, residu_pendulum, right_hand_side_pendulum

# %%

g = GRAVITY

m1 = 1.0e-2
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


# %% right hand side functions:
# solve_ivp * rk45 :
def right_hand_side_solve_ivp(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Right hand sides with right signature for scipy.integrate.solve_ivp.

    Parameters:
        t : time array
        y : y array

    Returns:
        RHS
    """
    theta, dthetadt = y
    gamma = -m1 * g * l1 * np.cos(theta) / J
    return np.array([dthetadt, gamma])


def right_hand_side_odeint(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Right hand sides with right signature for scipy.integrate.odeint.

    Parameters:
        t : time array
        y : y array

    Returns:
        RHS
    """
    gamma = -m1 * l1 * g * np.cos(y[0]) / J
    return np.array([y[1], gamma])


# %%
num_steps = 10000
t = np.linspace(0.0, 10.0, num_steps)

y0 = np.array([theta0, dthetadt0])

bool_odeint = False
bool_solve_ivp = True
bool_rk45 = False

if bool_solve_ivp:
    y0 = (theta0, dthetadt0)
    result = sp.integrate.solve_ivp(right_hand_side_solve_ivp, [0.0, 10.0], y0, t_eval=t)
    sol = np.array([result.y[0], result.y[1]]).T

if bool_odeint:
    sol = odeint(right_hand_side_odeint, y0, t)

if bool_rk45:
    rk = sp.integrate.RK45(right_hand_side_solve_ivp, 0.0, y0, 10.0, max_step=1.0e-3)

    times = [rk.t]
    states = [rk.y.copy()]

    while rk.status == "running":
        rk.step()
        times.append(rk.t)
        states.append(rk.y.copy())

    t = np.asarray(times)
    sol = np.asarray(states)

print(np.shape(sol))
# %%
post_treatment_pendulum(sol, l1, m1, t)

# %% PROTOTYPES :
t = np.linspace(0.0, 10.0, num_steps)
y0 = np.array([theta0, dthetadt0])
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

# %% test en fonction :

result = euler_forward(t, y0, right_hand_side_pendulum, **fargs)

post_treatment_pendulum(result, l1, m1, t)


# %% test to put in tests/


def test_expr_residu(y, h, ypred, **fargs):
    analytic = np.array([ypred[0] - y[0] - h * ypred[1], ypred[1] - y[1] + (h * g / l1) * np.cos(ypred[0])])
    res = ypred - y - h * right_hand_side_pendulum(ypred, **fargs)
    assert np.array([analytic[i] == res[i] for i in range(2)]).all()


ypred = y0 + h * right_hand_side_pendulum(y, **fargs)
res = test_expr_residu(y0, h, ypred, **fargs)

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
