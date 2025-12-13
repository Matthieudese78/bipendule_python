# %%
"""Starting point for the double pendulum simulation."""

import numpy as np
import numpy.linalg as la
import scipy as sp
from scipy.integrate import odeint

from double_pendulum.postreatment import post_treatment_pendulum
from double_pendulum.solvers import euler_backward_newton, euler_forward

GRAVITY = 9.81
# %%

g = GRAVITY

m1 = 1.0e-2
l1 = 1.0

J = m1 * l1**2

theta0 = 0 * np.pi / 180.0

x0 = l1 * np.cos(theta0)
y0 = l1 * np.sin(theta0)

dthetadt0 = 0.0

vx0 = -dthetadt0 * l1 * np.sin(theta0)
vy0 = dthetadt0 * l1 * np.cos(theta0)

y0 = np.array([theta0, dthetadt0])


# %% right hand side functions:
# solve_ivp * rk45 :
def right_hand_side_solve_ivp(t, y) -> np.ndarray:
    theta, dthetadt = y
    gamma = -m1 * g * l1 * np.cos(theta) / J
    return np.array([dthetadt, gamma])


def right_hand_side_odeint(y, t) -> np.ndarray:
    gamma = -m1 * l1 * g * np.cos(y[0]) / J
    return np.array([y[1], gamma])


# for custom euler forward and backward :
def right_hand_side(y) -> np.ndarray:
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
        y += right_hand_side(y) * h
        result[i] = y

post_treatment_pendulum(result, l1, m1, t)

# %% test en fonction :

result = euler_forward(t, y0, right_hand_side)

post_treatment_pendulum(result, l1, m1, t)


# %%
def residu(y, ypred):
    return ypred - y - h * right_hand_side(ypred)


def test_expr_residu(y, ypred):
    analytic = np.array([ypred[0] - y[0] - h * ypred[1], ypred[1] - y[1] + (h * g / l1) * np.cos(ypred[0])])
    res = ypred - y - h * right_hand_side(ypred)
    assert np.array([analytic[i] == res[i] for i in range(2)]).all()


ypred = pred(y0)
res = test_expr_residu(y0, ypred)


def residu_jacobian(ypred):
    return np.array(
        [
            [1.0, -h],
            [-(h * g / l1) * np.sin(ypred[0]), 1.0],
        ]
    )


# %%
result = euler_backward_newton(t, y0, right_hand_side, residu, residu_jacobian)
# %%
post_treatment_pendulum(
    result,
    l1,
    m1,
    t,
)


# %%
def f_pendulum(y: float | np.ndarray, **fargs: dict) -> float | np.ndarray:
    """Right hand side of the pendulum.

    Args:
        y : coordinates values
        m : mass
        l1 : length
        fargs : pendulum values

    Returns:
        right hand side

    """
    m = fargs["m"]
    g = fargs["g"]
    l1 = fargs["l"]
    return np.array([y[1], -m * g * l1 * np.cos(y[0])])


def res_pendulum_euler_backward(
    h: float, ypred: float | np.ndarray, y: float | np.ndarray, **fargs: dict
) -> float | np.ndarray:
    """Residu for the pendulum with euler backward.

    Args:
        h : time step
        y : coordinates values
        ypred : firs guess for y
        m : mass
        l1 : length
        fargs : arguments

    Returns:
        residu

    """
    g = fargs["g"]
    l1 = fargs["l"]
    th = ypred[0]
    w = ypred[1]
    return np.array([th - y[0] - h * w, w - y[1] + h * (g / l1) * np.cos(th)])


def jac_pendulum_euler_backward(h: float, ypred: float | np.ndarray, **fargs: dict) -> float | np.ndarray:
    """Residu for the pendulum with euler backward.

    Args:
        h : time step
        y : coordinates values
        ypred : firs guess for y
        m : mass
        l1 : length
        fargs : arguments

    Returns:
        residu

    """
    g = fargs["g"]
    l1 = fargs["l"]
    th = ypred[0]
    return np.array([[1.0, -h], [-(h * g / l1) * np.sin(th), 1.0]])


def euler_backward(
    t: np.ndarray, y0: float | np.ndarray, f: callable, res: callable, jac: callable, max_iter: int = 200, **fargs: dict
) -> np.ndarray:
    """Integrates an ivp solution.

    Args:
        t: time vector
        y0 : initial values
        f : right hand side
        res : residu function
        jac : residu function
        fargs : f arguments

    Returns:
        time integrated serie.

    """

    h = t[1] - t[0]

    y = y0.copy()

    result = np.zeros((len(t), y0.shape[0]))
    for i, time in enumerate(t):
        f0 = f(y, **fargs)
        ypred = np.array([y[0] + h * f0[0], y[1] + h * f0[1]])

        residu = res(h, ypred, y, **fargs)

        crit0 = la.norm(residu)
        crit = crit0
        niter = 0
        print(f"crit initial = {crit}")
        while (crit > 1.0e-12) and (niter < max_iter):
            jacobian = jac(h, ypred, **fargs)
            delta_res = -np.linalg.solve(jacobian, residu)
            ypred += delta_res
            residu = res(h, ypred, y, **fargs)
            crit = la.norm(residu)
            niter += 1

        print(f"step {i}, nb iterations = {niter}")
        print(f"          crit final = {crit}")

        y = ypred

        result[i] = y

    return result


# %% scipy example :
y0 = np.array([theta0, dthetadt0])

num_steps = 10000
t = np.linspace(0.0, 10.0, num_steps)

fargs = {"m": m1, "l": l1, "g": GRAVITY}

solver = "euler_forward"
if solver == "euler_forward":
    result = euler_forward(t, y0, f_pendulum, **fargs)
if solver == "euler_backward":
    result = euler_backward(t, y0, f_pendulum, res_pendulum_euler_backward, jac_pendulum_euler_backward, **fargs)

post_treatment_pendulum(result, l1, m1, t)

# %%
