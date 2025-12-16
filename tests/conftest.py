import numpy as np
import pytest
import scipy as sp

from double_pendulum.solvers import (
    euler_backward_iterative,
    euler_forward,
)
from double_pendulum.utils_pendulum import (
    residu_jacobian_pendulum,
    residu_pendulum,
    right_hand_side_odeint,
    right_hand_side_pendulum,
    right_hand_side_solve_ivp,
)

m1 = 1.0
l1 = 1.0
inertia = m1 * l1**2

theta0 = -45.0 * np.pi / 180.0
dthetadt0 = 0.0
y0 = np.array([theta0, dthetadt0])
num_steps = 10000
t = np.linspace(0.0, 1.0, num_steps)

fargs = {"mass": m1, "length": l1, "inertia tensor": inertia}


@pytest.fixture
def pendulum_setup():
    yield fargs


@pytest.fixture
def result_pendulum_euler_forward():
    yield euler_forward(t, y0, right_hand_side_pendulum, **fargs)


@pytest.fixture
def result_pendulum_euler_backward_iterative():
    yield euler_backward_iterative(t, y0, right_hand_side_pendulum, residu_pendulum, residu_jacobian_pendulum, **fargs)


@pytest.fixture
def result_pendulum_solve_ivp():
    result = sp.integrate.solve_ivp(
        right_hand_side_solve_ivp,
        [0.0, 10.0],
        y0,
        t_eval=t,
        args=(fargs["mass"], fargs["length"], fargs["inertia tensor"]),
    )
    yield np.array([result.y[0], result.y[1]]).T


@pytest.fixture
def result_pendulum_odeint():
    yield sp.integrate.odeint(
        right_hand_side_odeint, y0, t, args=(fargs["mass"], fargs["length"], fargs["inertia tensor"])
    )


def wrapper_rhs_rk45(t, y):
    return right_hand_side_solve_ivp(t, y, *(fargs["mass"], fargs["length"], fargs["inertia tensor"]))


@pytest.fixture
def result_pendulum_rk45():
    rk = sp.integrate.RK45(
        wrapper_rhs_rk45,
        0.0,
        y0,
        10.0,
        max_step=1.0e-3,
    )

    times = [rk.t]
    states = [rk.y.copy()]

    while rk.status == "running":
        rk.step()
        times.append(rk.t)
        states.append(rk.y.copy())

    yield np.asarray(times), np.asarray(states)
