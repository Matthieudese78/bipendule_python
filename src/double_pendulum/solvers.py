"""Numerical integration schemes."""

import numpy as np
import numpy.linalg as la

TOLERANCE_CONSTANT = 1.0e-6


def euler_forward(t: np.ndarray, y0: float | np.ndarray, f: callable, **fargs: dict) -> np.ndarray:
    """Integrates an ivp solution.

    Args:
        t: time vector
        y0 : initial values
        f : right hand side
        fargs : arguments of the right hand side callable.

    Returns:
        time integrated serie.

    """
    h = t[1] - t[0]
    y = y0.copy()
    result = np.zeros((len(t), y0.shape[0]))
    for i, step in enumerate(t):
        y += f(y, **fargs) * h
        result[i] = y
    return result


def euler_backward_iterative(
    t: np.ndarray,
    y0: float | np.ndarray,
    f: callable,
    residu: callable,
    residu_jacobian: callable,
    tol: float = 1.0e-15,
    max_iter: int = 500,
    **fargs: dict,
) -> np.ndarray:
    """Integrates an ivp solution.

    Args:
        t : time array
        y0 : initial conditions
        f : right hand side
        residu : computes the residu
        residu_jacobian : computes the residu jacobian
        fargs : arguments of the right hand side, residu & residu_jacobian callables.

    Returns:
        the result array (time, position, velocity)
    """
    result = np.zeros((len(t), y0.shape[0]))
    h = t[1] - t[0]
    y = y0.copy()
    for i, step in enumerate(t):
        # Euler Backward :
        # initialization newton raphson : one step of euler forward
        ypred = y + h * f(y, **fargs)
        res = residu(y, ypred, h, **fargs)
        crit = la.norm(res)
        niter = 0
        print(f"crit initial = {crit}")
        tol = TOLERANCE_CONSTANT * h * la.norm(f(ypred, **fargs))
        while (crit > tol) and (niter < max_iter):
            jac = residu_jacobian(ypred, h, **fargs)
            delta_res = -np.linalg.solve(jac, res)
            ypred += delta_res
            res = residu(y, ypred, h, **fargs)
            crit = la.norm(res)
            tol = TOLERANCE_CONSTANT * h * la.norm(f(ypred, **fargs))
            niter += 1

        print(f"step {i}, nb iterations = {niter}")
        print(f"          crit final = {crit}")

        y = ypred
        result[i] = y
    return result


# Singularity at theta = pi/2 --> we keep it to illustrate
def euler_backward_direct(
    t: np.ndarray,
    y0: float | np.ndarray,
    f: callable,
    finv: callable,
    **fargs: dict,
) -> np.ndarray:
    """Integrates an ivp solution.

    Args:
        t : time array
        y0 : initial conditions
        f : right hand side
        residu : computes the residu
        residu_jacobian : computes the residu jacobian
        fargs : arguments of the right hand side, residu & residu_jacobian callables.

    Returns:
        the result array (time, position, velocity)
    """
    result = np.zeros((len(t), y0.shape[0]))
    h = t[1] - t[0]
    y = y0.copy()
    for i, step in enumerate(t):
        # Euler Backward :
        ypred = y + h * f(y, **fargs)
        y = np.dot(np.eye(y.shape[0]) - h * finv(ypred, **fargs), ypred)
        result[i] = y
    return result
