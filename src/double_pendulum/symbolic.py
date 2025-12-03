"""Symbolic calulus for constraints and derivatives."""

# %%
from typing import Callable

import numpy as np
import sympy as sp


# %% Lagrange multipliers :
def pendulum_cq(l1: int) -> Callable[np.ndarray, np.ndarray]:
    """Computation of the constraint jacobian.

    Args:
    l1: pendulum length

    Returns:
    Callable
    """
    # Holonomic Constraints: sqrt(x^2 + y^2) = l
    x, y, theta = sp.symbols("x y theta")
    sp.init_printing(use_unicode=True)
    f = sp.simplify(sp.sqrt(x**2 + y**2))
    print(f)
    dfdx = sp.diff(f, x)
    dfdy = sp.diff(f, y)
    print(dfdx)
    print(dfdy)

    # Dependent and independent variables :
    # Theta is the independent variable
    # x , y are the dependent variables
    # Constraints (should be equal to zero):
    cx = sp.simplify(x - l1 * sp.cos(theta))
    cy = sp.simplify(y - l1 * sp.sin(theta))
    # ctheta = 0
    # Ordinating the variables : x, y, theta
    # Gradient of the constraint (d Ci/ d Xj)
    # C_q = [[ dCx/dx , dCx/dy, dCx/dtheta],
    #        [ dCy/dx , dCy/dy, dCy/dtheta]]
    dcxdx = sp.diff(cx, x)
    print(dcxdx)
    dcxdy = sp.diff(cx, y)
    print(dcxdy)
    dcxdtheta = sp.diff(cx, theta)
    print(dcxdtheta)

    dcydx = sp.diff(cy, x)
    print(dcydx)
    dcydy = sp.diff(cy, y)
    print(dcydy)
    dcydtheta = sp.diff(cy, theta)
    print(dcydtheta)

    # dcthetadx = 0
    # dcthetady = 0
    # dcthetadtheta = 0

    cq = sp.Matrix([[dcxdx, dcxdy, dcxdtheta], [dcydx, dcydy, dcydtheta]])

    # alternative, shorter version :
    c = sp.Matrix([cx, cy])
    jc = c.jacobian([x, y, theta])
    print(f"jc = {jc}")
    print(f"jc.T = {jc.T}")
    return jc, sp.lambdify((x, y, theta), jc, "numpy")


# %%
# def pendulum_qc(cq):
#     """Computes Qc for holonomic constraints (Shabana edition 1989 p. 166, 293 & p.404).

#     Args:
#     cq: the constraint jacobian matrix.

#     Returns:
#     qc : the generalized reaction forces.
#     """
#     x, y, theta, dxdt, dydt, dthetadt = sp.symbols("x y theta dxdt dydt dthetadt")

#     qdot = sp.Matrix([dxdt, dydt, dthetadt])

#     qc1 = sp.simplify(cq @ qdot)
#     qc2 = qc1.jacobian([x, y, theta])
#     qc = sp.simplify(qc2 @ qdot)
#     print(f"qc = {qc}")
#     return qc, sp.lambdify((x, y, theta, dxdt, dydt, dthetadt), qc, "numpy")


def pendulum_qc(l1):
    """Computes Qc for holonomic constraints (Shabana edition 1989 p. 166, 293 & p.404).

    Args:
    cq: the constraint jacobian matrix.

    Returns:
    qc : the generalized reaction forces.
    """
    theta, dthetadt = sp.symbols("theta dthetadt")
    qc = sp.Matrix([l1 * dthetadt**2 * sp.cos(theta), -l1 * dthetadt**2 * sp.sin(theta)])
    return qc, sp.lambdify((theta, dthetadt), qc, "numpy")


# %%
if __name__ == "__main__":
    l1 = 1.0
    cq2 = pendulum_cq(l1)[0]
    # %%
    # qc = pendulum_qc(cq2)
    qc = pendulum_qc(l1)[0]
    print(qc)
    # %%
    arg = np.zeros(6)
    print(arg)
    cq3 = pendulum_cq(l1)[1](arg[0], arg[1], arg[2])
    print(cq3)
    print(f"shape cq3 = {np.shape(cq3)}")
    # qc = pendulum_qc(cq2)[1](arg[0], arg[1], arg[2], arg[3], arg[4], arg[5])
    qc = pendulum_qc(l1)[1](arg[0], arg[1])
    print(f"shape qc = {np.shape(qc.ravel())}")
    print(f"qc = {qc.ravel()}")

# %%
