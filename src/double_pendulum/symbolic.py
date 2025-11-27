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

    cq = sp.Matrix([[dcxdx, dcxdy, dcxdtheta], [dcydx, dcydy, dcydtheta]])
    # alternative, shorter version :
    c = sp.Matrix([cx, cy])
    jc = c.jacobian([x, y, theta])
    print(f"jc = {jc}")
    print(f"jc.T = {jc.T}")
    # return sp.lambdify((x, y, theta), jc, "numpy")
    return jc, sp.lambdify((x, y, theta), jc, "numpy")




# %%
if __name__ == "__main__":
    cq2 = pendulum_cq()
    print(f"cq2 = {cq2}")

# %%
