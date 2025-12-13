"""Post-treatment of use cases."""

import matplotlib.pyplot as plt
import numpy as np

GRAVITY = 9.81


def post_treatment_pendulum(result: np.ndarray, length: float, mass: float, t: np.ndarray) -> int:
    """Postreatment of a pendulum tme integration.

    Parameters:
        result : time series of y = (angle, angular velocity)

    Returns:
        int : error code 0
    """
    J = mass * length**2
    g = GRAVITY
    theta = result[:, 0]
    dthetadt = result[:, 1]
    x = length * np.cos(result[:, 0])
    y = length * np.sin(result[:, 0])
    x_dot = -result[:, 1] * length * np.sin(result[:, 0])
    y_dot = result[:, 1] * length * np.cos(result[:, 0])

    plt.scatter(t, theta * 180.0 / np.pi, s=4)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\theta$")
    plt.show()

    plt.scatter(t, dthetadt, s=4)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\omega = \dot{\theta}$")
    plt.show()

    plt.scatter(t, x, s=4, label="x")
    plt.scatter(t, y, s=4, label="y")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x,y$")
    plt.legend()
    plt.show()

    plt.scatter(x, y, s=4)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title("Trajectory")
    plt.show()
    # T, K, Lagrangian, Hamiltonian :
    T = 0.5 * J * result[:, 1] ** 2
    # T = 0.5 * mass * np.square(x_dot**2 + y_dot**2)
    K = mass * g * np.sin(result[:, 0])
    H = T + K
    L = T - K
    plt.scatter(t, T, s=4, color="r", label="Kinetic Energy")
    plt.scatter(t, K, s=4, color="g", label="Potential Energy")
    plt.scatter(t, H, s=4, color="b", label="Hamiltonian")
    plt.scatter(t, L, s=4, color="orange", label="Lagrangian")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$T,K,H,L$")
    plt.legend()
    plt.show()

    return 0
