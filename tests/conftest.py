import pytest
import numpy as np


@pytest.fixture
def pendulum_setup():
    g = 9.81

    m1 = 1.0

    l1 = 1.0

    theta = -45.0 * np.pi / 180.0

    dthetadt0 = 0.0

    y0 = np.array([theta, dthetadt0])

    h = 0.1
    num_steps = 10000
    t = np.linspace(0.0, 10.0, num_steps)

    fargs = {"m": m1, "l": l1, "g": g}
