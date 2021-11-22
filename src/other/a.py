import numpy as np
from scipy.optimize import fsolve


def func(theta):
    theta2 = theta[0]
    theta3 = theta[1]

    re = 50 * np.cos(np.radians(30)) + 75 * np.cos(theta2) - 50 * np.cos(theta3) - 100
    im = 50 * np.sin(np.radians(30)) + 75 * np.sin(theta2) - 50 * np.sin(theta3) - 75

    return re, im

theta0 = [np.pi / 2, np.pi]

sol = fsolve(func, theta0, full_output=True)
print(np.degrees(sol[0]))
