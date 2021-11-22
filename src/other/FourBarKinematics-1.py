# from matplotlib.animation import FuncAnimation
# from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

#################Definition of the Four Bar Linkage
r1 = 15  # The crank
r2 = 12  # The coupler
r3 = 10  # The follower
r4 = 10  # The fixed link
theta4 = 0  # The orientation of the fixed link
N = 200  # Number of simulation points

#######################################################
# Initializing
##########################################
t1 = np.linspace(0, 360 * np.pi / 180, N)  # crank rotation range

# initializing x and y coordinates of all the links
x1 = np.zeros((t1.shape[0]))
x2 = np.zeros((t1.shape[0]))
x3 = np.zeros((t1.shape[0]))
x4 = np.zeros((t1.shape[0]))
y1 = np.zeros((t1.shape[0]))
y2 = np.zeros((t1.shape[0]))
y3 = np.zeros((t1.shape[0]))
y4 = np.zeros((t1.shape[0]))
t2 = np.zeros((t1.shape[0]))
t3 = np.zeros((t1.shape[0]))

# initializing the figure and plotting parameters
fig, ax = plt.subplots()


def plot_initialize():
    plt.xlim(-max(r3, r1), r1 + r2)
    plt.ylim(-max(r3, r1), max(r1, r3))
    plt.gca().set_aspect('equal', adjustable='box')


# Define the set of nonlinear equations that need to be solved
def func(theta, theta1, theta4, r1, r2, r3, r4):
    theta2 = theta[0]
    theta3 = theta[1]
    re = r1 * np.cos(theta1) + r2 * np.cos(theta2) - r3 * np.cos(theta3) - r4 * np.cos(theta4)  # Eq1
    im = r1 * np.sin(theta1) + r2 * np.sin(theta2) - r3 * np.sin(theta3) - r4 * np.sin(theta4)  # Eq2
    return (re, im)


i = 0
fr = 0
for theta1 in t1:  # for the range of input crank equations
    if i > 1:
        theta0 = [t2[i - 1], t3[i - 1]]
        # theta0=[0,0]#theta2 and theta3 initial guesses are assigned to the previous iteration
    else:
        theta0 = [0, 0]
    sol = fsolve(func, theta0, args=(theta1, theta4, r1, r2, r3, r4),
                 full_output=True)  # nonlinear solver that solves Eq1 and Eq2
    exit_flag = sol[2]  # if exit_flag==1, then the solution has reached and the algorithm is successful
    theta2 = sol[0][0]
    theta3 = sol[0][1]
    t2[i] = theta2
    t3[i] = theta3

    if exit_flag == 1:  # evaluating the x and y coordinates of the solved problem
        x1[fr] = 0
        y1[fr] = 0
        x2[fr] = x1[fr] + r1 * np.cos(theta1)
        y2[fr] = y1[fr] + r1 * np.sin(theta1)
        x3[fr] = x2[fr] + r2 * np.cos(theta2)
        y3[fr] = y2[fr] + r2 * np.sin(theta2)
        x4[fr] = x1[fr] + r4 * np.cos(theta4)
        y4[fr] = y1[fr] + r4 * np.sin(theta4)
        fr = fr + 1
    # plt.plot([x1,x2,x3,x4,x1],[y1,y2,y3,y4,y1])
    i = i + 1

    if i == 1:
        line, = ax.plot([x1[fr], x2[fr], x3[fr], x4[fr], x1[fr]], [y1[fr], y2[fr], y3[fr], y4[fr], y1[fr]], 'r')


def animation_frame(p):
    line.set_data([x1[p], x2[p], x3[p], x4[p], x1[p]], [y1[p], y2[p], y3[p], y4[p], y1[p]])

    return line


ani = animation.FuncAnimation(fig, func=animation_frame, init_func=plot_initialize, frames=np.arange(0, fr),
                              interval=100, repeat=True)

plt.show()

# def func(E,V_0):
# s = sqrt(c_sqr * (1 - E / V_0))
# f = s / tan(s) + sqrt(c_sqr - s**2)
#    f = E**2 -V_0
#    return f

# VV=4.
# guess = 9
# sol=fsolve(func, guess, args=(VV),full_output=True)
