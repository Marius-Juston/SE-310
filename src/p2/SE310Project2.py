# from matplotlib.animation import FuncAnimation
# from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

#################Definition of the Four Bar Linkage
# The values are placeholder. Please change the values below.
# Please leave theta4=0 and N=360

a = 19.5  # The crank
b = 45.0  # The coupler
c = 60.0  # The rocker
d = 40.0  # The fixed link
e = 90.0  # The wing
theta4 = 0  # The orientation of the fixed link
N = 360  # Number of simulation points

#######################################################
# Initializing
##########################################
ox = 0  # coordinates of the start point
oy = -d
link_length = np.array([[a], [b], [c], [d]])
s = min(link_length)
l = max(link_length)
t1 = np.linspace(0, 360 * np.pi / 180, N)  # crank rotation range

if a + max(link_length) > b + c + d - max(link_length):
    print("Grashoff criterion is not satisfied")
else:
    print("Grashoff criterion is satisfied")

# initializing x and y coordinates of all the links
x1 = np.zeros((t1.shape[0]))
x2 = np.zeros((t1.shape[0]))
x3 = np.zeros((t1.shape[0]))
x4 = np.zeros((t1.shape[0]))
x5 = np.zeros((t1.shape[0]))
x6 = np.zeros((t1.shape[0]))
y1 = np.zeros((t1.shape[0]))
y2 = np.zeros((t1.shape[0]))
y3 = np.zeros((t1.shape[0]))
y4 = np.zeros((t1.shape[0]))
y5 = np.zeros((t1.shape[0]))
y6 = np.zeros((t1.shape[0]))
t2 = np.zeros((t1.shape[0]))
t3 = np.zeros((t1.shape[0]))
v_tip = np.zeros((t1.shape[0]))

# initializing the figure and plotting parameters
fig, ax = plt.subplots()


def plot_initialize():
    plt.xlim(-105, 105)
    plt.ylim(-105, 105)
    plt.gca().set_aspect('equal', adjustable='box')


# Define the set of nonlinear equations that need to be solved
def func(x, a, b, c, d, theta):
    theta2 = x[0]
    theta3 = x[1]
    re = a * np.cos(theta) + b * np.cos(theta2) + c * np.cos(theta3)  # Eq1
    im = a * np.sin(theta) + b * np.sin(theta2) + c * np.sin(theta3) - d  # Eq2
    return (re, im)


def vfunc(x, a, b, c, d, theta, x1, x2, w1):
    theta2 = x[0]
    theta3 = x[1]
    re = a * w1 * np.cos(theta) + b * theta2 * np.cos(x1) + c * theta3 * np.cos(x2)  # Eq1
    im = a * w1 * np.sin(theta) + b * theta2 * np.sin(x1) + c * theta3 * np.sin(x2)  # Eq2
    return (re, im)


w = 900 * 2 * np.pi / 60  # input velocity of the crank
y0 = [0, 0]
i = 0
fr = 0
for theta1 in t1:  # for the range of input crank equations
    if i > 1:
        x0 = [t2[i - 1], t3[i - 1]]
        # theta0=[0,0]#theta2 and theta3 initial guesses are assigned to the previous iteration
    else:
        x0 = [np.pi / 2, np.pi]
    sol = fsolve(func, x0, args=(a, b, c, d, theta1), full_output=True)  # nonlinear solver that solves Eq1 and Eq2
    exit_flag = sol[2]  # if exit_flag==1, then the solution has reached and the algorithm is successful
    theta2 = sol[0][0]
    theta3 = sol[0][1]
    t2[i] = theta2
    t3[i] = theta3

    # Finding the tip velocity
    sol2 = fsolve(vfunc, y0, args=(a, b, c, d, theta1, theta2, theta3, w),
                  full_output=True)  # nonlinear solver that solves Eq1 and Eq2
    exit_flag2 = sol2[2]  # if exit_flag==1, then the solution has reached and the algorithm is successful
    v1 = sol2[0][0]
    v2 = sol2[0][1]
    v_tip[i] = abs(v2 * e)

    if exit_flag == 1:  # evaluating the x and y coordinates of the solved problem
        x1[fr] = ox
        y1[fr] = oy
        x2[fr] = x1[fr] + a * np.cos(theta1)
        y2[fr] = y1[fr] + a * np.sin(theta1)
        x3[fr] = x2[fr] + b * np.cos(theta2)
        y3[fr] = y2[fr] + b * np.sin(theta2)
        x4[fr] = x1[fr]
        y4[fr] = y1[fr] + d
        x5[fr] = x4[fr] + e * np.cos(theta3)
        y5[fr] = y4[fr] + e * np.sin(theta3)
        x6[fr] = x4[fr] - e * np.cos(theta3)
        y6[fr] = y4[fr] - e * np.sin(theta3)

        fr = fr + 1
    # plt.plot([x1,x2,x3,x4,x1],[y1,y2,y3,y4,y1])
    i = i + 1

    if i == 1:
        line, = ax.plot([x1[fr], x2[fr], x3[fr], x5[fr], x6[fr], x4[fr], x1[fr]],
                        [y1[fr], y2[fr], y3[fr], y5[fr], y6[fr], y4[fr], y1[fr]], 'r')

# Range of angular motion of the wing
theta3_range = (max(t3) - min(t3)) * 180 / np.pi
print('range = ')
print(theta3_range)

tip_vel = max(v_tip)

print('tip_vel=')
print(tip_vel)


def animation_frame(p):
    line.set_data([x1[p], x2[p], x3[p], x5[p], x6[p], x4[p], x1[p]], [y1[p], y2[p], y3[p], y5[p], y6[p], y4[p], y1[p]])

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
