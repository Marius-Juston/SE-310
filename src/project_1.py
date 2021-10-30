import itertools
from functools import partial

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, Bounds

from solver import Solver


def optimize_v1(x0, E, W, D, xi=None, plot=True):
    nodeCords, elemNodes, modE, Area, DispCon, Fval = parameter_calculation(x0, E, W, D, xi)

    solver = Solver(nodeCords, elemNodes, modE, Area, DispCon, Fval)
    solver.solve()

    if plot:
        solver.plot()

    return calculate_c(x0, E, W, D, xi)


def parameter_calculation(x0, E=1, W=0, D=0, xi=None):
    if xi is None:
        node1_x, node2_x, node2_y, node3_y, cross_section_width = x0

        area = cross_section_width ** 2
    else:
        node1_x, node2_x, node2_y, node3_y = x0

        area = xi ** 2

    nodeCords = np.array([[node1_x, 0.0],
                          [node2_x, node2_y],
                          [0, node3_y],

                          [-node1_x, 0.0],
                          [-node2_x, node2_y],
                          ])
    # Nodal Coordinates
    elemNodes = np.array([[0, 1], [0, 2], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4],
                          [3, 4]])  # Element connectivity: near node and far node

    modE = np.full((elemNodes.shape[0], 1), E)  # Young's modulus
    Area = np.full((elemNodes.shape[0], 1), area)  # cross section area

    # Displacement constraints # node number, degree of freedom (x=1, y=2), displacement value
    DispCon = np.array([[0, 1, 0.0], [0, 2, 0.0], [3, 1, 0.0],
                        [3, 2,
                         0.0]])

    Fval = np.array([[2, 2, -W], [2, 1, D]])  # Applied force # node number, orientation (x=1, y=2), value

    return nodeCords, elemNodes, modE, Area, DispCon, Fval


def stress_calculator(x0, E, W=0, D=0, xi=None):
    nodeCords, elemNodes, modE, Area, DispCon, Fval = parameter_calculation(x0, E, W, D, xi)

    solver = Solver(nodeCords, elemNodes, modE, Area, DispCon, Fval)
    result, _ = solver.solve()

    return result


def critical_buckling(x0, E, W=0, D=0, safety_factor=1, xi=None):
    nodeCords, elemNodes, modE, Area, DispCon, Fval = parameter_calculation(x0, E, W, D, xi)

    solver = Solver(nodeCords, elemNodes, modE, Area, DispCon, Fval)
    solver.solve()
    internal = -solver.internal_forces
    critical = solver.critical_loads
    mask = internal < 0

    i_ = internal[mask]
    critical = critical[mask]

    results = np.ones_like(internal)
    results[mask] = critical - i_ / safety_factor

    return results.ravel()


def node_distance(x0, E=1, W=0, D=0, xi=None):
    nodeCords, elemNodes, modE, Area, DispCon, Fval = parameter_calculation(x0, E, W, D, xi)

    t = [i for i in range(len(nodeCords))]
    c = list(itertools.combinations(t, 2))

    lengths = []

    for i, f in c:
        l_ = np.linalg.norm(nodeCords[i] - nodeCords[f])
        lengths.append(l_)

    return lengths


def calculate_c(x0, E=1, W=0, D=0, xi=None):
    nodeCords, elemNodes, modE, Area, DispCon, Fval = parameter_calculation(x0, E, W, D, xi)

    sum_ = 0

    for i in range(elemNodes.shape[0]):
        first, last = elemNodes[i]
        l_i = np.linalg.norm(nodeCords[first] - nodeCords[last])
        a_i = Area[i]

        sum_ += a_i * l_i

    return nodeCords.shape[0] * sum_


def test():
    W = 80 * 1000
    D = 78771
    E = 200 * 1e9
    MaxYield = 600 * 1e6

    x = [10, 8, 20, 50, 1]
    output = optimize_v1(x, E, W, D)
    print(output)


def slenderness_ratio(x0, E=1, W=0, D=0, xi=None):
    nodeCords, elemNodes, modE, Area, DispCon, Fval = parameter_calculation(x0, E, W, D, xi)

    slenderness = []

    for i in range(elemNodes.shape[0]):
        first, last = elemNodes[i]
        l_i = np.linalg.norm(nodeCords[first] - nodeCords[last])
        a_i = Area[i]
        I = a_i ** 2 / 12

        r = np.sqrt(I / a_i)

        slenderness.append((l_i / r)[0])

    return slenderness


def scipy_optimization(xi=None):
    W = 80 * 1000
    D = 78771
    E = 200 * 1e9
    MaxYield = 600 * 1e6
    safety_factor = 4

    # x0 = [5, 4, 25, 50, .1]
    x0 = [2.5, 2.5, 25, 50, .1]
    # x0 = [3.70040327, 1.12946584, 7.8710686, 50.35919667, 0.08178408]
    # x0 = [9.96818480e+00, 7.17917660e+00, 1.12287710e+01, 5.00110795e+01, 2.45318497e-02]  # 0.78044092
    # x0 = [5.03192112e+00, 4.12913951e+00, 2.49909974e+01, 5.00320637e+01, 2.24441655e-02]  # 0.66200234
    # x0 = [3.32076413,  2.10502301, 21.73408048, 50.01674174,  0.34744511]  # 0.66200234

    x0 = [2.55758293, 1.47541133, 22.74026144, 50.01267, 0.34720454]  # 150.35943793
    x0 = [9., 9., 25., 50., .3]

    l_b = [2.5, 0, 0, 50, 0]
    u_b = [10, np.inf, np.inf, np.inf, np.inf]

    if xi is None:
        bounds = Bounds(l_b, u_b)
    else:
        x0 = x0[:-1]

        bounds = Bounds(l_b[:-1], u_b[:-1])

    nlc = NonlinearConstraint(partial(stress_calculator, E=E, W=W, D=D, xi=xi), -MaxYield, MaxYield, jac='3-point')

    nlc2 = NonlinearConstraint(partial(critical_buckling, E=E, W=W, D=D, safety_factor=safety_factor, xi=xi), 0, np.inf,
                               jac='3-point')
    #
    nlc3 = NonlinearConstraint(partial(node_distance, E=E, W=W, D=D, xi=xi), 2, np.inf, jac='3-point')

    nlc4 = NonlinearConstraint(partial(slenderness_ratio, E=E, W=W, D=D, xi=xi), 0, 500, jac='3-point')

    print("------------ INITIAL ------------")
    print(f'{x0=}')
    print(optimize_v1(x0, E, W, D, xi=xi, plot=False))
    print(node_distance(x0, E, W, D, xi=xi))
    print(slenderness_ratio(x0, E, W, D, xi=xi))

    res = minimize(partial(calculate_c, E=E, W=W, D=D, xi=xi), x0, method='trust-constr', bounds=bounds,
                   constraints=[nlc, nlc2, nlc3, nlc4],
                   options={'verbose': 1, 'maxiter': 1 * 1e5}, jac='3-point')

    with open('../out.txt', 'w') as f:
        f.write(str(res))

    print("------------ FINAL ------------")
    print(f'{res.x=}')
    print(optimize_v1(res.x, E, W, D, xi=xi))
    print(node_distance(res.x, E, W, D, xi=xi))


if __name__ == '__main__':
    # xi = 2.30056943e-02
    xi = None

    scipy_optimization(xi=xi)
