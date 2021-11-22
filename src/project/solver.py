from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Solver:
    def __init__(self, nodeCords, edgeConnections, modulusE, crossArea, displacementConstraints, appliedForce):
        # Problem Definition
        self.nodeCords = nodeCords  # Nodal Coordinates
        self.elemNodes = edgeConnections  # Element connectivity: near node and far node
        self.modE = modulusE  # Young's modulus
        self.Area = crossArea  # Cross section area
        self.DispCon = displacementConstraints  # Displacement constraints # node number, degree of freedom (x=1, y=2), displacement value
        self.Fval = appliedForce  # Applied force # node number, orientation (x=1, y=2), value

        self.__initialze()

    def __initialze(self):
        # Problem Initialization
        self.nELEM = self.elemNodes.shape[0]  # Number of elements
        self.nNODE = self.nodeCords.shape[0]  # Number of nodes
        self.nDC = self.DispCon.shape[0]  # Number of constrained degrees of freedom
        self.nFval = self.Fval.shape[0]  # Number of degrees of freedom where forces are applied
        self.NDOF = self.nNODE * 2  # Total number of dofs = 2 times number of nodes
        self.uDisp = np.zeros((self.NDOF, 1))  # Initializing the displacement vector to 0
        self.forces = np.zeros((self.NDOF, 1))  # Initializing the force vector to 0
        self.Stiffness = np.zeros((self.NDOF, self.NDOF))  # Global stiffness matrix is assigned to 0
        self.Stress = np.zeros(self.nELEM)  # Stress is assigned to zero
        self.kdof = np.zeros(self.nDC)  # Degrees of freedom with known displacements
        self.xx = self.nodeCords[:, 0]  # X coordinate
        self.yy = self.nodeCords[:, 1]  # Y coordinate
        self.L_elem = np.zeros(self.nELEM)  # Length of each element is assigned to 0

        self.critical_loads = np.zeros((self.nELEM, 1))
        self.internal_forces = np.zeros((self.nELEM, 1))

    def __create_displacement_array(self):
        # Building the displacement array
        for i in range(self.nDC):  # looping over the number of known degrees of freedom
            indice = self.DispCon[i, :]
            v = indice[2]  # value of the known displacement
            v = v.astype(float)
            indice = indice.astype(int)

            index = self._find_index(indice)

            self.kdof[i] = index  # The corresponding degree of freedom that is constrained is assigned to kdof[i]
            self.uDisp[index] = v  # The corresponding displacement value is assigned to uDisp

    def __create_force_array(self):
        for i in range(self.nFval):  # looping over the dofs where forces are applied
            indice2 = self.Fval[i, :]
            v = indice2[2]
            v = v.astype(float)
            indice2 = indice2.astype(int)
            self.forces[self._find_index(indice2)] = v  # Assigning the value of the force in the forces vector

    def _find_index(self, index_array):
        return index_array[0] * 2 + index_array[1] - 1

    def solve(self):
        self.__initialze()
        self.__create_displacement_array()
        self.__create_force_array()

        self.kdof = self.kdof.astype(int)  # Contains all degrees of freedom with known displacement
        self.ukdof = np.setdiff1d(np.arange(self.NDOF),
                                  self.kdof)  # Contains all degrees of freedom with unknown displacement

        self.__calculate_global_stiffness()

        self.__partition_stiffness_matrix()
        self.__solve_unknown_dofs_and_forces()

        self.__evaluate_internal_forces_and_stresses()

        self.__calculate_critical_loads()

        return self.Stress, self.forces

    def plot(self, removed=None):
        self.__plot(removed=removed)
        plt.show()

    def __calculate_global_stiffness(self):
        # Loop over all the elements

        for e in range(self.nELEM):
            indiceE = self.elemNodes[e, :]  # Extracting the near and far node for element 'e'
            Y = self.modE[e]
            Ae = self.Area[e]
            elemDOF = np.array([indiceE[0] * 2, indiceE[0] * 2 + 1, indiceE[1] * 2,
                                indiceE[1] * 2 + 1])  # Contains all degrees of freedom for element 'e'
            elemDOF = elemDOF.astype(int)
            xa = self.xx[indiceE[1]] - self.xx[indiceE[0]]
            ya = self.yy[indiceE[1]] - self.yy[indiceE[0]]
            len_elem = np.sqrt(xa * xa + ya * ya)  # length of the element 'e'
            lx = xa / len_elem  # lambda x
            ly = ya / len_elem  # lambda y

            # Create stiffness matrix
            sub_ = np.array([[lx * lx, lx * ly],
                             [lx * ly, ly * ly]])
            K = np.empty((sub_.shape[0] * 2, sub_.shape[1] * 2))
            K[:sub_.shape[0], :sub_.shape[1]] = sub_
            K[sub_.shape[0]:, :sub_.shape[1]] = -sub_
            K[:sub_.shape[0], sub_.shape[1]:] = -sub_
            K[sub_.shape[0]:, sub_.shape[1]:] = sub_
            K *= (Ae * Y) / len_elem
            # Step 3. Assemble elemental stiffness matrices into a global stiffness matrix

            self.Stiffness[np.ix_(elemDOF, elemDOF)] += K

    def __partition_stiffness_matrix(self):
        self.k11 = self.Stiffness[np.ix_(self.ukdof, self.ukdof)]
        self.k12 = self.Stiffness[np.ix_(self.ukdof, self.kdof)]
        self.k21 = self.k12.transpose()
        self.k22 = self.Stiffness[np.ix_(self.kdof, self.kdof)]

    def __solve_unknown_dofs_and_forces(self):
        self.f_known = self.forces[self.ukdof] - np.matmul(self.k12, self.uDisp[self.kdof])
        self.uDisp[np.ix_(self.ukdof)] = np.linalg.solve(self.k11, self.f_known)

        self.forces[np.ix_(self.kdof)] = np.matmul(self.k21, self.uDisp[np.ix_(self.ukdof)]) + \
                                         np.matmul(self.k22, self.uDisp[np.ix_(self.kdof)])

    def __evaluate_internal_forces_and_stresses(self):
        for e in range(self.nELEM):
            indiceE = self.elemNodes[e, :]
            Y = self.modE[e]
            Ae = self.Area[e]
            elemDOF = np.array([indiceE[0] * 2, indiceE[0] * 2 + 1, indiceE[1] * 2, indiceE[1] * 2 + 1])
            elemDOF = elemDOF.astype(int)
            xa = self.xx[indiceE[1]] - self.xx[indiceE[0]]
            ya = self.yy[indiceE[1]] - self.yy[indiceE[0]]
            len_elem = np.sqrt(xa * xa + ya * ya)
            self.L_elem[e] = len_elem
            lx = xa / len_elem
            ly = ya / len_elem

            # Elemental Stiffness Matrix
            ke = (Ae * Y / len_elem) * np.array([[1., -1.], [-1., 1.]])

            # Transformation Matrix
            T = np.array([[lx, ly, 0, 0], [0, 0, lx, ly]])

            # Internal forces
            Fint = np.matmul(ke, np.matmul(T, self.uDisp[np.ix_(elemDOF)]))

            self.internal_forces[e] = Fint[1]
            # Stress
            self.Stress[e] = Fint[1] / Ae

    def __plot(self, removed=None):
        # Plotting and display
        # plt.xlim(min(self.xx) - abs(min(self.xx) / 10), max(self.xx) + abs(max(self.xx) / 10))
        # plt.ylim(min(self.yy) - abs(min(self.yy) / 10), max(self.yy) + abs(max(self.xx) / 10))

        r = .1

        for e in range(self.nELEM):
            indiceE = self.elemNodes[e, :]
            plt.plot(np.array([self.xx[indiceE[0]], self.xx[indiceE[1]]]),
                     np.array([self.yy[indiceE[0]], self.yy[indiceE[1]]]))

            s = np.array([self.xx[indiceE[0]], self.yy[indiceE[0]]])
            e_ = np.array([self.xx[indiceE[1]], self.yy[indiceE[1]]])

            p = (s + e_) / 2

            if removed is not None and e >= removed:
                e+= 1

            plt.text(p[0], p[1], e, ha="center", va="center",
                     bbox=dict(boxstyle=f"circle,pad={r}", fc="wheat"))

            plt.plot(np.array(
                [self.xx[indiceE[0]] + self.uDisp[indiceE[0] * 2], self.xx[indiceE[1]] + self.uDisp[indiceE[1] * 2]]),
                np.array([self.yy[indiceE[0]] + self.uDisp[indiceE[0] * 2 + 1],
                          self.yy[indiceE[1]] + self.uDisp[indiceE[1] * 2 + 1]]),
                '--')

        for i, p in enumerate(self.nodeCords):
            plt.text(p[0] + 1.25, p[1], i, ha="center", va="center",
                     bbox=dict(boxstyle=f"square,pad={r}", fc="lightblue"))

        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        pduDisp = pd.DataFrame({'disp': self.uDisp[:, 0]})
        pdforces = pd.DataFrame({'forces': self.forces[:, 0]})
        pdStress = pd.DataFrame({'Stress': self.Stress})
        pdLen = pd.DataFrame({'Length': self.L_elem})
        pdCritical = pd.DataFrame({'critical': self.critical_loads.ravel(), "internal": self.internal_forces.ravel()})
        # Displaying the results
        plt.savefig(f"{datetime.now().strftime('%H-%M-%S')}.png", dpi=1000, orientation='portrait', bbox_inches='tight',
                    pad_inches=None)
        print(pduDisp)
        print(pdforces)
        print(pdStress)
        print(pdLen)
        print(pdCritical)

    def __calculate_critical_loads(self):
        for e in range(self.nELEM):
            A = self.Area[e]

            I = A ** 2 / 12

            self.critical_loads[e] = np.pi ** 2 * self.modE[e] * I / (self.L_elem[e] ** 2)


if __name__ == '__main__':
    nodeCords = np.array([[0.0, 0.0],
                          [-10.0 * np.cos(60.0 * np.pi / 180), -10.0 * np.sin(60.0 * np.pi / 180)],
                          [10.0 * np.cos(60.0 * np.pi / 180), -10.0 * np.sin(60.0 * np.pi / 180)]])
    # Nodal Coordinates
    elemNodes = np.array([[1, 0], [1, 2], [2, 0]])  # Element connectivity: near node and far node
    modE = np.array([[200e9], [200e9], [200e9]])  # Young's modulus
    Area = np.array([[4e-2], [4e-2], [4e-2]])  # Cross section area
    DispCon = np.array([[1, 1, 0.0], [1, 2, 0.0], [2, 2,
                                                   0.0]])  # Displacement constraints # node number, degree of freedom (x=1, y=2), displacement value
    Fval = np.array([[0, 2, 6e8]])  # Applied force # node number, orientation (x=1, y=2), value

    solver = Solver(nodeCords, elemNodes, modE, Area, DispCon, Fval)
    solver.solve()
    solver.plot()

    nodeCords = np.array([[0.0, 0.0],
                          [4, 4],
                          [8, 4],
                          [8, 0],
                          [4, 0]
                          ])
    # Nodal Coordinates
    elemNodes = np.array(
        [[0, 1], [1, 2], [2, 3], [3, 4], [0, 4], [4, 1], [1, 3],
         [0, 2]])  # Element connectivity: near node and far node
    modE = np.full((elemNodes.shape[0], 1), 29e6)  # Young's modulus
    modE[-1] = 0
    Area = np.array(
        [[2 / 144.], [2 / 144.], [3 / 144.], [3 / 144.], [3 / 144.], [3 / 144.], [2 / 144.], [1]])  # Cross section area
    DispCon = np.array([[4, 2, 0.0], [2, 1, 0.0], [2, 2,
                                                   0.0]])  # Displacement constraints # node number, degree of freedom (x=1, y=2), displacement value
    Fval = np.array([[0, 2, -7000.], [3, 2, -3000.]])  # Applied force # node number, orientation (x=1, y=2), value

    solver = Solver(nodeCords, elemNodes, modE, Area, DispCon, Fval)
    solver.solve()
    solver.plot()
