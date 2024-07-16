from quantum.core import DensityMatrix, Operator
from quantum_network.runge_kutta import rungeKutta
from quantum_network.dynamics import AnalyticDynamicsFunc, DerivativeDynamicsFunc

import numpy as np
import scipy


class DynamicsManager:
    def __init__(self):
        self.dynamic_funcs = []

    def addDynamics(self, dynamics):
        self.dynamic_funcs.append(dynamics)

    def removeDynamics(self, dynamics):
        self.dynamic_funcs.remove(dynamics)

    def evolve(self, time: float):
        pass


class AnalyticDynamicsManager(DynamicsManager):

    def u(self, time):
        H = 0
        for dynamic in self.dynamic_funcs:
            H += dynamic.hamiltonian
        u_t = Operator(scipy.linalg.expm(-1j * H.matrix * time))
        return u_t

    def setTimestep(self, time):
        self.u_t = self.u(time)
        self.u_t_dagger = self.u_t.hermConj()
        self.timestep = time

    def evolve(self, state: DensityMatrix):
        return self.u_t * state * self.u_t_dagger


class NumericalDynamicsManager(DynamicsManager):
    def evolve(self, state: DensityMatrix):
        return rungeKutta(self.computeDerivative, self.timestep, state)

    def addDynamics(self, dynamics):
        super().addDynamics(dynamics)
        self.computeH()

    def computeH(self):
        self.H = 0
        for dynamic in self.dynamic_funcs:
            if type(dynamic).__base__ == AnalyticDynamicsFunc:
                self.H += dynamic.hamiltonian

    def setTimestep(self, t):
        self.timestep = t

    def computeDerivative(self, state):
        derivative = 0

        if self.H != 0:
            derivative = -1j * self.H.commutator(state)

        for func in self.dynamic_funcs:
            if type(func).__base__ == DerivativeDynamicsFunc:
                derivative += func.calcDerivative(state)

        return derivative
