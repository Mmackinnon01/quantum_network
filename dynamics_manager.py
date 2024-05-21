from quantum.core import DensityMatrix, Operator
from quantum_network.runge_kutta import rungeKutta

import numpy as np
import scipy


class DynamicsManager:
    def __init__(self, timestep):
        self.timestep = timestep
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

    def precompute_u(self, time):
        self.u_t = self.u(time)
        self.u_t_dagger = self.u_t.hermConj()

    def evolve(self, state: DensityMatrix):
        return self.u_t * state * self.u_t_dagger


class NumericalDynamicsManager(DynamicsManager):
    def evolve(self, state: DensityMatrix, time: float):
        total_time = 0
        while total_time < time:
            state = rungeKutta(self.computeDerivative, self.timestep, state)
            total_time += self.timestep
        return state

    def computeDerivative(self, state):
        derivative = 0
        for func in self.dynamic_funcs:
            derivative += func.calcDerivative(state)

        return derivative


class ReducedNumericalDynamicsManager(DynamicsManager):
    def evolve(self, state: DensityMatrix, time: float):
        total_time = 0

        self.H = 0
        for dynamic in self.dynamic_funcs:
            self.H += dynamic.hamiltonian

        while total_time < time:
            state = rungeKutta(self.computeDerivative, self.timestep, state)
            total_time += self.timestep
        return state

    def computeDerivative(self, state):
        return -1j * self.H.commutator(state)
