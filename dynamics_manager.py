from quantum.core import DensityMatrix, Operator
from quantum_network.runge_kutta import rungeKutta
from quantum_network.dynamics import UnitaryDynamics, NonUnitaryDynamics

import numpy as np
import scipy


class DynamicsManager:
    def __init__(self):
        self.dynamic_funcs = []
        self.unitary = True
        self.u_t = None

    def addDynamics(self, dynamics):
        self.dynamic_funcs.append(dynamics)
        self.unitary = self.isUnitary()
        self.u_t = None

    def removeDynamics(self, dynamics):
        self.dynamic_funcs.remove(dynamics)
        self.unitary = self.isUnitary()
        self.u_t = None

    def isUnitary(self):
        for func in self.dynamic_funcs:
            if not issubclass(func.__class__, UnitaryDynamics):
                return False
        return True

    def u(self, time):
        H = 0
        for dynamic in self.dynamic_funcs:
            H += dynamic.H
        u_t = Operator(scipy.linalg.expm(-1j * H.matrix * time))
        return u_t
    
    def setTimestep(self, t):
        self.timestep = t

    def evolve(self, state: DensityMatrix):
        if self.unitary:
            return self.unitaryEvolution(state)
        else:
            return self.nonUnitaryEvolution(state)

    def unitaryEvolution(self, state: DensityMatrix):
        if self.u_t is None:
            self.u_t = self.u(self.timestep)
            self.u_t_dagger = self.u_t.hermConj()
        return self.u_t * state * self.u_t_dagger
    
    def nonUnitaryEvolution(self, state: DensityMatrix):
        return rungeKutta(self.computeDerivative, self.timestep, state)
    
    def computeDerivative(self, state):
        derivative = 0
        unitary_component = 0

        for func in self.dynamic_funcs:
            if issubclass(func.__class__, NonUnitaryDynamics):
                derivative += func.calcDerivative(state)
            elif issubclass(func.__class__, UnitaryDynamics):
                unitary_component += func.H

        derivative += -1j * unitary_component.commutator(state)

        return derivative
