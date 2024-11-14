import numpy as np
import copy
from quantum.core import (
    DensityMatrix,
    Operator,
    sigmaX,
    sigmaY,
    sigmaZ,
    sigmaMinus,
    sigmaPlus,
    lambda1,
    lambda2,
    lambda6,
    lambda7,
)


class Dynamics:

    def __init__(self):
        self.operators = []

    def updateOperators(self, config, dims):
        for operator in self.operators:
            operator.transform(copy.deepcopy(config), copy.deepcopy(dims))


class UnitaryDynamics(Dynamics):
    def __init__(self):
        self.H = None

    def calcDerivative(self, ro: DensityMatrix) -> DensityMatrix:
        ro_dot = -1j * self.H.commutator(ro)
        return ro_dot


class NonUnitaryDynamics(Dynamics):
    def calcDerivative(self, ro: DensityMatrix):
        pass


class EnergyExchangeDynamics(UnitaryDynamics):
    def __init__(self, system1, system2, coupling_strength: float):
        super().__init__()
        self.system1 = system1
        self.system2 = system2
        self.systems = [self.system1, self.system2]
        self.coupling_strength = coupling_strength
        self.H = self.coupling_strength * (
            sigmaX.tensor(sigmaX) + sigmaY.tensor(sigmaY)
        )
        self.operators = [self.H]


class LocalDecoherenceDynamics(NonUnitaryDynamics):
    def __init__(self, system, gamma: float):
        super().__init__()
        self.system = system
        self.systems = [self.system]
        self.gamma = gamma
        self.sigma_plus = sigmaPlus
        self.sigma_minus = sigmaMinus
        self.operators = [self.sigma_minus, self.sigma_plus]

    def calcDerivative(self, ro: DensityMatrix):
        return self.gamma * (
            2 * self.sigma_minus * ro * self.sigma_plus
            - ro * self.sigma_plus * self.sigma_minus
            - self.sigma_plus * self.sigma_minus * ro
        )


class DampedCascadeFunction(NonUnitaryDynamics):
    def __init__(self, system1, system2, gamma_1, gamma_2):
        super().__init__()
        self.system1 = system1
        self.system2 = system2
        self.systems = [self.system1, self.system2]
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.sigma_plus_1 = sigmaPlus.tensor(Operator(np.eye(2)))
        self.sigma_plus_2 = Operator(np.eye(2)).tensor(sigmaPlus)
        self.sigma_minus_1 = sigmaMinus.tensor(Operator(np.eye(2)))
        self.sigma_minus_2 = Operator(np.eye(2)).tensor(sigmaPlus)
        self.operators = [self.sigma_plus_1, self.sigma_plus_2, self.sigma_minus_1, self.sigma_minus_2]

    def calcDerivative(self, ro: DensityMatrix):

        term1 = (self.gamma_1) * (
            2 * self.sigma_minus_1 * ro * self.sigma_plus_1
            - ro * self.sigma_plus_1 * self.sigma_minus_1
            - self.sigma_plus_1 * self.sigma_minus_1 * ro
        )
        term2 = 0.1 * (self.gamma_2) * (
            2 * self.sigma_minus_2 * ro * self.sigma_plus_2
            - ro * self.sigma_plus_2 * self.sigma_minus_2
            - self.sigma_plus_2 * self.sigma_minus_2 * ro
        )
        term3 = - ((self.gamma_1 * self.gamma_2) ** 0.5) * self.sigma_plus_2.commutator(self.sigma_minus_1 * ro)
        term4 = - ((self.gamma_1 * self.gamma_2) ** 0.5) * (ro * self.sigma_plus_1).commutator(self.sigma_minus_2)
        return term1 + term2 + term3 + term4


class QutritQubitExchangeDynamics(UnitaryDynamics):
    def __init__(self, system1, system2, coupling_strength: float):
        super().__init__()
        self.system1 = system1
        self.system2 = system2
        self.systems = [self.system1, self.system2]
        self.coupling_strength = coupling_strength
        self.H = self.coupling_strength * (
            (lambda1 + lambda6).tensor(sigmaX) + (lambda2 + lambda7).tensor(sigmaY)
        )
        self.operators = [self.H]


class LevelCouplingDynamics(UnitaryDynamics):
    def __init__(
        self,
        system1,
        system2,
        coupling_levels_sys_1: list,
        coupling_levels_sys_2: list,
        coupling_strength: float,
    ):
        super().__init__()
        self.system1 = system1
        self.system2 = system2
        self.systems = [system1, system2]
        self.coupling_strength = coupling_strength
        dim = self.system1.dim * self.system2.dim
        h = np.zeros((dim, dim)).astype("complex")
        i = coupling_levels_sys_1[0] * self.system2.dim + coupling_levels_sys_2[1]
        j = coupling_levels_sys_1[1] * self.system2.dim + coupling_levels_sys_2[0]
        h[i][j] = 1
        h[j][i] = 1
        self.H = coupling_strength * (Operator(h))
        self.operators = [self.H]


class QuditExchangeDynamics(UnitaryDynamics):
    def __init__(self, system1, system2, coupling_strength: float):
        super().__init__()
        self.system1 = system1
        self.system2 = system2
        self.systems = [system1, system2]
        self.coupling_strength = coupling_strength
        dim = self.system1.dim * self.system2.dim
        h = np.random.rand(dim, dim).astype("complex") + 1j * np.random.rand(
            dim, dim
        ).astype("complex")
        h = np.matmul(h, np.conjugate(h.T))
        h = Operator(h)
        self.H = coupling_strength * (h)
        self.operators = [self.H]


class QuditEnergyDynamics(UnitaryDynamics):
    def __init__(self, system, eigen_values=None):
        super().__init__()
        self.system = system
        self.systems = [self.system]
        self.H = np.eye(self.system.dim)
        if eigen_values is not None:
            if len(eigen_values) != system.dim:
                raise ValueError("Incorrect number of eigenvalues specified")
            for i, val in enumerate(eigen_values):
                self.H[i][i] = val
        else:
            for i in range(self.system.dim):
                self.H[i][i] = 1 - 2 * (i / (self.system.dim - 1))
        self.H = Operator(self.H)
        self.operators = [self.H]
