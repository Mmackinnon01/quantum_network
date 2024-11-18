from quantum.core import DensityMatrix
from quantum_network.dynamics_manager import DynamicsManager

import math
import numpy as np
import tqdm
import multiprocessing
import copy


class System:

    def __init__(self, logger=False, numerical=True):
        self.dynamics = []
        self.name = None
        self.t = 0
        self.dynamics_manager = DynamicsManager()
        if logger:
            self.logger = StateLogger()
        else:
            self.logger = None

    @property
    def state(self):
        return self.__state

    @state.setter
    def state(self, ro: DensityMatrix):
        self.__state = ro

        if self.logger is not None:
            self.logger.record(self.t, copy.deepcopy(ro))

    @property
    def configuration(self):
        raise NotImplementedError

    def addDynamics(self, dynamicFunc):
        self.dynamics.append(dynamicFunc)

    def getDynamics(self):
        raise NotImplementedError

    def removeDynamics(self, dynamicFunc):
        return self.dynamics.remove(dynamicFunc)


class SingleSystem(System):

    def __init__(self, state=None, logger=False, numerical=True):
        super().__init__(logger, numerical)
        self.state = state
        self.nsystems = 1

    def __repr__(self):
        return f"Single Quantum System: {self.name}"

    @property
    def state(self):
        return self.__state

    @state.setter
    def state(self, ro: DensityMatrix):
        if ro is not None:
            self.__dim = ro.dim
        self.__state = ro

        if self.logger is not None:
            self.logger.record(self.t, copy.deepcopy(ro))

    def computeState(self):
        return self.state

    def getSubsystem(self, name: str):
        return None

    @property
    def dim(self):
        return self.__dim

    @dim.setter
    def dim(self, dim):
        self.__dim = dim

    @property
    def configuration(self):
        return [self.dim]

    def getDynamics(self):
        return self.dynamics

    def updateDynamics(self):
        for dynamicFunc in self.getDynamics():
            dynamicFunc.updateOperators([0], self.dim)


class CompositeSystem(System):

    def __init__(self, logger=False, numerical=True):
        super().__init__(logger, numerical)
        self.subsystems = []
        self.nsystems = 0
        self.__state = DensityMatrix(matrix=np.array([[1]]))

    def __repr__(self):
        return f"Composite Quantum System: {self.name}"

    @property
    def configuration(self):
        config = []
        for system in self.subsystems:
            config += system.configuration
        return config

    @property
    def dim(self):
        return math.prod([system.dim for system in self.subsystems])

    @property
    def state(self):
        return self.__state

    @state.setter
    def state(self, ro):
        for system in self.subsystems:
            system.state = None
        if ro is not None:
            ro.configuration = self.configuration
        self.__state = ro

        if self.logger is not None:
            self.logger.record(self.t, copy.deepcopy(ro))

    def computeState(self):
        if self.state:
            return self.state
        else:
            states = [system.computeState() for system in self.subsystems]
            if any(val is None for val in states):
                return None
            state = states[0]
            for s in states[1:]:
                state = state.tensor(s)
            self.state = state
            return self.state

    def reduceState(self):
        for system in self.subsystems:
            system.state = self.getSubsystemState(system)
        self.__state = None

    def subsystemIndex(self, target_system: System):
        subsystem_index = 0
        for system in self.subsystems:
            if system == target_system:
                return [subsystem_index + i for i in range(system.nsystems)]
            else:
                if type(system) != SingleSystem:
                    if system.subsystemIndex(target_system) is not None:
                        return [
                            subsystem_index + i
                            for i in system.subsystemIndex(target_system)
                        ]
            subsystem_index += system.nsystems
        return None

    def getSubsystem(self, name: str):
        for system in self.subsystems:
            if name == system.name:
                return system
            else:
                if system.getSubsystem(name) is not None:
                    return system.getSubsystem(name)
        return None

    def addSubsystem(self, system: System):
        if system in self.subsystems:
            raise ValueError("System being added is already in this composite system")
        self.subsystems.append(system)
        self.state = self.state.tensor(system.state)
        self.nsystems += system.nsystems

    def removeSubsystem(self, system: System):
        if self.subsystemIndex(system) is None:
            raise ValueError("System to be removed is not in this composite system")

        system_indices = self.subsystemIndex(system)
        config1 = self.configuration
        config2 = self.configuration
        self.subsystems.remove(system)

        subsystem_state = self.state
        for i in reversed(range(self.nsystems)):
            if i not in system_indices:
                subsystem_state = subsystem_state.partialTrace(i)
                config1.pop(i)

        system.state = subsystem_state
        for index in reversed(system_indices):

            self.state = self.state.partialTrace(index)
            config2.pop(index)

        self.nsystems -= system.nsystems

    def getDynamics(self):
        dynamics = [d for d in self.dynamics]

        for system in self.subsystems:
            dynamics += system.getDynamics()

        return dynamics

    def updateDynamics(self):
        self.dynamics_manager.dynamic_funcs = []

        with multiprocessing.Pool() as p:
            funcs = p.map(self.updateDynamic, tqdm.tqdm(self.getDynamics()))

        for dynamicFunc in funcs:
            self.dynamics_manager.addDynamics(dynamicFunc)

    def updateDynamic(self, dynamic):
        systems = dynamic.systems
        indices = []
        for system in systems:
            indices += self.subsystemIndex(system)
        config, dims = self.generateConfiguration(indices)
        dynamic.updateOperators(config, dims)
        return dynamic

    def generateConfiguration(self, indices):
        dims = self.configuration
        config = [-1 for i in dims]

        for i, index in enumerate(indices):
            config[index] = i

        return config, dims

    def evolve(self, time):
        t0 = self.t
        while self.t < t0 + time:
            self.state = self.dynamics_manager.evolve(self.state)
            self.t += self.dynamics_manager.timestep
        if not self.state.isLegitamate():
            print(f"Illegitimate state")

    def getSubsystemState(self, subsystem, time=-1):
        system_indices = self.subsystemIndex(subsystem)
        if time == -1 or time == self.t:
            subsystem_state = self.state
        else:
            subsystem_state = self.logger.log[time]
        config = self.configuration
        for i in reversed(range(self.nsystems)):
            if i not in system_indices:
                subsystem_state = subsystem_state.partialTrace(i)
                config.pop(i)

        return subsystem_state

    def measureSubsystem(self, subsystem, measurement):
        return self.getSubsystemState(subsystem).measure(measurement)


class StateLogger:

    def __init__(self):
        self.log = {}

    def record(self, time, state):
        if state:
            self.log[round(time, 6)] = state
