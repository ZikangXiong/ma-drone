from abc import abstractmethod, ABC

import numpy as np
import pybullet as p

from ma_drone.utils import no_render


class WorldBase(ABC):
    def __init__(self, enable_gui: bool):
        if enable_gui:
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)

        self._init_param()

        with no_render():
            self._build_world()

        p.setRealTimeSimulation(0, physicsClientId=self.client_id)

    @abstractmethod
    def _init_param(self):
        raise NotImplementedError()

    @abstractmethod
    def _build_world(self):
        raise NotImplementedError()


class RobotBase(ABC):
    def __init__(self, world: WorldBase):
        self.world = world

        self.client_id = self.world.client_id

        with no_render():
            self.robot_id = self._load_robot()

        self._init_param()

    @abstractmethod
    def _load_robot(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def _init_param(self):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def ctrl(self, cmd: np.ndarray):
        raise NotImplementedError()

    @abstractmethod
    def get_obs(self) -> np.ndarray:
        pass
