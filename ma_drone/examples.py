import time

import numpy as np
import pybullet as p

from ma_drone.ctrl.ctrl import DronePID
from ma_drone.envs.drone import Drone


def fly_one_drone():
    drone = Drone()

    for _ in range(1000):
        drone.ctrl(np.array([drone.hover_rpm + 10] * 4))
        p.stepSimulation(drone.client_id)
        time.sleep(0.01)


def ctrl_one_drone():
    drone = Drone()
    controller = DronePID(drone)
    goal = np.array([0, 0, 3])

    for _ in range(1000):
        drone.ctrl(controller.control(goal))
        p.stepSimulation(drone.client_id)
        time.sleep(0.01)


if __name__ == '__main__':
    # fly_one_drone()
    ctrl_one_drone()
