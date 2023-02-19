import time

import numpy as np
import pybullet as p

from ma_drone.config import DATA_ROOT
from ma_drone.envs.drone import DroneWorld
from ma_drone.nlf.data import DataCollector
from ma_drone.nlf.nn import NLF


def fly_one_drone():
    world = DroneWorld(enable_gui=True)
    world.add_drone(np.array([0, 0, 1]))
    drone = world.drones["drone-0"]

    for _ in range(1000):
        drone.ctrl(np.array([drone.hover_rpm + 10] * 4))
        p.stepSimulation(drone.client_id)
        time.sleep(0.01)


def ctrl_one_drone():
    world = DroneWorld(enable_gui=True)
    world.add_drone(np.array([0, 0, 1]))
    drone = world.drones["drone-0"]
    controller = world.controllers["drone-0"]
    goal = np.array([0, 1, 3])

    for _ in range(1000):
        drone.ctrl(controller.control(goal))
        p.stepSimulation(drone.client_id)
        time.sleep(0.01)


def ctrl_multiple_drones():
    world = DroneWorld(enable_gui=True)
    world.add_drone(np.array([0, -1, 1]))
    world.add_drone(np.array([0, 1, 1]))
    world.add_drone(np.array([-1, 0, 1]))
    world.add_drone(np.array([1, 0, 1]))

    goals = {
        "drone-0": np.array([0, 0, 2]),
        "drone-1": np.array([0, 0, 2.5]),
        "drone-2": np.array([0, 0, 3]),
        "drone-3": np.array([0, 0, 3.5])
    }
    drones = world.drones
    controller = world.controllers

    for _ in range(1000):
        for k in drones.keys():
            action = controller[k].control(goals[k])
            drones[k].ctrl(action)
        time.sleep(0.01)

        p.stepSimulation(world.client_id)


def collect_data():
    data_collector = DataCollector()
    data_collector.collect(100_000)
    data_collector.dump(f"{DATA_ROOT}/transitions.npy")


def train_nlf():
    data = np.load(f"{DATA_ROOT}/transitions.npy")
    nlf = NLF()
    nlf.train_nlf(data, 100, 128)
    nlf.save(f"{DATA_ROOT}/nlf.pth")


if __name__ == '__main__':
    # fly_one_drone()
    # ctrl_one_drone()
    # ctrl_multiple_drones()
    # collect_data()
    train_nlf()
