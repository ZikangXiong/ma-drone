import numpy as np
import pybullet as p
from gym.spaces import Box

from ma_drone.envs.drone import DroneWorld


class DataCollector:
    def __init__(self):
        self.buffer = []
        goal_lb = np.array([-5, -5, 2], dtype=np.float64)
        goal_ub = np.array([5, 5, 7], dtype=np.float64)
        self.goal_space = Box(low=goal_lb, high=goal_ub, shape=(3,), dtype=np.float64)
        self.reach_radius = 0.3
        self.action_comb = 10

    def collect(self, n_timesteps: int):
        # TODO: parallelize
        buffer = self.worker(n_timesteps)
        self.buffer.extend(buffer)

    def worker(self, n_timesteps: int) -> np.ndarray:
        world = DroneWorld(enable_gui=False)
        world.add_drone(np.array([0, 0, 1]))
        drone = world.drones["drone-0"]
        controller = world.controllers["drone-0"]

        p.stepSimulation(world.client_id)
        goal = self.goal_space.sample()
        prev_obs = drone.get_relative_obs(goal)

        buffer = []
        for _ in range(n_timesteps):
            reach = np.linalg.norm(prev_obs[:3]) < self.reach_radius
            if reach:
                goal = self.goal_space.sample()
                controller.reset()
                prev_obs = drone.get_relative_obs(goal)

            for _ in range(self.action_comb):
                cmd = controller.control(goal)
                drone.ctrl(cmd)
                p.stepSimulation(world.client_id)

            obs = drone.get_relative_obs(goal)
            buffer.append([prev_obs, obs])
            prev_obs = obs

        return np.array(buffer)

    def dump(self, path: str):
        np.save(path, self.buffer)
