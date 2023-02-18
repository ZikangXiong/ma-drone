import gym


class MultiDroneEnv(gym.Env):
    def __init__(self, n_drones=1):
        self.n_drones = n_drones
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(n_drones, 3))

    def reset(self, **kwargs):
        pass

    def render(self, mode='human'):
        pass

    def step(self, action):
        pass

    def close(self):
        pass
