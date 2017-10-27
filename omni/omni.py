import gym
import numpy as np
from gym import error, spaces
import itertools
import uuid
from omni.interfaces.registration import affordance_registry, task_registry
from omni.config import MAX_PARAMS, LINE_LENGTH, CHAR_EMBEDDING


class Omni():
    def __init__(self):
        self.instances = {}
        self.id_len = 10

    def instantiate(self):
        instance_id = str(uuid.uuid4().hex)[:self.id_len]
        instance = Instance(self, instance_id)
        self.instances[instance_id] = instance
        return instance

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=len(CHAR_EMBEDDING), shape=(LINE_LENGTH))

    @property
    def reward_space(self):
        return spaces.Box(low=-100, high=100, shape=len(task_registry.list_all()))

    @property
    def action_space(self):
        return spaces.Box(low=-1, high=1, shape=1+MAX_PARAMS)


class Instance(gym.Env):
    def __init__(self, omni, instance_id):
        super().__init__()
        self.omni = omni
        self.instance_id = instance_id
        self.step_count = 0
        self.done = False
        self.penalty = 0

    def _close(self):
        #closer_registry.close()
        print("Cannot close a live environment!")

    def _reset(self):
        return self.observation_space.sample()

    def _seed(self, seed=None):
        raise Warning("Cannot seed a live environment!")

    def _step(self, action):
        affordance = affordance_registry.lookup(int(action[:1]))
        self.observation= affordance(action[1:])
        self.reward = task_registry.aggregate()
        self.step_count += 1
        return self.observation, self.reward, self.done, self.info

    @property
    def info(self):
        return {
            'step': self.step_count
        }

    @property
    def discrete_count(self):
        return len(affordance_registry.list_all())

    @property
    def action_space(self):
        return self.omni.action_space

    @property
    def observation_space(self):
        return self.omni.observation_space

    @property
    def reward_space(self):
        return self.omni.reward_space

    @property
    def action_embedding(self):
        return affordance_registry.affordances.keys()

omni = Omni()

def instantiate():
    instance = omni.instantiate()
    return instance