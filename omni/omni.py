import gym
import numpy as np
from gym import error, spaces
import itertools
import uuid
from omni.interfaces.registration import affordance_registry, task_registry
from omni.config import MAX_PARAMS, MAX_LENGTH, CHAR_EMBEDDING


class Omni():
    def __init__(self):
        self.instances = {}
        self.id_len = 10

    def instantiate(self):
        instance_id = str(uuid.uuid4().hex)[:self.id_len]
        instance = Instance(instance_id)
        self.instances[instance_id] = instance
        return instance

    @property
    def action_embedding(self):
        return affordance_registry.active_affordances


class Instance(gym.Env):
    def __init__(self, instance_id):
        self.instance_id = instance_id
        self.done = False

    def _close(self):
        print("Not Implemented yet!")

    def _reset(self):
        return self.observation_space.sample()

    def _render(self, mode='human', close=False):
        print("Not Implemented yet!")

    def _seed(self, seed=None):
        print("Cannot seed a live environment!")

    def _step(self, action):
        print(action)
        affordance = affordance_registry.lookup(15) #
        self.observation= affordance(*action[1])
        print("rewarding")
        self.reward = task_registry.aggregate()
        print("rewarded")
        print(self.reward)
        return self.observation, self.reward, self.done, self.info

    def spawn(self, index, k):
        return affordance_registry.spawn(index, k)

    def candidate_action_space(self, k):
        return spaces.Box(low=0, high=len(affordance_registry.list_all()), shape=k)

    @property
    def info(self):
        return {"Not Implemented yet!"}

    @property
    def action_embedding(self):
        return affordance_registry.active_affordances

    @property
    def index_action_space(self):
        return spaces.Box(low=-1, high=1, shape=1)

    @property
    def param_action_space(self):
        return spaces.Box(low=-1, high=1, shape=MAX_PARAMS)

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=len(CHAR_EMBEDDING) + 1, shape=(MAX_LENGTH))

    @property
    def reward_space(self):
        return spaces.Box(low=-100, high=100, shape=len(task_registry.list_all()))

omni = Omni()

def instantiate():
    instance = omni.instantiate()
    return instance


