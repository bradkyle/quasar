import numpy as np



class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()

        self.data[(self.start + self.length - 1) % self.maxlen] = v


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    def __init__(self, limit, param_action_shape, index_action_shape, candidate_action_shape, observation_shape, reward_shape):
        self.limit = limit

        self.observations0 = RingBuffer(limit, shape=observation_shape)
        self.index_actions = RingBuffer(limit, shape=index_action_shape)
        self.candidate_actions = RingBuffer(limit, shape=candidate_action_shape)
        self.param_actions = RingBuffer(limit, shape=param_action_shape)
        self.rewards0 = RingBuffer(limit, shape=reward_shape)
        self.rewards1 = RingBuffer(limit, shape=reward_shape)
        self.steps = RingBuffer(limit, shape=(1,))
        self.terminals1 = RingBuffer(limit, shape=(1,))
        self.observations1 = RingBuffer(limit, shape=observation_shape)

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.randint(low=self.nb_entries - 2, size=batch_size, high=self.nb_entries)

        obs0_batch = self.observations0.get_batch(batch_idxs)
        obs1_batch = self.observations1.get_batch(batch_idxs)
        index_action_batch = self.index_actions.get_batch(batch_idxs)
        candidate_action_batch = self.candidate_actions.get_batch(batch_idxs)
        param_action_batch = self.param_actions.get_batch(batch_idxs)
        reward_batch = self.rewards0.get_batch(batch_idxs)
        new_reward_batch = self.rewards1.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)
        step_batch = self.steps.get_batch(batch_idxs)

        result = {
            'obs': array_min2d(obs0_batch),
            'new_obs': array_min2d(obs1_batch),
            'rewards': array_min2d(reward_batch),
            'new_rewards': array_min2d(new_reward_batch),
            'param_actions': array_min2d(param_action_batch),
            'candidate_actions': array_min2d(candidate_action_batch),
            'index_actions': array_min2d(index_action_batch),
            'terminals1': array_min2d(terminal1_batch),
            'step':array_min2d(step_batch),
        }
        return result

    def append(self, obs, new_obs, index_action, candidate_actions, param_action, rewards, new_rewards, done, step, training=True):
        if not training:
            return
        
        self.observations0.append(obs)
        self.rewards0.append(rewards)
        self.steps.append(step)
        self.index_actions.append(index_action)
        self.candidate_actions.append(candidate_actions)
        self.param_actions.append(param_action)
        self.observations1.append(new_obs)
        self.rewards1.append(new_rewards)
        self.terminals1.append(done)

    @property
    def nb_entries(self):
        return len(self.observations0)

    def get(self):
        batch_idxs = np.array([0,1])
        obs0_batch = self.observations0.get_batch(batch_idxs)
        obs1_batch = self.observations1.get_batch(batch_idxs)
        index_action_batch = self.index_actions.get_batch(batch_idxs)
        candidate_action_batch = self.candidate_actions.get_batch(batch_idxs)
        param_action_batch = self.param_actions.get_batch(batch_idxs)
        reward_batch = self.rewards0.get_batch(batch_idxs)
        new_reward_batch = self.rewards1.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)
        step_batch = self.steps.get_batch(batch_idxs)

        result = {
            'obs': array_min2d(obs0_batch),
            'new_obs': array_min2d(obs1_batch),
            'rewards': array_min2d(reward_batch),
            'new_rewards': array_min2d(new_reward_batch),
            'param_actions': array_min2d(param_action_batch),
            'candidate_actions': array_min2d(candidate_action_batch),
            'index_actions': array_min2d(index_action_batch),
            'terminals1': array_min2d(terminal1_batch),
            'step': array_min2d(step_batch),
        }
        return result