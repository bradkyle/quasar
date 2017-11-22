import tensorflow as tf
import omni
import numpy as np
from agents.common.mpi_running_mean_std import RunningMeanStd


def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean


class Agent():
    def __init__(self,
                 scope,
                 observation_shape,
                 rewards_shape,
                 candidates_shape,
                 memory,
                 gamma=0.99,
                 tau=0.001,
                 normalize_observations=True,
                 normalize_returns=False,
                 action_noise=None,
                 param_noise=None,
                 action_range=(-1., 1.),
                 return_range=(-np.inf, np.inf),
                 observation_range=(-2., 100.),
                 reward_scale=1,
                 clip_norm=None,
                 enable_popart=False,
                 batch_size=128
                 ):

        with tf.variable_scope(scope):
            self.obs0 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs0')
            self.obs1 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs1')
            self.step = tf.placeholder(tf.float32, shape=(None, 1), name='step')
            self.rewards0 = tf.placeholder(tf.float32, shape=(None,) + rewards_shape, name='reward0')
            self.rewards1 = tf.placeholder(tf.float32, shape=(None,) + rewards_shape, name='reward1')
            self.candidate_actions = tf.placeholder(tf.float32, shape=(None, candidates_shape),
                                                    name='candidate_actions')

            self.gamma = gamma
            self.tau = tau
            self.memory = memory
            self.normalize_observations = normalize_observations
            self.normalize_returns = normalize_returns
            self.action_noise = action_noise
            self.param_noise = param_noise
            self.action_range = action_range
            self.return_range = return_range
            self.observation_range = observation_range
            self.rewards_scale = reward_scale
            self.clip_norm = clip_norm
            self.enable_popart = enable_popart
            self.batch_size = batch_size
            self.stats_sample = None

            if self.normalize_observations:
                with tf.variable_scope('obs_rms'):
                    self.obs_rms = RunningMeanStd(shape=observation_shape)
            else:
                self.obs_rms = None

            self.normalized_obs0 = tf.clip_by_value(normalize(self.obs0, self.obs_rms),
                                               self.observation_range[0], self.observation_range[1])
            self.normalized_obs1 = tf.clip_by_value(normalize(self.obs1, self.obs_rms),
                                               self.observation_range[0], self.observation_range[1])

            # Return normalization.
            if self.normalize_returns:
                with tf.variable_scope('ret_rms'):
                    self.ret_rms = RunningMeanStd()
            else:
                self.ret_rms = None

    def _initialize(self, sess): raise NotImplementedError
    def _setup(self): raise NotImplementedError

    def initialize(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        return self._initialize(sess)

    def store_transition(self, obs, new_obs, index_action, candidate_actions, param_action, rewards, new_rewards, done, step):
        self.memory.append(obs, new_obs, index_action, candidate_actions, param_action, rewards, new_rewards, done, step)
        if self.normalize_observations:
            self.obs_rms.update(np.array([obs]))

class DiscreteAgent():
    def __init__(self):
        raise NotImplementedError

class DisConAgent():
    def __init__(self):
        raise NotImplementedError