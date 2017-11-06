import tensorflow as tf
from copy import copy
from collections import deque
import numpy as np
from common.mpi_adam import MpiAdam
from functools import reduce
import common.tf_util as U
import tensorflow.contrib as tc
from common.mpi_running_mean_std import RunningMeanStd
from tscl_agent.util import reduce_std, mpi_mean
import common.logger as logger
import numpy.random as rng
from operator import sub

def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean


def get_target_updates(vars, target_vars, tau):
    # todo logger.info('setting up target updates ...')
    soft_updates = []
    init_updates = []
    assert len(vars) == len(target_vars)
    for var, target_var in zip(vars, target_vars):
        # todo logger.info('  {} <- {}'.format(target_var.name, var.name))
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(vars)
    assert len(soft_updates) == len(vars)
    return tf.group(*init_updates), tf.group(*soft_updates)


def get_perturbed_actor_updates(actor, perturbed_actor, param_noise_stddev):
    assert len(actor.vars) == len(perturbed_actor.vars)
    assert len(actor.perturbable_vars) == len(perturbed_actor.perturbable_vars)

    updates = []
    for var, perturbed_var in zip(actor.vars, perturbed_actor.vars):
        if var in actor.perturbable_vars:
            logger.info('  {} <- {} + noise'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var + tf.random_normal(tf.shape(var), mean=0., stddev=param_noise_stddev)))
        else:
            logger.info('  {} <- {}'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var))
    assert len(updates) == len(actor.vars)
    return tf.group(*updates)


class Agent():
    def __init__(self,
                 env,
                 shared,
                 index_actor,
                 index_critic,
                 param_actor,
                 param_critic,
                 memory,
                 rewards_shape,
                 candidates_shape,
                 param_action_shape,
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
                 index_actor_lr=1e-4,
                 index_critic_lr=1e-3,
                 param_actor_lr=1e-4,
                 param_critic_lr=1e-3,
                 clip_norm=None,
                 enable_popart=False,
                 batch_size=128,
                 param_critic_l2_reg=0.,
                 index_critic_l2_reg=0.
                 ):

        self.env = env

        # Placeholders
        self.obs0 = tf.placeholder(tf.float32, shape=(None,) + self.env.observation_space.shape, name='obs0')
        self.obs1 = tf.placeholder(tf.float32, shape=(None,) + self.env.observation_space.shape, name='obs1')
        self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
        self.step = tf.placeholder(tf.float32, shape=(None, 1), name='step')
        self.reward = tf.placeholder(tf.float32, shape=(None, 1), name='reward')
        self.reward0 = tf.placeholder(tf.float32, shape=(None,) + rewards_shape, name='reward0')
        self.reward1 = tf.placeholder(tf.float32, shape=(None,) + rewards_shape, name='reward1')
        self.candidate_actions = tf.placeholder(tf.float32, shape=(None, candidates_shape), name='candidate_actions')
        self.param_actions = tf.placeholder(tf.float32, shape=(None,) + param_action_shape, name='actions')
        self.index_critic_target = tf.placeholder(tf.float32, shape=(None, 1), name='index_critic_target')
        self.param_critic_target = tf.placeholder(tf.float32, shape=(None, 1), name='param_critic_target')
        self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')

        # Parameters.
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
        self.index_critic = index_critic
        self.index_actor = index_actor
        self.index_actor_lr = index_actor_lr
        self.index_critic_lr = index_critic_lr
        self.param_critic = param_critic
        self.param_actor = param_actor
        self.param_actor_lr = param_actor_lr
        self.param_critic_lr = param_critic_lr
        self.clip_norm = clip_norm
        self.enable_popart = enable_popart
        self.batch_size = batch_size
        self.stats_sample = None
        self.param_critic_l2_reg = param_critic_l2_reg
        self.index_critic_l2_reg = index_critic_l2_reg

        # Observation Normalization
        # ------------------------------------------------------------------------------------------------------------->

        if self.normalize_observations:
            with tf.variable_scope('obs_rms'):
                self.obs_rms = RunningMeanStd(shape=self.env.observation_space.shape)
        else:
            self.obs_rms = None

        normalized_obs0 = tf.clip_by_value(normalize(self.obs0, self.obs_rms),
                                           self.observation_range[0], self.observation_range[1])
        normalized_obs1 = tf.clip_by_value(normalize(self.obs1, self.obs_rms),
                                           self.observation_range[0], self.observation_range[1])

        # Return normalization.
        if self.normalize_returns:
            with tf.variable_scope('ret_rms'):
                self.ret_rms = RunningMeanStd()
        else:
            self.ret_rms = None


        # Target Models
        # ------------------------------------------------------------------------------------------------------------->

        # Create target index actor.
        target_index_actor = copy(index_actor)
        target_index_actor.name = 'target_index_actor'
        self.target_index_actor = target_index_actor

        # Create target index critic.
        target_index_critic = copy(index_critic)
        target_index_critic.name = 'target_index_critic'
        self.target_index_critic = target_index_critic

        # Create target param critic.
        target_param_critic = copy(param_critic)
        target_param_critic.name = 'target_param_critic'
        self.target_param_critic = target_param_critic

        # Create target param actor.
        target_param_actor = copy(param_actor)
        target_param_actor.name = 'target_param_actor'
        self.target_param_actor = target_param_actor

        target_shared = copy(shared)
        target_shared.name = 'target_shared'
        self.target_shared = target_shared

        # Models
        # ------------------------------------------------------------------------------------------------------------->

        # Shared layers
        self.shared_tf = shared(normalized_obs0)

        # Index action path
        self.index_actor_tf = index_actor(self.shared_tf)

        self.normalized_index_critic_tf = index_critic(normalized_obs0, self.candidate_actions)
        self.index_critic_tf = denormalize(tf.clip_by_value(self.normalized_index_critic_tf, self.return_range[0], self.return_range[1]), self.ret_rms)

        # Param action path
        self.param_actor_tf = param_actor(self.shared_tf)

        self.normalized_param_critic_tf = param_critic(normalized_obs0, self.param_actions)
        self.param_critic_tf = denormalize(tf.clip_by_value(self.normalized_param_critic_tf, self.return_range[0], self.return_range[1]), self.ret_rms)

        self.normalized_critic_with_actor_tf = param_critic(normalized_obs0, self.param_actor_tf, reuse=True)
        self.param_critic_with_actor_tf = denormalize(tf.clip_by_value(self.normalized_critic_with_actor_tf, self.return_range[0], self.return_range[1]), self.ret_rms)


        # Target Models
        # ------------------------------------------------------------------------------------------------------------->

        self.reward_delta = tf.subtract(self.reward1, self.reward0, name="reward_delta")
        self.abs_reward = tf.abs(self.reward_delta)
        self.greed_index = tf.argmax(self.abs_reward , axis=1)
        self.rewardy = tf.gather(self.reward1, self.greed_index, axis=1)

        self.target_shared_tf = target_shared(normalized_obs1)
        self.target_index_actor_tf = target_index_actor(self.target_shared_tf)

        Q_param_obs1 = denormalize(target_param_critic(normalized_obs1, target_param_actor(self.target_shared_tf)), self.ret_rms)
        self.target_param_Q = self.reward + (1. - self.terminals1) * gamma * Q_param_obs1 # self.qi(self.reward1, self.reward0)

        Q_index_obs1 = denormalize(target_index_critic(normalized_obs1, self.candidate_actions), self.ret_rms)
        self.target_index_Q = self.reward + (1. - self.terminals1) * gamma * Q_index_obs1

        # Set up parts
        # ------------------------------------------------------------------------------------------------------------->
        if self.param_noise is not None:
            self.setup_param_noise(normalized_obs0)
        self.setup_optimizers()
        if self.normalize_returns and self.enable_popart:
            self.setup_popart()

        self.setup_stats()
        self.setup_target_network_updates()


    def initialize(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.index_actor_optimizer.sync()
        self.index_critic_optimizer.sync()
        self.sess.run(self.target_index_init_updates)
        self.param_actor_optimizer.sync()
        self.param_critic_optimizer.sync()
        self.sess.run(self.target_param_init_updates)

    def setup_target_network_updates(self):
        index_actor_init_updates, index_actor_soft_updates = get_target_updates(self.index_actor.vars, self.target_index_actor.vars, self.tau)
        index_critic_init_updates, index_critic_soft_updates = get_target_updates(self.index_critic.vars, self.target_index_critic.vars, self.tau)

        self.target_index_init_updates = [index_actor_init_updates, index_critic_init_updates]
        self.target_index_soft_updates = [index_actor_soft_updates, index_critic_soft_updates]

        param_actor_init_updates, param_actor_soft_updates = get_target_updates(self.param_actor.vars, self.target_param_actor.vars, self.tau)
        param_critic_init_updates, param_critic_soft_updates = get_target_updates(self.param_critic.vars, self.target_param_critic.vars, self.tau)

        self.target_param_init_updates = [param_actor_init_updates, param_critic_init_updates]
        self.target_param_soft_updates = [param_actor_soft_updates, param_critic_soft_updates]

    def setup_param_noise(self, normalized_obs0):
        assert self.param_noise is not None

        # Configure perturbed actor.
        param_noise_index_actor = copy(self.index_actor)
        param_noise_param_actor = copy(self.param_actor)
        param_noise_index_actor.name = 'param_noise_index_actor'
        param_noise_param_actor.name = 'param_noise_param_actor'
        self.perturbed_index_actor_tf = param_noise_index_actor(normalized_obs0)
        self.perturbed_param_actor_tf = param_noise_param_actor(normalized_obs0)

        logger.info('setting up param noise')
        self.perturb_policy_ops = get_perturbed_actor_updates(self.param_actor, param_noise_param_actor, self.param_noise_stddev)
        self.perturb_policy_ops = get_perturbed_actor_updates(self.index_actor, param_noise_index_actor, self.param_noise_stddev)


        # Configure separate copy for stddev adoption.
        adaptive_param_noise_index_actor = copy(self.index_actor)
        adaptive_param_noise_index_actor.name = 'adaptive_param_noise_index_actor'
        adaptive_index_actor_tf = adaptive_param_noise_index_actor(normalized_obs0)
        self.perturb_adaptive_policy_ops = get_perturbed_actor_updates(self.index_actor, adaptive_param_noise_index_actor, self.param_noise_stddev)
        self.adaptive_policy_distance = tf.sqrt(tf.reduce_mean(tf.square(self.index_actor_tf - adaptive_index_actor_tf)))

        adaptive_param_noise_param_actor = copy(self.index_actor)
        adaptive_param_noise_param_actor.name = 'adaptive_param_noise_param_actor'
        adaptive_param_actor_tf = adaptive_param_noise_param_actor(normalized_obs0)
        self.perturb_adaptive_policy_ops = get_perturbed_actor_updates(self.param_actor, adaptive_param_noise_param_actor, self.param_noise_stddev)
        self.adaptive_policy_distance = tf.sqrt(tf.reduce_mean(tf.square(self.param_actor_tf - adaptive_param_actor_tf)))

    def setup_optimizers(self):

        #logger.info('setting up param actor optimizer')
        self.param_actor_loss = -tf.reduce_mean(self.param_critic_with_actor_tf)

        param_actor_shapes = [var.get_shape().as_list() for var in self.param_actor.trainable_vars]
        param_actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in param_actor_shapes])

        self.param_actor_grads = U.flatgrad(self.param_actor_loss, self.param_actor.trainable_vars, clip_norm=self.clip_norm)
        self.param_actor_optimizer = MpiAdam(var_list=self.param_actor.trainable_vars, beta1=0.9, beta2=0.999, epsilon=1e-08)


        #logger.info('setting up index actor optimizer')
        self.index_actor_loss = -tf.reduce_mean(self.index_critic_tf)

        index_actor_shapes = [var.get_shape().as_list() for var in self.index_actor.trainable_vars]
        index_actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in index_actor_shapes])

        #logger.info('  index actor shapes: {}'.format(index_actor_shapes))
        #logger.info('  index actor params: {}'.format(index_actor_nb_params))

        self.index_actor_grads = U.flatgrad(self.index_actor_loss, self.index_actor.trainable_vars, clip_norm=self.clip_norm)
        self.index_actor_optimizer = MpiAdam(var_list=self.index_actor.trainable_vars, beta1=0.9, beta2=0.999,  epsilon=1e-08)

        #logger.info('setting up param critic optimizer')
        normalized_param_critic_target_tf = tf.clip_by_value(normalize(self.param_critic_target, self.ret_rms),
                                                       self.return_range[0], self.return_range[1])
        self.param_critic_loss = tf.reduce_mean(tf.square(self.normalized_param_critic_tf - normalized_param_critic_target_tf))
        if self.param_critic_l2_reg > 0.:
            param_critic_reg_vars = [var for var in self.param_critic.trainable_vars if
                               'kernel' in var.name and 'output' not in var.name]
            #for var in param_critic_reg_vars:
                #logger.info('  regularizing: {}'.format(var.name))
            #logger.info('  applying l2 regularization with {}'.format(self.param_critic_l2_reg))
            param_critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.param_critic_l2_reg),
                weights_list=param_critic_reg_vars
            )
            self.param_critic_loss += param_critic_reg
        param_critic_shapes = [var.get_shape().as_list() for var in self.param_critic.trainable_vars]
        param_critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in param_critic_shapes])
        #l#ogger.info('  param critic shapes: {}'.format(param_critic_shapes))
        #logger.info('  param critic params: {}'.format(param_critic_nb_params))
        self.param_critic_grads = U.flatgrad(self.param_critic_loss, self.param_critic.trainable_vars, clip_norm=self.clip_norm)
        self.param_critic_optimizer = MpiAdam(var_list=self.param_critic.trainable_vars,
                                        beta1=0.9, beta2=0.999, epsilon=1e-08)

        #logger.info('setting up index critic optimizer')
        normalized_index_critic_target_tf = tf.clip_by_value(normalize(self.index_critic_target, self.ret_rms),
                                                       self.return_range[0], self.return_range[1])
        self.index_critic_loss = tf.reduce_mean(tf.square(self.normalized_index_critic_tf - normalized_index_critic_target_tf))
        if self.index_critic_l2_reg > 0.:
            index_critic_reg_vars = [var for var in self.index_critic.trainable_vars if
                               'kernel' in var.name and 'output' not in var.name]
            #for var in index_critic_reg_vars:
                #logger.info('  regularizing: {}'.format(var.name))
            #logger.info('  applying l2 regularization with {}'.format(self.index_critic_l2_reg))
            index_critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.index_critic_l2_reg),
                weights_list=index_critic_reg_vars
            )
            self.index_critic_loss += index_critic_reg
        index_critic_shapes = [var.get_shape().as_list() for var in self.index_critic.trainable_vars]
        index_critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in index_critic_shapes])
        #logger.info('  index critic shapes: {}'.format(index_critic_shapes))
        #logger.info('  index critic params: {}'.format(index_critic_nb_params))
        self.index_critic_grads = U.flatgrad(self.index_critic_loss, self.index_critic.trainable_vars, clip_norm=self.clip_norm)
        self.index_critic_optimizer = MpiAdam(var_list=self.index_critic.trainable_vars,
                                        beta1=0.9, beta2=0.999, epsilon=1e-08)


    def setup_popart(self):
        # See https://arxiv.org/pdf/1602.07714.pdf for details.
        self.old_param_std = tf.placeholder(tf.float32, shape=[1], name='old_param_std')
        new_param_std = self.ret_rms.std
        self.old_param_mean = tf.placeholder(tf.float32, shape=[1], name='old_param_mean')
        new_param_mean = self.ret_rms.mean

        self.old_index_std = tf.placeholder(tf.float32, shape=[1], name='old_index_std')
        new_index_std = self.ret_rms.std
        self.old_index_mean = tf.placeholder(tf.float32, shape=[1], name='old_index_mean')
        new_index_mean = self.ret_rms.mean

        self.renormalize_index_Q_outputs_op = []
        for vs in [self.index_critic.output_vars, self.target_index_critic.output_vars]:
            assert len(vs) == 2
            M, b = vs
            assert 'kernel' in M.name
            assert 'bias' in b.name
            assert M.get_shape()[-1] == 1
            assert b.get_shape()[-1] == 1
            self.renormalize_index_Q_outputs_op += [M.assign(M * self.old_index_std / new_index_std)]
            self.renormalize_index_Q_outputs_op += [b.assign((b * self.old_index_std + self.old_index_mean - new_index_mean) / new_index_std)]

        self.renormalize_param_Q_outputs_op = []
        for vs in [self.param_critic.output_vars, self.target_param_critic.output_vars]:
            assert len(vs) == 2
            M, b = vs
            assert 'kernel' in M.name
            assert 'bias' in b.name
            assert M.get_shape()[-1] == 1
            assert b.get_shape()[-1] == 1
            self.renormalize_param_Q_outputs_op += [M.assign(M * self.old_param_std / new_param_std)]
            self.renormalize_param_Q_outputs_op += [b.assign((b * self.old_param_std + self.old_param_mean - new_param_mean) / new_param_std)]

    # policy
    def pi(self, obs, apply_noise=True, compute_Q=True):

        if self.param_noise is not None and apply_noise:
            index_actor_tf = self.perturbed_index_actor_tf
            param_actor_tf = self.perturbed_param_actor_tf
        else:
            index_actor_tf = self.index_actor_tf
            param_actor_tf = self.param_actor_tf

        feed_dict = {self.obs0: [obs]}

        if compute_Q:
            index_action, param_action, q = self.sess.run([index_actor_tf, param_actor_tf, self.param_critic_with_actor_tf], feed_dict=feed_dict)
        else:
            index_action, param_action = self.sess.run(index_actor_tf, param_actor_tf, feed_dict=feed_dict)
            q = None


        index_action, param_action = index_action.flatten(), param_action.flatten()
        if self.action_noise is not None and apply_noise:
            noise = self.action_noise()
            assert noise.shape == index_action.shape and param_action.shape
            param_action += noise
            index_action += noise

        param_action = np.clip(param_action, self.action_range[0], self.action_range[1])
        index_action = np.clip(index_action, self.action_range[0], self.action_range[1])

        return index_action, param_action, q

    def val(self, obs, candidates):
        candidate_dist = self.sess.run(
            fetches=[self.index_critic_tf],
            feed_dict={self.obs0: [obs], self.candidate_actions: [candidates]}
        )
        return candidates[np.argmax(candidate_dist)]

    def sum_changes(self, rewards1, rewards0):
        delta = 0
        for reward0, reward1 in zip(rewards0, rewards1):
            delta += abs(reward1 - reward0)
        return delta

    def qi(self, rewards1, rewards0):
        reward_delta = []
        for reward0, reward1 in zip(rewards0, rewards1):
            reward_delta.append(abs(reward1-reward0))
        task_index = np.argmax(reward_delta)
        return rewards1[task_index]

    def store_transition(self, obs, new_obs, index_action, candidate_actions, param_action, rewards, new_rewards, done, step):
        self.memory.append(obs, new_obs, index_action, candidate_actions, param_action, rewards, new_rewards, done, step)
        #if self.normalize_observations: #todo
        #    self.obs_rms.update(np.array([obs0]))

    def train(self):
        batch = self.memory.sample(batch_size=self.batch_size)


        if self.normalize_returns and self.enable_popart:
            old_mean, old_std, target_param_Q, target_index_Q = self.sess.run(
                fetches=[
                    self.ret_rms.mean,
                    self.ret_rms.std,
                    self.target_param_Q,
                    self.target_index_Q
                ],
                feed_dict={
                    self.obs1: batch['obs1'],
                    self.reward0: batch['rewards0'],
                    self.reward1: batch['rewards1'],
                    self.terminals1: batch['terminals1'].astype('float32'),
                }
            )

            self.ret_rms.update(target_index_Q.flatten())
            self.ret_rms.update(target_param_Q.flatten())

            self.sess.run(
                fetches=[
                    self.renormalize_index_Q_outputs_op
                ],
                feed_dict={
                    self.old_index_std: np.array([old_std]),
                    self.old_index_mean: np.array([old_mean]),
                }
            )

            self.sess.run(
                fetches = [
                    self.renormalize_param_Q_outputs_op
                ],
                feed_dict = {
                    self.old_param_std: np.array([old_std]),
                    self.old_param_mean: np.array([old_mean])
                }
            )

        else:
            target_index_Q, target_param_Q = self.sess.run(
                fetches = [
                    self.target_index_Q,
                    self.target_param_Q
                ],
                feed_dict={
                    self.obs1: batch['obs1'],
                    self.reward0: batch['rewards0'],
                    self.reward1: batch['rewards1'],
                    self.terminals1: batch['terminals1'].astype('float32'),
                }
            )

        # Get all gradients and perform a synced update.
        index_actor_grads,\
        index_actor_loss, \
        param_actor_grads, \
        param_actor_loss, \
        index_critic_grads, \
        index_critic_loss, \
        param_critic_grads, \
        param_critic_loss \
            = self.sess.run(
            fetches=[
                self.index_actor_grads,
                self.index_actor_loss,
                self.param_actor_grads,
                self.param_actor_loss,
                self.index_critic_grads,
                self.index_critic_loss,
                self.param_critic_grads,
                self.param_critic_loss
            ],
            feed_dict={
                self.obs0: batch['obs0'],
                self.param_actions: batch['param_actions'],
                self.candidate_actions: batch['candidate_actions'],
                self.index_critic_target: target_index_Q,
                self.param_critic_target: target_param_Q,
            }
        )

        self.index_actor_optimizer.update(index_actor_grads, stepsize=self.index_actor_lr)
        self.index_critic_optimizer.update(index_critic_grads, stepsize=self.index_critic_lr)

        self.param_actor_optimizer.update(index_actor_grads, stepsize=self.param_actor_lr)
        self.param_critic_optimizer.update(index_critic_grads, stepsize=self.param_critic_lr)

        return index_critic_loss, index_actor_loss, param_critic_loss, param_actor_loss

    def update_target_net(self):
        self.sess.run(self.target_index_soft_updates)
        self.sess.run(self.target_param_soft_updates)

    def setup_stats(self):
        ops = []
        names = []

        if self.normalize_returns:
            ops += [self.ret_rms.mean, self.ret_rms.std]
            names += ['ret_rms_mean', 'ret_rms_std']

        if self.normalize_observations:
            ops += [tf.reduce_mean(self.obs_rms.mean), tf.reduce_mean(self.obs_rms.std)]
            names += ['obs_rms_mean', 'obs_rms_std']

        ops += [tf.reduce_mean(self.index_critic_tf)]
        names += ['reference_index_Q_mean']
        ops += [reduce_std(self.index_critic_tf)]
        names += ['reference_index_Q_std']

        ops += [tf.reduce_mean(self.index_actor_tf)]
        names += ['reference_index_action_mean']
        ops += [reduce_std(self.index_actor_tf)]
        names += ['reference_index_action_std']

        ops += [tf.reduce_mean(self.param_critic_tf)]
        names += ['reference_param_Q_mean']
        ops += [reduce_std(self.param_critic_tf)]
        names += ['reference_param_Q_std']

        ops += [tf.reduce_mean(self.param_critic_with_actor_tf)]
        names += ['reference_param_actor_Q_mean']
        ops += [reduce_std(self.param_critic_with_actor_tf)]
        names += ['reference_param_actor_Q_std']

        ops += [tf.reduce_mean(self.param_actor_tf)]
        names += ['reference_param_action_mean']
        ops += [reduce_std(self.param_actor_tf)]
        names += ['reference_param_action_std']

        if self.param_noise:
            ops += [tf.reduce_mean(self.perturbed_index_actor_tf)]
            names += ['reference_perturbed_index_action_mean']
            ops += [reduce_std(self.perturbed_index_actor_tf)]
            names += ['reference_perturbed_index_action_std']

            ops += [tf.reduce_mean(self.perturbed_param_actor_tf)]
            names += ['reference_perturbed_param_action_mean']
            ops += [reduce_std(self.perturbed_param_actor_tf)]
            names += ['reference_perturbed_param_action_std']

        self.stats_ops = ops
        self.stats_names = names

    def get_stats(self):
        if self.stats_sample is None:
            # Get a sample and keep that fixed for all further computations.
            # This allows us to estimate the change in value for the same set of inputs.
            self.stats_sample = self.memory.sample(batch_size=self.batch_size)
        values = self.sess.run(self.stats_ops, feed_dict={
            self.obs0: self.stats_sample['obs0'],
            self.candidate_actions: self.stats_sample['candidate_actions'],
            self.param_actions: self.stats_sample['param_actions'],
        })

        names = self.stats_names[:]
        assert len(names) == len(values)
        stats = dict(zip(names, values))

        if self.param_noise is not None:
            stats = {**stats, **self.param_noise.get_stats()}

        return stats

    def adapt_param_noise(self):
        if self.param_noise is None:
            return 0.

        # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
        batch = self.memory.sample(batch_size=self.batch_size)
        self.sess.run(self.perturb_adaptive_policy_ops, feed_dict={
            self.param_noise_stddev: self.param_noise.current_stddev,
        })
        distance = self.sess.run(self.adaptive_policy_distance, feed_dict={
            self.obs0: batch['obs0'],
            self.param_noise_stddev: self.param_noise.current_stddev,
        })

        mean_distance = mpi_mean(distance)
        self.param_noise.adapt(mean_distance)
        return mean_distance

    def reset(self):
        # Reset internal state after an episode is complete.
        if self.action_noise is not None:
            self.action_noise.reset()
        if self.param_noise is not None:
            self.sess.run(self.perturb_policy_ops, feed_dict={
                self.param_noise_stddev: self.param_noise.current_stddev,
            })