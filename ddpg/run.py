import argparse
import time
import os
import logging
import logger
from bench import Monitor
from common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
import omni.omni as omni
import ddpg.training as training
from ddpg.models import Actor, Critic, ProtoCritic
from ddpg.memory import Memory
from ddpg.noise import *
import gym
import tensorflow as tf
from mpi4py import MPI

def run(noise_type, layer_norm, evaluation, **kwargs):
    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0: logger.set_level(logger.DISABLED)

    # Create envs.
    env = omni.instantiate()

    reward_space = env.reward_space
    env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), "%i.monitor.json"%rank))
    gym.logger.setLevel(logging.WARN)

    if evaluation and rank==0:
        eval_env = omni.instantiate()
        eval_env = Monitor(eval_env, os.path.join(logger.get_dir(), 'gym_eval'))
        env = Monitor(env, None)
    else:
        eval_env = None

    # Parse noise_type
    action_noise = None
    param_noise = None
    nb_actions = env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Configure components.
    memory = Memory(
        limit=int(1e4),
        action_shape=env.action_space.shape,
        observation_shape=env.observation_space.shape,
        reward_shape=reward_space.shape
    )
    critic = Critic(layer_norm=layer_norm)
    proto_critic = ProtoCritic(layer_norm=layer_norm)
    actor = Actor(nb_actions, layer_norm=layer_norm)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()



    training.train(
        env=env,
        eval_env=eval_env,
        param_noise=param_noise,
        action_noise=action_noise,
        actor=actor,
        critic=critic,
        proto_critic=proto_critic,
        memory=memory,
        action_embedding=env.action_embedding,
        **kwargs)




    env.close()
    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))





