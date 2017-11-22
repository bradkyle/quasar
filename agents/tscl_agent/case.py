import logging
import os
import time
from agents.tscl_agent.memory import Memory
import agents.common.logger as logger
from agents.tscl_agent.models import Shared, Critic, Actor
from agents.tscl_agent.noise import *
import omni
from agents.tscl_agent.util import mpi_mean, mpi_std, mpi_max, mpi_sum
import gym
import pickle
import tensorflow as tf
from collections import deque
import agents.common.tf_util as U
from agents.tscl_agent.agent import Agent


class TSCL_case():
    def __init__(self,
                 id,
                 layer_norm,
                 noise_type,
                 render=False,
                 evaluation=False,
                 k=5,
                 gamma=0.99,
                 tau=0.001,
                 normalize_observations=True,
                 param_noise_adaption_interval=50,
                 normalize_returns=False,
                 action_noise=None,
                 param_noise=None,
                 reward_scale=1,
                 clip_norm=None,
                 enable_popart=False,
                 batch_size=128,
                 save_path='./data/model.ckpt',
                 render_eval=False
                 ):
        self.model_dir = "/agents/tscl_agent/model/"

        self.render_eval = render_eval
        self.param_noise_adaption_interval = param_noise_adaption_interval
        self.batch_size = batch_size
        self.render = render
        self.id = id
        self.rank = id
        self.name = "worker_" + str(id)
        self.evaluation = evaluation
        self.env = omni.instantiate()
        self.eval_env = None
        self.k = k
        self.save_path = save_path
        self.max_param_action = self.env.param_action_space.high
        self.max_index_action = self.env.index_action_space.high

        gym.logger.setLevel(logging.WARN)

        if self.rank != 0: logger.set_level(logger.DISABLED)

        self.candidate_action_space = self.env.candidate_action_space(self.k)

        self.memory = Memory(limit=int(1e4),
                             index_action_shape=self.env.index_action_space.shape,
                             param_action_shape=self.env.param_action_space.shape,
                             candidate_action_shape=self.candidate_action_space.shape,
                             observation_shape=self.env.observation_space.shape,
                             reward_shape=self.env.reward_space.shape)

        self.nb_param_actions = self.env.param_action_space.shape[-1]
        self.shared = Shared(layer_norm=layer_norm, name="shared")
        self.index_critic = Critic(layer_norm=layer_norm, n=k, name="index_critic")
        self.index_actor = Actor(1, layer_norm=layer_norm, name="index_actor")
        self.param_critic = Critic(layer_norm=layer_norm, name="param_critic")
        self.param_actor = Actor(self.nb_param_actions, layer_norm=layer_norm, name="param_actor")

        # if evaluation and self.rank == 0:
        #     eval_env = omni.instantiate()
        #     eval_env = Monitor(eval_env, os.path.join(logger.get_dir(), 'gym_eval'))
        #     env = Monitor(eval_env, None)
        # else:
        #     eval_env = None

        # Parse noise_type
        self.action_noise = None
        self.param_noise = None
        for current_noise_type in noise_type.split(','):
            current_noise_type = current_noise_type.strip()
            if current_noise_type == 'none':
                pass
            elif 'adaptive-param' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                self.param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev),
                                                          desired_action_stddev=float(stddev))
            elif 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                self.action_noise = NormalActionNoise(mu=np.zeros(self.nb_param_actions),
                                                      sigma=float(stddev) * np.ones(self.nb_param_actions))
            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                self.action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.nb_param_actions),
                                                                 sigma=float(stddev) * np.ones(self.nb_param_actions))
            else:
                raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

        assert (np.abs(
            self.env.param_action_space.low) == self.env.param_action_space.high).all()  # we assume symmetric actions.
        max_param_action = self.env.param_action_space.high
        logger.info('scaling actions by {} before executing in env'.format(max_param_action))

        self.agent = Agent(
            env=self.env,
            shared=self.shared,
            index_actor=self.index_actor,
            index_critic=self.index_critic,
            param_actor=self.param_actor,
            param_critic=self.param_critic,
            memory=self.memory,
            param_action_shape=self.env.param_action_space.shape,
            rewards_shape=self.env.reward_space.shape,
            candidates_shape=self.k,
            gamma=gamma,
            tau=tau,
            normalize_returns=normalize_returns,
            normalize_observations=normalize_observations,
            batch_size=batch_size,
            action_noise=action_noise,
            param_noise=param_noise,
            enable_popart=enable_popart,
            clip_norm=clip_norm,
            reward_scale=reward_scale,
            index_actor_lr=1e-4,
            index_critic_lr=1e-3,
            param_actor_lr=1e-4,
            param_critic_lr=1e-3
        )

        self.step = 0
        self.episode = 0
        self.episode_rewards_history = deque(maxlen=100)

        # Prepare everything.
        self.obs = np.zeros(self.env.observation_space.shape)
        self.rewards = np.zeros(self.env.reward_space.shape)

        self.done = False
        self.episode_reward = []
        self.episode_step = 0
        self.episodes = 0
        self.step = 0

        self.epoch = 0
        self.start_time = time.time()

        self.epoch_episode_rewards = []
        self.epoch_episode_steps = []
        self.epoch_start_time = time.time()
        self.epoch_actions = []
        self.epoch_qs = []
        self.epoch_episodes = 0

        self.epoch_index_actor_losses = []
        self.epoch_index_critic_losses = []
        self.epoch_param_actor_losses = []
        self.epoch_param_critic_losses = []
        self.epoch_adaptive_distances = []



    def _reset(self):
        obs = np.zeros(self.env.observation_space.shape)
        rewards = np.zeros(self.env.reward_space.shape)
        return obs, rewards

    def _work(self):

            obs, rewards = self._reset()

            # Predict next action.
            index_action, param_action, q = self.agent.pi(obs, apply_noise=True, compute_Q=True)
            assert param_action.shape == self.env.param_action_space.shape
            assert index_action.shape == self.env.index_action_space.shape
            assert self.max_param_action.shape == param_action.shape
            assert self.max_index_action.shape == index_action.shape

            if self.rank == 0 and self.render:
                self.env.render()

            # spawns k closest actions in action embedding and thereafter
            # determines most favorable of the candidates based on
            # predicted future reward.
            candidate_actions = self.env.spawn(index_action, self.k)
            index = self.agent.val(obs, candidate_actions)
            action = [index, list((np.asarray(param_action) + 1) / 2)]

            # Execute step in environment
            print("stepping")
            new_obs, new_rewards, done, info = self.env.step(action)

            self.step += 1
            if self.rank == 0 and self.render:
                self.env.render()
            self.episode_reward.append(new_rewards)
            self.episode_step += 1

            # Book-keeping.
            self.epoch_actions.append(action)
            self.epoch_qs.append(q)
            self.agent.store_transition(obs, new_obs, index_action, candidate_actions, param_action, rewards,
                                        new_rewards, done, self.step)
            obs = new_obs
            rewards = new_rewards

            if done:
                # Episode done.
                self.epoch_episode_rewards.append(self.episode_reward)
                self.episode_rewards_history.append(self.episode_reward)
                self.epoch_episode_steps.append(self.episode_step)
                self.episode_reward = 0.
                self.episode_step = 0
                self.epoch_episodes += 1
                self.episodes += 1

                self.agent.reset()
                obs, rewards = self._reset()

            print("done")

    def _train(self):
        # Adapt param noise, if necessary.
        if self.memory.nb_entries >= self.batch_size and t % self.param_noise_adaption_interval == 0:
            distance = self.agent.adapt_param_noise()
            self.epoch_adaptive_distances.append(distance)
        icl, ial, pcl, pal = self.agent.train()
        self.epoch_index_actor_losses.append(icl)
        self.epoch_index_critic_losses.append(ial)
        self.epoch_param_actor_losses.append(pcl)
        self.epoch_param_critic_losses.append(pal)
        self.agent.update_target_net()


    def _render(self):
        stats = self.agent.get_stats()
        combined_stats = {}
        for key in sorted(stats.keys()):
            combined_stats[key] = mpi_mean(stats[key])

            # Time related
        combined_stats['duration/epoch'] = epoch_train_duration

        # Time related
        combined_stats['duration/epoch'] = epoch_train_duration

        # Rollout statistics.
        combined_stats['rollout/return'] = mpi_mean(epoch_episode_rewards)
        combined_stats['rollout/return_history'] = mpi_mean(np.mean(episode_rewards_history))
        combined_stats['rollout/episode_steps'] = mpi_mean(epoch_episode_steps)
        combined_stats['rollout/episodes'] = mpi_sum(epoch_episodes)
        combined_stats['rollout/actions_mean'] = mpi_mean(epoch_actions)
        combined_stats['rollout/actions_std'] = mpi_std(epoch_actions)
        combined_stats['rollout/Q_mean'] = mpi_mean(epoch_qs)

        # Train statistics.
        combined_stats['train/loss_index_actor'] = mpi_mean(epoch_index_actor_losses)
        combined_stats['train/loss_index_critic'] = mpi_mean(epoch_index_critic_losses)
        combined_stats['train/loss_param_actor'] = mpi_mean(epoch_param_actor_losses)
        combined_stats['train/loss_param_critic'] = mpi_mean(epoch_param_critic_losses)
        combined_stats['train/param_noise_distance'] = mpi_mean(epoch_adaptive_distances)

        # Total statistics.
        combined_stats['total/duration'] = mpi_mean(duration)
        combined_stats['total/steps_per_second'] = mpi_mean(float(t) / float(duration))
        combined_stats['total/episodes'] = mpi_mean(episodes)
        combined_stats['total/epochs'] = epoch + 1
        combined_stats['total/steps'] = step

        # Env Statistics
        combined_stats['projected_return/current_balance/day'] = NotImplemented
        combined_stats['projected_return/current_balance/week'] = NotImplemented
        combined_stats['projected_return/current_balance/month'] = NotImplemented
        combined_stats['projected_return/current_balance/year'] = NotImplemented

        combined_stats['projected_return/100_dollars/day'] = NotImplemented
        combined_stats['projected_return/100_dollars/week'] = NotImplemented
        combined_stats['projected_return/100_dollars/month'] = NotImplemented
        combined_stats['projected_return/100_dollars/year'] = NotImplemented

        combined_stats['projected_return/1000_dollars/day'] = NotImplemented
        combined_stats['projected_return/1000_dollars/week'] = NotImplemented
        combined_stats['projected_return/1000_dollars/month'] = NotImplemented
        combined_stats['projected_return/1000_dollars/year'] = NotImplemented

        combined_stats['projected_return/10000_dollars/day'] = NotImplemented
        combined_stats['projected_return/10000_dollars/week'] = NotImplemented
        combined_stats['projected_return/10000_dollars/month'] = NotImplemented
        combined_stats['projected_return/10000_dollars/year'] = NotImplemented

        combined_stats['profit/time'] = NotImplemented
        combined_stats['profit/step'] = NotImplemented
        combined_stats['profit/expenditure'] = NotImplemented

        combined_stats['account/balances'] = NotImplemented

        for key in sorted(combined_stats.keys()):
            logger.record_tabular(key, combined_stats[key])
        logger.dump_tabular()
        logger.info('')
        logdir = logger.get_dir()
        if self.rank == 0 and logdir:
            if hasattr(self.env, 'get_state'):
                with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                    pickle.dump(self.env.get_state(), f)


