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




class Worker():
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
                             candidate_action_shape= self.candidate_action_space.shape,
                             observation_shape=self.env.observation_space.shape,
                             reward_shape= self.env.reward_space.shape)

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
                self.action_noise = NormalActionNoise(mu=np.zeros(self.nb_param_actions), sigma=float(stddev) * np.ones(self.nb_param_actions))
            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                self.action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.nb_param_actions), sigma=float(stddev) * np.ones(self.nb_param_actions))
            else:
                raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))


        assert (np.abs(self.env.param_action_space.low) == self.env.param_action_space.high).all()  # we assume symmetric actions.
        max_param_action = self.env.param_action_space.high
        logger.info('scaling actions by {} before executing in env'.format(max_param_action))
        self.agent = Agent(
                     env = self.env,
                     shared = self.shared,
                     index_actor = self.index_actor,
                     index_critic = self.index_critic,
                     param_actor = self.param_actor,
                     param_critic = self.param_critic,
                     memory = self.memory,
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

        #logger.info('Using agent with the following configuration:')
        #logger.info(str(self.agent.__dict__.items()))

    def work(self, nb_epochs, nb_epoch_cycles, nb_rollout_steps, nb_train_steps, coord, sess):

        step = 0
        episode = 0
        episode_rewards_history = deque(maxlen=100)

        # Prepare everything.
        obs = np.zeros(self.env.observation_space.shape)
        rewards = np.zeros(self.env.reward_space.shape)

        done = False
        episode_reward = []
        episode_step = 0
        episodes = 0
        step = 0

        epoch = 0
        start_time = time.time()

        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_start_time = time.time()
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0

        while not coord.should_stop():
            for epoch in range(nb_epochs):
                for cycle in range(nb_epoch_cycles):
                    for t_rollout in range(nb_rollout_steps):

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
                        action = [index, list((np.asarray(param_action) + 1)/2)]

                        # Execute step in environment
                        new_obs, new_rewards, done, info = self.env.step(action)

                        step += 1
                        if self.rank == 0 and self.render:
                            self.env.render()
                        episode_reward.append(new_rewards)
                        episode_step += 1

                        # Book-keeping.
                        epoch_actions.append(action)
                        epoch_qs.append(q)
                        self.agent.store_transition(obs, new_obs, index_action, candidate_actions, param_action, rewards, new_rewards, done, step)
                        obs = new_obs
                        rewards = new_rewards

                        if done:
                            # Episode done.
                            epoch_episode_rewards.append(episode_reward)
                            episode_rewards_history.append(episode_reward)
                            epoch_episode_steps.append(episode_step)
                            episode_reward = 0.
                            episode_step = 0
                            epoch_episodes += 1
                            episodes += 1

                            self.agent.reset()
                            obs = self.env.reset()
                            rewards = self.env.reward_space.sample()

                    # Train.
                    epoch_index_actor_losses = []
                    epoch_index_critic_losses = []
                    epoch_param_actor_losses = []
                    epoch_param_critic_losses = []
                    epoch_adaptive_distances = []
                    for t_train in range(nb_train_steps):
                        # Adapt param noise, if necessary.
                        if self.memory.nb_entries >= self.batch_size and t % self.param_noise_adaption_interval == 0:
                            distance = self.agent.adapt_param_noise()
                            epoch_adaptive_distances.append(distance)
                        icl, ial, pcl, pal = self.agent.train()
                        epoch_index_actor_losses.append(icl)
                        epoch_index_critic_losses.append(ial)
                        epoch_param_actor_losses.append(pcl)
                        epoch_param_critic_losses.append(pal)
                        self.agent.update_target_net()



                    epoch_train_duration = time.time() - epoch_start_time
                    duration = time.time() - start_time
                    stats = self.agent.get_stats()
                    combined_stats = {}
                    for key in sorted(stats.keys()):
                        combined_stats[key] = mpi_mean(stats[key])

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

                    if self.rank == 0:
                        saver = tf.train.Saver()
                        save_path = saver.save(sess, self.save_path)
                        print("Model saved in : %s" % save_path)

                    #todo randomizer randomize normal env

        def train():
            return NotImplemented