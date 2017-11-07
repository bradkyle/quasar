import json
import unittest
from tscl_agent.run import Worker
import tensorflow as tf
import baselines.common.tf_util as U
from tscl_agent.models import Shared, Critic, Actor
from tscl_agent.memory import Memory
from tscl_agent.agent import Agent
from omni import omni
import numpy as np
from omni.config import MAX_PARAMS

np.set_printoptions(0)


class TestWorker(unittest.TestCase):
    def setUp(self):
        # self.worker = Worker(id=0, layer_norm=True, noise_type='adaptive-param_0.2')
        # self.coord = tf.train.Coordinator()
        return NotImplemented

    def test_rollout(self):
        #with U.single_threaded_session() as sess:
        #    self.worker.agent.initialize(sess)
        #    self.worker.work(nb_epochs=1, nb_epoch_cycles=1,  nb_rollout_steps=1, nb_train_steps=0, coord=self.coord, sess=sess)
        return NotImplemented

    def test_train(self):
        #with U.single_threaded_session() as sess:
        #    self.worker.agent.initialize(sess)
        #    self.worker.work(nb_epochs=1, nb_epoch_cycles=1,  nb_rollout_steps=1, nb_train_steps=1, coord=self.coord, sess=sess)
        return NotImplemented

    def test_eval(self):
        #with U.single_threaded_session() as sess:
        #    self.worker.agent.initialize(sess)
        #    self.worker.work(nb_epochs=1, nb_epoch_cycles=1,  nb_rollout_steps=1, nb_train_steps=1, coord=self.coord, sess=sess)
        return NotImplemented

class TestAgent(unittest.TestCase):
   def setUp(self):
       self.k = 5
       self.layer_norm = True
       self.batch_size = 128
       self.env = omni.instantiate()
       self.nb_param_actions = self.env.param_action_space.shape[-1]
       self.candidate_action_space = self.env.candidate_action_space(self.k)
       self.memory =  Memory(limit=int(1e4),
                             index_action_shape=self.env.index_action_space.shape,
                             param_action_shape=self.env.param_action_space.shape,
                             candidate_action_shape= self.candidate_action_space.shape,
                             observation_shape=self.env.observation_space.shape,
                             reward_shape= self.env.reward_space.shape)

       self.shared = Shared(layer_norm=self.layer_norm, name="shared")
       self.index_critic = Critic(layer_norm=self.layer_norm, n=self.k, name="index_critic")
       self.index_actor = Actor(1, layer_norm=self.layer_norm, name="index_actor")
       self.param_critic = Critic(layer_norm=self.layer_norm, name="param_critic")
       self.param_actor = Actor(self.nb_param_actions, layer_norm=self.layer_norm, name="param_actor")

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
           candidates_shape=self.k
       )



       with open('./test/test.json') as data_file:
           self.data = json.load(data_file)

           for d in self.data:
               self.agent.memory.append(obs=d["obs"],
                                  new_obs=d["new_obs"],
                                  index_action=d["index_action"],
                                  candidate_actions=d["candidate_action"],
                                  param_action=d["param_action"][0],
                                  rewards=d["rewards"],
                                  new_rewards=d["new_rewards"],
                                  done=d["done"],
                                  step=d["step"]
                                  )




   def test_qi(self):
       with U.single_threaded_session() as sess:
           batch = self.agent.memory.sample(batch_size=self.batch_size)

           reward_delta, abs_reward, greed_index, reward0, reward1, gathered = sess.run(
               fetches=[
                        self.agent.reward_delta,
                        self.agent.abs_reward,
                        self.agent.greed_index,
                        self.agent.rewards0,
                        self.agent.rewards1,
                        self.agent.gathered,
               ],
               feed_dict={
                   self.agent.rewards1: batch['new_rewards'],
                   self.agent.rewards0: batch['rewards']
               }
           )

           print("--------------------------------------------------------------------------")
           print("reward: " + str(reward0), str(reward0.shape))
           print("--------------------------------------------------------------------------")
           print("new reward: " + str(reward1), str(reward1.shape))
           print("--------------------------------------------------------------------------")
           print("reward delta: "+str(reward_delta), str(reward_delta.shape))
           print("--------------------------------------------------------------------------")
           print("absolute reward: "+str(abs_reward), str(abs_reward.shape))
           print("--------------------------------------------------------------------------")
           print("greed index: "+str(greed_index), str(greed_index.shape))
           print("--------------------------------------------------------------------------")
           print("gathered: " + str(gathered[0]), str(gathered.shape))

           reward = self.agent.qi(batch['new_rewards'][1], batch['rewards'][1])
           assert reward in batch['new_rewards'][1]

           sess.close()

   def test_train(self):
       with U.single_threaded_session() as sess:
           self.agent.initialize(sess)
           icl, ial, pcl, pal = self.agent.train()
