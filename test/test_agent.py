import unittest
from agent.run import Worker
import tensorflow as tf
import baselines.common.tf_util as U

class TestWorker(unittest.TestCase):
    def setUp(self):
        self.worker = Worker(id=0, layer_norm=True, noise_type='adaptive-param_0.2')
        self.coord = tf.train.Coordinator()

    def test_rollout(self):
        with U.single_threaded_session() as sess:
            self.worker.agent.initialize(sess)
            self.worker.work(nb_epochs=1, nb_epoch_cycles=1,  nb_rollout_steps=1, nb_train_steps=0, nb_eval_steps=0, coord=self.coord, sess=sess)

    def test_train(self):
        with U.single_threaded_session() as sess:
            self.worker.agent.initialize(sess)
            self.worker.work(nb_epochs=1, nb_epoch_cycles=1,  nb_rollout_steps=1, nb_train_steps=1, nb_eval_steps=0, coord=self.coord, sess=sess)

    def test_eval(self):
        with U.single_threaded_session() as sess:
            self.worker.agent.initialize(sess)
            self.worker.work(nb_epochs=1, nb_epoch_cycles=1,  nb_rollout_steps=1, nb_train_steps=1, nb_eval_steps=1, coord=self.coord, sess=sess)

class TestAgent(unittest.TestCase):
   def setUp(self):
       return NotImplemented


class TestMemory(unittest.TestCase):
    def setUp(self):
        self.worker = Worker(id=0, layer_norm=True, noise_type='adaptive-param_0.2')
        self.coord = tf.train.Coordinator()