import os

import time
import baselines.common.tf_util as U
from agent.run import Worker
import multiprocessing
import threading
import argparse
from  common.misc_util import boolean_flag

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

boolean_flag(parser, 'render-eval', default=False)
boolean_flag(parser, 'layer-norm', default=True)
boolean_flag(parser, 'render', default=False)
boolean_flag(parser, 'normalize-returns', default=False)
boolean_flag(parser, 'normalize-observations', default=True)
boolean_flag(parser, 'load_model', default=True)
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--save-path', help='Model directory', type=str, default='./data/model.ckpt')
parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
parser.add_argument('--actor-lr', type=float, default=1e-4)
parser.add_argument('--critic-lr', type=float, default=1e-3)
boolean_flag(parser, 'popart', default=False)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--reward-scale', type=float, default=1.)
parser.add_argument('--clip-norm', type=float, default=None)
parser.add_argument('--nb-epochs', type=int, default=500)  # with default settings, perform 1M steps total
parser.add_argument('--nb-epoch-cycles', type=int, default=20)
parser.add_argument('--nb-train-steps', type=int, default=50)  # per epoch cycle and MPI worker
parser.add_argument('--nb-rollout-steps', type=int, default=100)  # per epoch cycle and MPI worker
parser.add_argument('--noise-type', type=str, default='adaptive-param_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
boolean_flag(parser, 'evaluation', default=False)
args = vars(parser.parse_args())

if __name__ == '__main__':

    import tensorflow as tf

    with tf.device("/cpu:0"):
        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        num_workers = 1  # multiprocessing.cpu_count()
        saver = tf.train.Saver()
        workers = []
        for i in range(num_workers):
            workers.append(Worker(id=i, layer_norm=True, noise_type='adaptive-param_0.2'))

    with U.single_threaded_session() as sess:
        coord = tf.train.Coordinator()

        if args['load_model'] == True and os.path.exists(args['save_path']):
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(args['save_path'])
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Starting without loading model...")



        worker_threads = []
        for worker in workers:
            worker.agent.initialize(sess)
            worker_work = lambda: worker.work(sess=sess,
                                              nb_epochs=args['nb_epochs'],
                                              nb_epoch_cycles=args['nb_epoch_cycles'],
                                              nb_rollout_steps=args['nb_rollout_steps'],
                                              nb_train_steps=args['nb_train_steps'],
                                              coord=coord)
            t = threading.Thread(target=(worker_work))
            t.start()
            time.sleep(0.2)
            worker_threads.append(t)
        coord.join(worker_threads)
