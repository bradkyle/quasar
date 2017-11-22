import tensorflow as tf
import omni



class Case():
    def __new__(cls, *args, **kwargs):
        case = super(Case, cls).__new__(cls)
        case._closed = False
        case.load_model = False
        return case

    def _render(self): raise NotImplementedError


    model_dir = None
    global_agent = None

    def close(self):
        pass

    def _setup_run(self):
        with tf.device("/cpu:0"):
            global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
            trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
            master_network = AC_Network(s_size, a_size, 'global', None)  # Generate global network
            num_workers = multiprocessing.cpu_count()  # Set workers ot number of available CPU threads
            workers = []
            # Create worker classes
            for i in range(num_workers):
                workers.append(Worker(DoomGame(), i, s_size, a_size, trainer, model_path, global_episodes))
            saver = tf.train.Saver(max_to_keep=5)


    def run(self):
        assert self.model_dir is not None
        assert self.global_agent is not None


        # Run
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            if self.load_model == True:
                print('Loading Model...')
                ckpt = tf.train.get_checkpoint_state(self.model_dir)
                saver.restore(sess, ckpt.model_checkpoint_path)

            # This is where the asynchronous magic happens.
            # Start the "work" process for each worker in a separate threat.
            worker_threads = []
            for worker in workers:
                worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver)
                t = threading.Thread(target=(worker_work))
                t.start()
                sleep(0.5)
                worker_threads.append(t)
            coord.join(worker_threads)

    def config(self):
        return NotImplemented

    def render(self):
        return self._render()

    def save(self, sess):
        saver = tf.train.Saver()
        save_path = saver.save(sess, self.model_dir)
        print("Model saved in : %s" % save_path)


class Worker():
    def __new__(cls, *args, **kwargs):
        worker = super(Worker, cls).__new__(cls)
        worker._closed = False
        worker.id = id
        return worker

    agent = None

    def _work(self, sess): raise NotImplementedError
    def _train(self, sess): raise NotImplementedError
    def _evaluate(self, sess): raise NotImplementedError

    def work(self, sess):
        assert self.agent is not None
        # USE CPU
        return self._work(sess)

    def train(self, sess):
        assert self.agent is not None
        # USE GPU IF PRESENT ELSE CPU
        return self._train(sess)

    def evaluate(self, sess):
        # USE CPU
        assert self.agent is not None
        return self._evaluate(sess)
