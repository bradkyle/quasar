import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import math_ops

class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm

    def __call__(self, input, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = input
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x


class Critic(Model):
    def __init__(self, name='critic', layer_norm=True, n=1):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.n = n

    def __call__(self, input, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = input
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.concat([x, action], axis=-1)
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, self.n, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars


class Shared(Model):
    def __init__(self, name='shared', layer_norm=True):
        super(Shared, self).__init__(name=name)
        self.layer_norm = layer_norm

    def __call__(self, input, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = input
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

        return x


class VDCNN(Model):
    def __init__(self, name="vdcnn",
                 permutations = 97, # char size
                 cnn_filter_size=3,
                 pooling_filter_size=2,
                 num_filters_per_size=(64,128,256,512),
                 num_rep_block=(16, 16, 16, 6)
                 ):
        super(VDCNN, self).__init__(name=name)
        self.permutations = permutations
        self.cnn_filter_size = cnn_filter_size
        self.pooling_filter_size = pooling_filter_size
        self.num_filters_per_size = num_filters_per_size
        self.num_rep_block = num_rep_block

    def __call__(self, input, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = input
            x = slim.conv2d(x, self.num_filters_per_size[0], [self.permutations, self.cnn_filter_size], normalizer_fn=slim.batch_norm, scope = 'conv0', padding='VALID')

            def resUnit(input_layer, num_filters_per_size_i, cnn_filter_size, i, j):
                with tf.variable_scope("res_unit_" + str(i) + "_" + str(j)):
                    part1 = slim.batch_norm(input_layer, activation_fn=None)
                    part2 = tf.nn.relu(part1)
                    part3 = slim.conv2d(part2, num_filters_per_size_i, [1, cnn_filter_size], activation_fn=None)
                    part4 = slim.batch_norm(part3, activation_fn=None)
                    part5 = tf.nn.relu(part4)
                    part6 = slim.conv2d(part5, num_filters_per_size_i, [1, cnn_filter_size], activation_fn=None)
                    output = input_layer + part6
                    return output

            for i in range(0,len(self.num_filters_per_size)):
                for j in range(0,self.num_rep_block[i]):
                    x = resUnit(x, self.num_filters_per_size[i], self.cnn_filter_size, i, j)
                x = slim.max_pool2d(x, [1,self.pooling_filter_size], scope='pool_%s' % i)

            x = math_ops.reduce_mean(x, [1, 2], name='pool5', keep_dims=True)

            return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars

class DNC():
    def __init__(self):
        raise NotImplemented

class VDCNNDNC():
    def __init__(self):
        raise NotImplemented