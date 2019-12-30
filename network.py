import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
from loss import *

config = {
    'vgg16': [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
    'vgg16-large': [[128, 128], [256, 256], [512, 512, 512], [1024, 1024, 1024], [1024, 1024, 1024]],
    'vgg19': [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]],
}

def make_layers_vgg_net(scope, input_x, config, dropout_rate=0.2, num_classes=100, is_training=None, batch_norm=False):
    modules = {}
    layers = [input_x]
    nc = 0
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        for block_id, sizes in enumerate(config):

            # conv units in one block
            for layer_id, channels in enumerate(sizes):
                h = layers[-1]
                nc += 1
                unit_name = 'l{}-conv{}.{}'.format(nc, block_id, layer_id)
                h = slim.conv2d(h, num_outputs=channels, kernel_size=3, stride=1,
                    activation_fn=None, weights_initializer=tf.contrib.layers.variance_scaling_initializer(),#tf.truncated_normal_initializer(stddev=0.01), 
                    biases_initializer=tf.zeros_initializer(), #tf.contrib.layers.xavier_initializer(uniform=False),
                    scope=unit_name)
                if batch_norm:
                    h = tf.layers.batch_normalization(h, training=is_training)
                h = tf.nn.relu(h)
                modules[unit_name] = h
                layers.append(h)

            # pooling layers in one block
            h = layers[-1]
            unit_name = 'pooling{}'.format(block_id)
            h = slim.max_pool2d(h, [2, 2], scope=unit_name)
            modules[unit_name] = h
            layers.append(h)

        h = slim.flatten(layers[-1])
        h = slim.dropout(h, dropout_rate, is_training=is_training, scope='dropout0')
        fc1 = slim.fully_connected(h, 4096, activation_fn=tf.nn.relu, scope='l{}.fc1'.format(nc + 1))
        fc1_drop = slim.dropout(fc1, dropout_rate, is_training=is_training, scope='dropout1')
        fc2 = slim.fully_connected(fc1_drop, 4096, activation_fn=tf.nn.relu, scope='l{}.fc2'.format(nc + 2))
        fc2_drop = slim.dropout(fc2, dropout_rate, is_training=is_training, scope='dropout2')
        out = slim.fully_connected(fc2_drop, num_classes, activation_fn=None, scope='l{}.output'.format(nc + 3))

        modules['fc1'], modules['fc2'], modules['out'] = fc1, fc2, out
        return modules
