import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
from loss import *

config = {
    'vgg16': [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
    'vgg16-large': [[128, 128], [256, 256], [512, 512, 512], [1024, 1024, 1024], [1024, 1024, 1024]],
    'vgg19': [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]],
}

def make_layers_vgg_net(scope, input_x, config, dropout_rate=0.2, num_classes=100, is_training=None, 
                        batch_norm=False, layer_mask=None, scaling=1, scale=None, preact=False,
                        init='he'):
    modules = {}
    layers = [input_x]
    nc = 0
    if scale is None:
        scale = (0, scaling, scaling)
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        for block_id, sizes in enumerate(config):

            # conv units in one block
            for layer_id, channels in enumerate(sizes):
                h = layers[-1]
                nc += 1
                unit_name = 'l{}-conv{}.{}'.format(nc, block_id, layer_id)
                if layer_mask is not None:
                    if not layer_mask[nc - 2] and layer_mask[nc - 1]:
                        unit_name += 'graft'
                
                if nc < scale[0]:
                    scaling_val = scale[1]
                else:
                    scaling_val = scale[2]
                if init == 'normal':
                    weight_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
                elif init == 'uniform':
                    weight_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True)
                num_channels = int(channels * scaling_val)
                            
                h = slim.conv2d(h, num_outputs=num_channels, kernel_size=3, stride=1,
                    activation_fn=None, weights_initializer=weight_init,#tf.truncated_normal_initializer(stddev=0.01), 
                    biases_initializer=tf.zeros_initializer(), #tf.contrib.layers.xavier_initializer(uniform=False),
                    scope=unit_name)
                if batch_norm:
                    h = tf.layers.batch_normalization(h, training=is_training)
                pre = tf.identity(h)
                h = tf.nn.relu(h)
                if preact:
                    modules[unit_name] = h
                else:
                    modules[unit_name] = pre
                if layer_mask is not None:
                    if layer_mask[nc - 1]: 
                        #if preact:
                        #    layers.append(pre)
                        #else:
                        layers.append(h)
                else:
                    #if preact:
                    #    layers.append(pre)
                    #else:
                    layers.append(h)

            # pooling layers in one block
            h = layers[-1]
            unit_name = 'pooling{}'.format(block_id)
            h = slim.max_pool2d(h, [2, 2], scope=unit_name)
            modules[unit_name] = h
            #if layer_mask is not None:
            #    if layer_mask[nc - 1]:
            #        layers.append(h)
            #else:
            layers.append(h)

        h = slim.flatten(layers[-1])
        h = slim.dropout(h, dropout_rate, is_training=is_training, scope='dropout0')
        fc1 = slim.fully_connected(h, 4096, activation_fn=tf.nn.relu, scope='l{}-fc1'.format(nc + 1))
        fc1_drop = slim.dropout(fc1, dropout_rate, is_training=is_training, scope='dropout1')
        fc2 = slim.fully_connected(fc1_drop, 4096, activation_fn=tf.nn.relu, scope='l{}-fc2'.format(nc + 2))
        fc2_drop = slim.dropout(fc2, dropout_rate, is_training=is_training, scope='dropout2')
        out = slim.fully_connected(fc2_drop, num_classes, activation_fn=None, scope='l{}-output'.format(nc + 3))
        
        modules['l{}-fc1'.format(nc + 1)], modules['l{}-fc2'.format(nc + 2)], modules['out'] = fc1, fc2, out
        return modules
