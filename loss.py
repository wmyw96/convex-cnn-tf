import tensorflow as tf
import numpy as np


def calc_l12_norm(weight):
    shape = weight.get_shape()
    print('calculate l12 norm: variable {}, shape = {}'.format(weight.name, shape))
    if len(shape) == 2:
        c_in = int(shape[0])
        l1 = tf.reduce_mean(tf.abs(weight) * np.sqrt(c_in), 1)
        return tf.reduce_mean(l1 * l1)
    else:
        c_in = int(shape[2])
        l1 = tf.reduce_mean(tf.abs(weight) * np.sqrt(c_in), (0, 1, 3))
        return tf.reduce_mean(l1 * l1)


def get_regularizer_loss(weights, reg_type):
    if reg_type == 'l2':
        reg_loss = 0.0
        for weight in weights:
            #if 'weight' in weight.name:
            reg_loss += 0.5 * tf.reduce_sum(weight * weight)
        return reg_loss
    elif reg_type == 'l12':
        reg_loss = 0.0
        for weight in weights:
            if 'weight' in weight.name:
                reg_loss += calc_l12_norm(weight)
        return reg_loss
    else:
        raise NotImplemented


def compute_kernel_lst(x, y, sigma):
    # x: [... , nx]
    # y: [... , ny]
    rk = int(x.shape.ndims)
    nx = int(x.get_shape()[rk - 1])
    ny = int(y.get_shape()[rk - 1])
    print('compute_kernel rk = {}, nx = {}, ny = {}'.format(rk, nx, ny))
    tilde_x = tf.tile(tf.expand_dims(x, rk), tf.stack([1] * rk + [ny]))
    tilde_y = tf.tile(tf.expand_dims(y, rk - 1), tf.stack([1] * (rk-1) + [nx, 1]))
    l2_dist = tf.square(tilde_x - tilde_y) / (2 * sigma**2)
    return tf.reduce_mean(tf.exp(-l2_dist))


def mmd_loss_lst(x, y, sigmas):
    mmd_loss_avg = 0.
    for sigma in sigmas:
        x_kernel = compute_kernel_lst(x, x, sigma)
        y_kernel = compute_kernel_lst(y, y, sigma)
        xy_kernel = compute_kernel_lst(x, y, sigma)
        mmd_loss_avg += tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)
    return mmd_loss_avg
