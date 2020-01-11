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


def getw_l1_norm(weight):
    shape = weight.get_shape()
    if len(shape) == 2:
        ra = [1]
    else:
        ra = [0, 1, 3]
    value = tf.reduce_mean(tf.abs(weight), axis=ra)
    return value


def getw_l2_norm(weight):
    shape = weight.get_shape()
    if len(shape) == 2:
        ra = [1]
    else:
        ra = [0, 1, 3]
    value = tf.reduce_mean(tf.square(weight), axis=ra)
    return value


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


'''
fork from slackoverflow answer @Yamaneko:
https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
'''

def calculate_pairwise_l2_dist(A, B):
    assert A.shape.as_list() == B.shape.as_list()
    row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
    row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.
    row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
    row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.
    return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B


def calculate_pairwise_l2_ndist(x, y):
    nx = x / (tf.reduce_std(x, 0, keepdims=True) + 1e-9)
    ny = y / (tf.reduce_std(y, 0, keepdims=True) + 1e-9)
    return calculate_pairwise_l2_dist(x, y)


def compute_gradient_l2_norm(y, x):
    print('compute_gradient_l2_norm: y shape = {}'.format(y.get_shape()))
    grad = 0.0
    for i in range(int(y.get_shape()[-1]))
        dimi_grad = tf.gradient(y[:, i], x)
        grad += tf.square(dimi_grad)
    return grad
