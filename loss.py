import tensorflow as tf


def get_regularizer_loss(weights, reg_type):
    if reg_type == 'l2':
        reg_loss = 0.0
        for weight in weights:
            if 'weight' in weight.name:
                reg_loss += tf.reduce_sum(weight * weight)
        return reg_loss
    else:
        raise NotImplemented
