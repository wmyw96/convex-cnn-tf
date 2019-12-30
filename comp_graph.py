import tensorflow as tf
import numpy as np
from network import *
from loss import *


def show_params(domain, var_list):
    print('Domain {}:'.format(domain))
    for var in var_list:
        print('{}: {}'.format(var.name, var.shape))


def show_grads(domain, gd_list):
    print('Domain {}:'.format(domain))
    for (grad, var) in gd_list:
        if grad is not None:
            print(var.name)

def build_image_classfication_model(params):
    # build placeholder

    inp_x = tf.placeholder(dtype=tf.float32, 
                           shape=[None] + params['data']['x_size'],
                           name='x')
    label = tf.placeholder(dtype=tf.int64,
                           shape=[None],
                           name='label')
    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')
    lr_decay = tf.placeholder(dtype=tf.float32, shape=[], name='lr_decay')

    ph = {
        'x': inp_x,
        'y': label,
        'is_training': is_training,
        'lr_decay': lr_decay
    }

    # build computation graph
    nclass = params['data']['nclass']
    model_name = params['network']['model']    # vgg16, vgg16-large, vgg19 
    use_bn = params['network']['batch_norm']

    modules = \
        make_layers_vgg_net(scope=model_name,
                            input_x=inp_x, 
                            config=config[model_name],
                            num_classes=nclass,
                            dropout_rate=params['network']['dropout'],
                            is_training=is_training,
                            batch_norm=use_bn)

    graph = {
        'network': modules,
        'one_hot_y': tf.one_hot(label, nclass)
    }

    # fetch variables
    net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_name)
    show_params('network', net_vars)

    # build tragets
    logits = modules['out']
    ce_loss = tf.nn.softmax_cross_entropy_with_logits(labels=graph['one_hot_y'], 
        logits=logits, dim=1)
    ce_loss = tf.reduce_mean(ce_loss)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), ph['y']), tf.float32))   # [1,]
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=model_name)

    regularizer = get_regularizer_loss(net_vars, params['network']['regularizer'])

    loss = ce_loss + regularizer * params['network']['regw']
    
    #sl_op = tf.train.GradientDescentOptimizer(params['train']['lr'] * ph['lr_decay'])
    sl_op = tf.train.MomentumOptimizer(params['train']['lr'] * ph['lr_decay'], 0.9, use_nesterov=True)
    sl_grads = sl_op.compute_gradients(loss=loss, var_list=net_vars)
    sl_train_op = sl_op.apply_gradients(grads_and_vars=sl_grads)
    sl_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=model_name)
    
    show_grads('network', sl_grads)
    
    train = {
        'train': sl_train_op,
        'update': sl_update_ops,
        'overall_loss': loss,
        'ce_loss': ce_loss,
        'reg_loss': regularizer,
        'acc_loss': acc
    }
    #for gd, var in sl_grads:
    #    train[var.name + '_gd_loss'] = tf.reduce_mean(tf.abs(gd)) - 5e-4 * tf.reduce_mean(tf.abs(var))
    
    test = {
        'overall_loss': loss,
        'ce_loss': ce_loss,
        'reg_loss': regularizer,
        'acc_loss': acc
    }

    targets = {
        'train': train,
        'eval': test
    }

    return ph, graph, net_vars, targets


def build_grafting_onecut_model(params):
    # build placeholder

    inp_x = tf.placeholder(dtype=tf.float32, 
                           shape=[None] + params['data']['x_size'],
                           name='x')
    label = tf.placeholder(dtype=tf.int64,
                           shape=[None],
                           name='label')
    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')
    lr_decay = tf.placeholder(dtype=tf.float32, shape=[], name='lr_decay')

    ph = {
        'x': inp_x,
        'y': label,
        'is_training': is_training,
        'lr_decay': lr_decay
    }

    # build computation graph
    nclass = params['data']['nclass']
    model_name = params['network']['model']    # vgg16, vgg16-large, vgg19 
    use_bn = params['network']['batch_norm']
    domains = ['grafting', 'net1', 'net2']

    modules = {}
    graph = {
        'one_hot_y': tf.one_hot(label, nclass)
    }
    for domain in domains:
        modules[domain] = \
            make_layers_vgg_net(scope=domain,
                                input_x=inp_x, 
                                config=config[model_name],
                                num_classes=nclass,
                                dropout_rate=params['network']['dropout'],
                                is_training=is_training,
                                batch_norm=use_bn)
        graph[domain] = modules[domain]

    net_vars = {}
    save_vars = []
    # fetch variables
    for domain in domains:
        net_vars[domain] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
            scope=domain)
        show_params(domain, net_vars[domain])
        save_vars += net_vars[domain]

    targets = {}
    # two default-training domain
    for domain in ['net1', 'net2']:

        # build tragets
        logits = modules[domain]['out']
        ce_loss = tf.nn.softmax_cross_entropy_with_logits(labels=graph['one_hot_y'], 
            logits=logits, dim=1)
        ce_loss = tf.reduce_mean(ce_loss)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), ph['y']), tf.float32))   # [1,]

        regularizer = get_regularizer_loss(net_vars[domain], params['network']['regularizer'])

        loss = ce_loss + regularizer * params['network']['regw']
    
        sl_op = tf.train.MomentumOptimizer(params['train']['lr'] * ph['lr_decay'], 0.9, use_nesterov=True)
        sl_grads = sl_op.compute_gradients(loss=loss, var_list=net_vars[domain])
        sl_train_op = sl_op.apply_gradients(grads_and_vars=sl_grads)
        sl_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=domain)
    
        show_grads('network', sl_grads)
    
        train = {
            'train': sl_train_op,
            'update': sl_update_ops,
            'overall_loss': loss,
            'ce_loss': ce_loss,
            'reg_loss': regularizer,
            'acc_loss': acc
        }
        #for gd, var in sl_grads:
        #    train[var.name + '_gd_loss'] = tf.reduce_mean(tf.abs(gd)) - 5e-4 * tf.reduce_mean(tf.abs(var))
    
        test = {
            'overall_loss': loss,
            'ce_loss': ce_loss,
            'reg_loss': regularizer,
            'acc_loss': acc
        }

        targets[domain] = {
            'train': train,
            'test': test
        }
    # assign weights
    for domain in ['net1', 'net2']:
        


    return ph, graph, save_vars, net_vars, targets





