import tensorflow as tf
import numpy as np
from network import *
from loss import *


def random_select(x, m):
    n = int(x.get_shape()[-1])
    prob = [1.0/n] * n
    dist = tf.contrib.distributions.Categorical(tf.log(prob))
    samples = dist.sample(m)
    vv = tf.gather(x, samples, axis=int(x.shape.ndims) - 1)
    print(samples.get_shape())
    print('random select {}'.format(vv.get_shape()))
    return vv


def fixed_random_select(x, m):
    samples = np.random.choice(x.get_shape()[-1], m)
    print('fixed random select = {}'.format(samples))
    xsamples = tf.gather(x, samples, axis=int(x.shape.ndims) - 1)
    return xsamples


def allocate_channels(channels, pweight):
    total_left = channels
    alloc = []
    for i in range(len(pweight) - 1):
        cur = int(pweight[i] * channels)
        alloc.append(cur)
        total_left -= cur
    alloc.append(total_left)
    return alloc


def show_params(domain, var_list):
    print('Domain {}:'.format(domain))
    for var in var_list:
        print('{}: {}'.format(var.name, var.shape))


def show_grads(domain, gd_list):
    print('Grad Domain {}:'.format(domain))
    for (grad, var) in gd_list:
        if grad is not None:
            print(var.name)

def create_var_dict(var_list):
    var_dict = {}
    for var in var_list:
        var_dict[var.name] = var
    return var_dict

def find_layer_feature_map(modules, name):
    for key, value in modules.items():
        if name in key:
            print('Find Layer {} Feature Map: shape = {}'.format(name, value.shape))
            return value

def find_weight(var_list, name):
    for weight in var_list:
        if name in weight.name and 'weight' in weight.name:
            print('Hit weight name [{}] in dict: name = {}'.format(name, weight.name))
            return weight


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
    inter_w = tf.placeholder(dtype=tf.float32, shape=[], name='interpolation_weight')
    ph = {
        'x': inp_x,
        'y': label,
        'is_training': is_training,
        'lr_decay': lr_decay,
        'inter_weight': inter_w
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
        mask = params['network']['layer_mask']
        scale = None
        if 'graft' in domain:
            mask = params['grafting']['layer_mask']
        if 'diff_scale' in params['grafting']:
            if domain in set(['net1', 'net2']):
                scale = (0, params['grafting']['diff_scale'][domain], 
                    params['grafting']['diff_scale'][domain])
            else:
                l = params['grafting']['nanase']
                scale = (l, params['grafting']['diff_scale']['net1'], 
                    params['grafting']['diff_scale']['net2'])
        modules[domain] = \
            make_layers_vgg_net(scope=domain,
                                input_x=inp_x, 
                                config=config[model_name],
                                num_classes=nclass,
                                dropout_rate=params['network']['dropout'],
                                is_training=is_training,
                                batch_norm=use_bn,
                                layer_mask=mask,
                                scale=scale)
        graph[domain] = modules[domain]

    net_vars = {}
    save_vars = []
    # fetch variables
    for domain in domains:
        net_vars[domain] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
            scope=domain)
        if not ('graft' in domain):
            net_vars[domain] = [weight for weight in net_vars[domain] if 'graft' not in weight.name]
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
    
        show_grads(domain, sl_grads)
    
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
            'eval': test
        }

    var_dict = create_var_dict(net_vars['grafting'])
    targets['grafting'] = {}
    # assign weights
    for domain in ['net1', 'net2']:
        op_name = 'assign_{}'.format(domain)
        targets['grafting'][op_name] = {}
        
        for layer_id in range(params['grafting']['nlayers']):
            # fetch all weights in layer l
            if layer_id + 1 <= params['grafting']['nanase'] and domain == 'net2':
                continue
            if layer_id + 1 >= params['grafting']['nanase'] and domain == 'net1':
                continue
            weights = []
            for weight in net_vars[domain]:
                if ('l{}-'.format(layer_id + 1)) in weight.name:
                    weights.append(weight)
            if not params['grafting']['layer_mask'][layer_id]:
                continue
            if layer_id >= 1 and not params['grafting']['layer_mask'][layer_id - 1]:
                continue
            assigns = []
            for weight in weights:
                weight_name = weight.name[len(domain)+1:]
                print('drafting variable {}'.format(var_dict['grafting/' + weight_name].name))
                assigns.append(tf.assign(var_dict['grafting/' + weight_name], weight))
            print('Assign Model')
            print(assigns)
            targets['grafting'][op_name]['l{}'.format(layer_id + 1)] = assigns
    
    # fetch trainable weights
    layer_name = 'l{}-'.format(params['grafting']['nanase'])
    train_weights = [weight for weight in net_vars['grafting'] if layer_name in weight.name]
    show_params('grafting', train_weights)

    # to check
    logits = modules['grafting']['out']
    ce_loss = tf.nn.softmax_cross_entropy_with_logits(labels=graph['one_hot_y'], 
        logits=logits, dim=1)
    ce_loss = tf.reduce_mean(ce_loss)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), ph['y']), tf.float32))   # [1,]
    
    net2_feature = find_layer_feature_map(modules['net2'], layer_name)
    graft_feature = find_layer_feature_map(modules['grafting'], layer_name)

    regularizer = get_regularizer_loss(net_vars['grafting'], params['network']['regularizer'])
    l2_diff = tf.reduce_mean(tf.square(net2_feature - graft_feature))
    #loss = '''regularizer * params['network']['regw']''' + l2_diff * params['grafting']['diffw']
    loss = l2_diff * params['grafting']['diffw']
    if 'use_adam' in params['grafting']:
        graft_op = tf.train.AdamOptimizer(params['grafting']['adam_lr'] * ph['lr_decay'])
    else:    
        graft_op = tf.train.MomentumOptimizer(params['grafting']['lr'] * ph['lr_decay'], 0.9, use_nesterov=True)
    graft_grads = graft_op.compute_gradients(loss=loss, var_list=train_weights)
    graft_train_op = graft_op.apply_gradients(grads_and_vars=graft_grads)
    graft_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='grafting')
    
    show_grads('grafting', sl_grads)
    
    targets['grafting']['train'] = {
        'train': graft_train_op,
        'update': graft_update_ops,
        'overall_loss': loss,
        'ce_loss': ce_loss,
        'reg_loss': regularizer,
        'acc_loss': acc,
        'l2diff_loss': l2_diff
    }

    targets['grafting']['eval'] = {
        'overall_loss': loss,
        'ce_loss': ce_loss,
        'reg_loss': regularizer,
        'acc_loss': acc,
        'l2_diff': l2_diff
    }
    def reduce_value(vvalue):
        shape = vvalue.get_shape()
        if len(shape) == 2:
            return tf.reduce_mean(vvalue, 0)
        elif len(shape) == 4:
            return tf.reduce_mean(vvalue, [0, 1, 2])
    targets['cp2net'] = {}
    # comparison of net1 and net2
    for layer_id in range(params['grafting']['nlayers'] - 1):
        layer_name = 'l{}-'.format(layer_id + 1)
        net1_feat = find_layer_feature_map(modules['net1'], layer_name)
        net2_feat = find_layer_feature_map(modules['net2'], layer_name)
        
        nchannels = int(net1_feat.get_shape()[-1])
        print('layer {} feature shape = {}, # channels = {}'.format(
            layer_id + 1, net1_feat.get_shape(), nchannels))
        net1_feat_c = tf.transpose(tf.reshape(net1_feat, [-1, nchannels]))
        net2_feat_c = tf.transpose(tf.reshape(net2_feat, [-1, nchannels]))

        l2_dist = compute_pairwise_l2_dist(net1_feat_c, net2_feat_c) / tf.cast(tf.shape(net1_feat_c)[1], tf.float32)
        l2_ndist = compute_pairwise_l2_ndist(net1_feat_c, net2_feat_c) / tf.cast(tf.shape(net2_feat_c)[1], tf.float32)
        assert l2_dist.shape.as_list() == [nchannels, nchannels]

        nextw_net1 = find_weight(net_vars['net1'], 'l{}-'.format(layer_id + 2))
        nextw_net2 = find_weight(net_vars['net2'], 'l{}-'.format(layer_id + 2))

        net1v = {
            #'gradnorm': grad_net1, #getw_l1_norm(nextw_net1), #grad_net1,
            'l1_norm': getw_l1_norm(nextw_net1),
            'l2_norm': getw_l2_norm(nextw_net1),
            'std': compute_std(net1_feat_c, 1)
        }
        net2v = {
            #'gradnorm': grad_net2, #getw_l1_norm(nextw_net2), #grad_net2,
            'l1_norm': getw_l1_norm(nextw_net2),
            'l2_norm': getw_l2_norm(nextw_net2),
            'std': compute_std(net2_feat_c, 1)
        }

        targets['cp2net']['l{}'.format(layer_id + 1)] = {
            'l2_dist': l2_dist,
            'l2_ndist': l2_ndist,
            'mmd_dist': normalized_mmd_loss(net1_feat_c, net2_feat_c, [0.1, 0.5, 1, 5]),
            'net1': net1v,
            'net2': net2v
        }
    net2_out_feat = find_layer_feature_map(modules['net2'], 'out')
    net2_out_std = compute_std(tf.transpose(net2_out_feat), 1)
    targets['cp2net']['l' + str(params['grafting']['nlayers'])] = {'net2': {'std': net2_out_std}}
    targets['cp2net']['graft_er'] = {}
    for i in range(params['grafting']['nanase'], params['grafting']['nlayers']):
        ground_truth = find_layer_feature_map(modules['net2'], 'l{}-'.format(i))
        predicted = find_layer_feature_map(modules['grafting'], 'l{}-'.format(i))
        targets['cp2net']['graft_er'][i] = reduce_value(tf.square(ground_truth - predicted))
    gt_out = find_layer_feature_map(modules['net2'], 'out')
    pd_out = find_layer_feature_map(modules['grafting'], 'out')
    targets['cp2net']['graft_er'][params['grafting']['nlayers']] = \
        reduce_value(tf.square(gt_out - pd_out))
    

    return ph, graph, save_vars, net_vars, targets



def build_neural_network_hybrid_model(params):
    # build placeholder

    inp_x = tf.placeholder(dtype=tf.float32, 
                           shape=[None] + params['data']['x_size'],
                           name='x')
    label = tf.placeholder(dtype=tf.int64,
                           shape=[None],
                           name='label')
    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')
    lr_decay = tf.placeholder(dtype=tf.float32, shape=[], name='lr_decay')
    inter_w = tf.placeholder(dtype=tf.float32, shape=[], name='interpolation_weight')
    ph = {
        'x': inp_x,
        'y': label,
        'is_training': is_training,
        'lr_decay': lr_decay,
        'inter_weight': inter_w
    }
  # build computation graph
    nclass = params['data']['nclass']
    model_name = params['network']['model']    # vgg16, vgg16-large, vgg19 
    use_bn = params['network']['batch_norm']
    num_nets = params['hybrid']['num_nets']
    domains = ['hybrid'] +  ['net' + str(k) for k in range(num_nets)]

    modules = {}
    graph = {
        'one_hot_y': tf.one_hot(label, nclass)
    }
    for domain in domains:
        scaling = 1
        if domain == 'hybrid':
            scaling = params['hybrid']['scaling']
        modules[domain] = \
            make_layers_vgg_net(scope=domain,
                                input_x=inp_x, 
                                config=config[model_name],
                                num_classes=nclass,
                                dropout_rate=params['network']['dropout'],
                                is_training=is_training,
                                batch_norm=use_bn,
                                layer_mask=None,
                                preact=True,
                                scaling=scaling)
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

    nets_domain = ['net' + str(k) for k in range(num_nets)]
    # num_nets default-training domain
    for domain in nets_domain:
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
    
        show_grads(domain, sl_grads)
    
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
            'eval': test
        }

    targets['hybrid'] = {}
    reg_loss = 0
    fm_loss = 0
    train_weights = []
    # layer-wise feature matching
    for lid in range(params['hybrid']['nlayers'] - 1):
        layer_name = 'l{}-'.format(lid + 1)
        print('[HYBRID] Layerwise Feature Matching Building at Layer {}'.format(lid + 1))
        weight_lid = [weight for weight in net_vars['hybrid'] if layer_name in weight.name]

        hybrid_feature = find_layer_feature_map(modules['hybrid'], layer_name)
        nchannels = int(hybrid_feature.get_shape()[-1])
        cnl_alloc = allocate_channels(nchannels, params['hybrid']['pweight'])
        print('[HYBRID] -- number of channles = {}, allocation = {}'.format(nchannels, cnl_alloc))

        nets_feature = []
        for k in range(num_nets):
            netk_feature = find_layer_feature_map(modules['net' + str(k)], layer_name)
            if cnl_alloc[k] > 0:
                netk_feature = fixed_random_select(netk_feature, cnl_alloc[k])
                print('[HYBRID] -- part {}, feature shape = {}'.format(k, netk_feature.get_shape()))
                nets_feature.append(netk_feature)
            else:
                print('[HYBRID] -- part {}, empty'.format(k))

        all_feature = tf.concat(nets_feature, -1)
        
        def relu_feat_match(x, y):
            #ind = tf.cast(tf.greater(0.0, x), tf.float32) * tf.cast(tf.greater(0.0, y), tf.float32)
            #dif = (1 - ind) * tf.square(x - y)
            dif = tf.square(tf.nn.relu(x) - tf.nn.relu(y))
            return tf.reduce_mean(dif)
            
        fm_loss_l = tf.reduce_mean(relu_feat_match(all_feature, hybrid_feature))
        reg_loss_l = get_regularizer_loss(weight_lid, params['network']['regularizer'])

        if params['hybrid']['cum']:
            reg_loss += reg_loss_l
            fm_loss += fm_loss_l
            train_weights += weight_lid
        else:
            reg_loss = reg_loss_l
            fm_loss = fm_loss_l
            train_weights = weight_lid            

        loss = fm_loss * params['hybrid']['fmw'] + reg_loss * params['network']['regw']

        fml_op = tf.train.AdamOptimizer(1e-3)
        fml_grads = fml_op.compute_gradients(loss=loss, var_list=train_weights)
        fml_train_op = fml_op.apply_gradients(grads_and_vars=fml_grads)
    
        show_grads('hybrid layer {}'.format(lid + 1), fml_grads)

        train = {
            'train': fml_train_op,
            'overall_loss': loss,
            'fml_loss': fm_loss,
            'regl_loss': reg_loss,
        }

        test = {
            'overall_loss': loss,
            'fml_loss': fm_loss,
            'regl_loss': reg_loss,
        }

        targets['hybrid']['layer{}'.format(lid + 1)] = {
            'train': train,
            'eval': test
        }

    for domain in domains:
        logits = modules[domain]['out']
        ce_loss = tf.nn.softmax_cross_entropy_with_logits(labels=graph['one_hot_y'], 
            logits=logits, dim=1)
        ce_loss = tf.reduce_mean(ce_loss)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), ph['y']), tf.float32))   # [1,]

        regularizer = get_regularizer_loss(net_vars[domain], params['network']['regularizer'])

        loss = ce_loss + regularizer * params['network']['regw']
        lst_weights = [weight for weight in net_vars[domain] if 'output' in weight.name]

        lst_op = tf.train.AdamOptimizer(1e-3)
        lst_grads = lst_op.compute_gradients(loss=loss, var_list=lst_weights)
        lst_train_op = lst_op.apply_gradients(grads_and_vars=lst_grads)
    
        show_grads(domain + ' lst', lst_grads)
    
        train = {
            'train': lst_train_op,
            'overall_loss': loss,
            'ce_loss': ce_loss,
            'reg_loss': regularizer,
            'acc_loss': acc
        }

        test = {
            'overall_loss': loss,
            'ce_loss': ce_loss,
            'reg_loss': regularizer,
            'acc_loss': acc
        }

        targets[domain]['lst'] = {
            'train': train,
            'eval': test
        }

    var_dict = create_var_dict(net_vars['hybrid'])
    var_dict_net1 = create_var_dict(net_vars['net0'])
    var_dict_net2 = create_var_dict(net_vars['net1'])
    targets['hybrid']['linear_inter'] = []

    for layer_id in range(params['hybrid']['nlayers']):
        # fetch all weights in layer l
        weights, assigns = [], []
        for weight in net_vars['hybrid']:
            if ('l{}-'.format(layer_id + 1)) in weight.name:
                weights.append(weight)
        for weight in weights:
            weight_name = weight.name[len('hybrid')+1:]
            print('drafting variable {}'.format(var_dict['hybrid/' + weight_name].name))
            net1_weight = var_dict_net1['net0/' + weight_name]
            net2_weight = var_dict_net2['net1/' + weight_name]
            value = net1_weight * ph['inter_weight'] + (1 - ph['inter_weight']) * net2_weight
            assigns.append(tf.assign(weight, value))
        targets['hybrid']['linear_inter'] += assigns
    
    return ph, graph, save_vars, net_vars, targets


