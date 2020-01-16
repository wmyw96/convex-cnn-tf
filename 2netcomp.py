import numpy as np
import datetime
import tensorflow as tf
import json, sys, os
from os import path
import time
import shutil
import matplotlib
import importlib
import argparse
from tqdm import tqdm
from utils import *
from data import *
from comp_graph import *

# os settings
sys.path.append(os.getcwd() + '/..')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parse cmdline args
parser = argparse.ArgumentParser(description='Image Classification With Two Networks')
parser.add_argument('--logdir', default='../../data/cifar-100-logs/', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--exp_id', default='sl.vgg16_nobn_l2', type=str)
parser.add_argument('--figlogdir', default='.logs/vgg16-l12', type=str)
parser.add_argument('--gpu', default=-1, type=int)
parser.add_argument('--modeldir', default='../../data/cifar-100-models/', type=str)
parser.add_argument('--model1dir', default='', type=str)
parser.add_argument('--nanase', default=5, type=int)
args = parser.parse_args()

# GPU settings
if args.gpu > -1:
    print("GPU COMPATIBLE RUN...")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

# Print experiment details
print('Booting with exp params: {}...'.format(args.exp_id))
time.sleep(2)

# Experiment parameters
mod = importlib.import_module('saved_params.' + args.exp_id)
params = mod.generate_params()

# set grafting layer
params['grafting']['nanase'] = args.nanase

# set seed
params['train']['seed'] = args.seed
np.random.seed(args.seed)
tf.set_random_seed(args.seed)

# target log dir
log_dir = os.path.join(args.logdir, datetime.datetime.now().strftime('[%m_%d_%H_%M]') + args.exp_id)
print('Experiment Logs will be written at {}'.format(log_dir))
logger = LogWriter(log_dir, 'main.log')

# model save log dir
model_dir = args.modeldir

# load dataset
dataset = load_dataset(params)
train_loader, test_loader = dataset['train'], dataset['test']

# build model
ph, graph, save_vars, graph_vars, targets = build_grafting_onecut_model(params)
saver = tf.train.Saver(var_list=graph_vars['net1'] + graph_vars['net2'])
iter_per_epoch = params['train']['iter_per_epoch']
time.sleep(5)

    
def eval(ph, graph, targets, epoch, dsdomain, data_loader):
    #base_lr = train_scheduler.step()
    eval_log = {}
    for batch_idx in range(params[dsdomain]['iter_per_epoch']):
        x, y = data_loader.next_batch(params[dsdomain]['batch_size'])

        fetch = sess.run(targets['grafting']['eval'],
            feed_dict={
                ph['x']: x,
                ph['y']: y,
                ph['is_training']: False
            }
        )
        update_loss(fetch, eval_log)

    print_log('{} {}'.format('grafting', dsdomain), epoch, eval_log)
    logger.print(epoch, '{} {}'.format('grafting', dsdomain), eval_log)


gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
sess.run(tf.global_variables_initializer())

saver.restore(sess, os.path.join(model_dir, 'vgg2.ckpt'))

if len(args.model1dir) > 5:
    saver1 = tf.train.Saver(var_list=graph_vars['net1'])
    saver1.restore(sess, os.path.join(args.model1dir, 'vgg2.ckpt'))


def semi_matching(dist_mat):
    value = np.min(dist_mat, 1)
    ind = np.argmin(dist_mat, 1)
    return value, ind


def eval_layer(ph, graph, targets, data_loader, dsdomain, layerid):
    slid = 'l' + str(layerid + 1)
    values = []
    for batch_idx in range(params[dsdomain]['iter_per_epoch']):
        x, y = data_loader.next_batch(params[dsdomain]['batch_size'])
        fetch = sess.run([targets['cp2net'][slid]['l2_dist'],
                          targets['cp2net'][slid]['l2_ndist'],
                          targets['cp2net'][slid]['net1']['gradnorm'],
                          targets['cp2net'][slid]['net1']['l2_norm'],
                          targets['cp2net'][slid]['net2']['gradnorm'],
                          targets['cp2net'][slid]['net2']['l2_norm'],
                          targets['cp2net'][slid]['net1']['std'],
                          targets['cp2net'][slid]['net2']['std'],
                          targets['net1']['eval']['acc_loss'],
                          targets['net2']['eval']['acc_loss']
                         ],
                          feed_dict={
                                ph['x']: x, ph['y']: y,
                                ph['is_training']: False
                          })
        if len(values) == 0:
            for i in range(len(fetch)):
                values.append([fetch[i]])
        else:
            for i in range(len(fetch)):
                values[i].append(fetch[i])
    
    valuec = []
    for i in range(len(values)):
        valuec.append(np.array(values[i]))
        print(valuec[i].shape)
        valuec[i] = np.mean(valuec[i], 0)

    nchannels = valuec[0].shape[0]

    l2_dist = valuec[0]
    l2_ndist = valuec[1]
    net1_gn = valuec[2]
    net2_gn = valuec[4]
    net1_l2n = valuec[3]
    net2_l2n = valuec[5]
    net1_std = valuec[6]
    net2_std = valuec[7]

    print('Domain {} Layer {}: Net1 acc = {}, Net2 acc = {}'.format(
        dsdomain, layerid, valuec[8], valuec[9]))
    
    v, ind = semi_matching(l2_ndist)
    vind = np.argsort(v)

    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Times New Roman"
    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Set1')

    # nearest channel in net2 (sorted)
    plt.figure(figsize=(16, 20))
    ax = plt.subplot(3, 4, 1)
    plt.plot(np.arange(nchannels) + 1, v[vind])
    plt.xlabel('channel index in net1')
    plt.ylabel('matched l2 distance (after normalization)')
    plt.ylim(0, 2.0)

    ax = plt.subplot(3, 4, 2)
    plt.plot(np.arange(nchannels) + 1, net1_std[vind], label='net1 std', color=palette(1))
    plt.plot(np.arange(nchannels) + 1, net2_std[ind[vind]], label='net2 std', color=palette(2))
    plt.xlabel('channel index in net1')
    plt.ylabel('std')
    plt.legend(loc='lower right', frameon=True)
    
    ax = plt.subplot(3, 4, 5)
    plt.plot(np.arange(nchannels) + 1, net1_gn[vind])
    plt.xlabel('channel index in net1')
    plt.ylabel('net1 gradient')

    ax = plt.subplot(3, 4, 6)
    plt.plot(np.arange(nchannels) + 1, net1_gn[vind] * net1_std[vind])
    plt.xlabel('channel index in net1')
    plt.ylabel('net1 gradient * std')

    ax = plt.subplot(3, 4, 7)
    plt.plot(np.arange(nchannels) + 1, net1_l2n[vind])
    plt.xlabel('channel index in net1')
    plt.ylabel('net1 l2 norm')

    ax = plt.subplot(3, 4, 8)
    plt.plot(np.arange(nchannels) + 1, net1_l2n[vind] * net1_std[vind])
    plt.xlabel('channel index in net1')
    plt.ylabel('net1 l2 norm * std')

    ax = plt.subplot(3, 4, 9)
    plt.plot(np.arange(nchannels) + 1, net2_gn[ind[vind]])
    plt.xlabel('channel index in net1')
    plt.ylabel('matched net2 gradient')

    ax = plt.subplot(3, 4, 10)
    plt.plot(np.arange(nchannels) + 1, net2_gn[ind[vind]] * net2_std[ind[vind]])
    plt.xlabel('channel index in net1')
    plt.ylabel('matched net2 gradient * std')

    ax = plt.subplot(3, 4, 11)
    plt.plot(np.arange(nchannels) + 1, net2_l2n[ind[vind]])
    plt.xlabel('channel index in net1')
    plt.ylabel('matched net2 l2 norm')

    ax = plt.subplot(3, 4, 12)
    plt.plot(np.arange(nchannels) + 1, net2_l2n[ind[vind]] * net2_std[ind[vind]])
    plt.xlabel('channel index in net1')
    plt.ylabel('matched net2 l2 norm * std')

    filename = os.path.join(args.figlogdir, '{}layer{}.pdf'.format(dsdomain, layerid+1))
    plt.savefig(filename)
    plt.close()
    plt.clf()

# compare the following terms

# for each layer:
#   for each channel in net1, find the nearest channel in net2 (sorted)
#   for each channel in net1, calculate the 
#      calculate  |{\partial (f)}| / |{\partial (z^l)}|,
#      calculate  the (l1 norm) and (l2 norm) of connected weights


for i in range(params['grafting']['nlayers'] - 1):
    eval_layer(ph, graph, targets, train_loader, 'train', i)
    eval_layer(ph, graph, targets, test_loader, 'test', i)










