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
parser.add_argument('--gpu', default=-1, type=int)
parser.add_argument('--modeldir', default='../../data/cifar-100-models/', type=str)
parser.add_argument('--model2dir', default='../../data/cifar-100-models/', type=str)
parser.add_argument('--nanase', default=5, type=int)
parser.add_argument('--pweight', default=0.5, type=float)
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

# set seed
params['train']['seed'] = args.seed
np.random.seed(args.seed)
tf.set_random_seed(args.seed)

params['hybrid']['pweight'] = [args.pweight, 1 - args.pweight]

# target log dir
log_dir = os.path.join(args.logdir, datetime.datetime.now().strftime('[%m_%d_%H_%M]') + args.exp_id + '_hybrid_' + str(args.pweight))
print('Experiment Logs will be written at {}'.format(log_dir))
logger = LogWriter(log_dir, 'main.log')

# model save log dir
model_dir = args.modeldir

# load dataset
dataset = load_dataset(params)
train_loader, test_loader = dataset['train'], dataset['test']

# build model
ph, graph, save_vars, graph_vars, targets = build_neural_network_hybrid_model(params)
saver = tf.train.Saver(var_list=save_vars)
iter_per_epoch = params['train']['iter_per_epoch']
time.sleep(5)

dnets = ['net' + str(k) for k in range(params['hybrid']['num_nets'])] + ['hybrid']


def train_layerwise(lid, ph, graph, targets, epoch, data_loader, debug=False):
    train_log = {}
    layern = 'layer{}'.format(lid)
    for batch_idx in range(params['train']['iter_per_epoch']):
        x, y = data_loader.next_batch(params['hybrid']['batch_size'])
        fetch = sess.run(targets['hybrid'][layern]['train'],
            feed_dict={
                ph['x']: x,
                ph['y']: y,
                ph['is_training']: True
            }
        )
        update_loss(fetch, train_log)
    print('=' * 32)
    print_log('hybrid train', epoch, train_log)
    logger.print(epoch, 'hybrid layer {} train'.format(lid), train_log)

    
def eval_layerwise(lid, ph, graph, targets, epoch, dsdomain, data_loader):
    eval_log = {}
    layern = 'layer{}'.format(lid)
    for batch_idx in range(params[dsdomain]['iter_per_epoch']):
        x, y = data_loader.next_batch(params[dsdomain]['batch_size'])

        fetch = sess.run(targets['hybrid'][layern]['eval'],
            feed_dict={
                ph['x']: x,
                ph['y']: y,
                ph['is_training']: False
            }
        )
        update_loss(fetch, eval_log)
    print('=' * 32)
    print_log('Layer {} {} {}'.format(lid, 'hybrid', dsdomain), epoch, eval_log)
    logger.print(epoch, 'Layer {} {} {}'.format(lid, 'hybrid', dsdomain), eval_log)



def train_lst(ph, graph, targets, epoch, data_loader, debug=False):
    train_log = {}
    for d in dnets:
        train_log[d] = {}

    for batch_idx in range(params['train']['iter_per_epoch']):
        x, y = data_loader.next_batch(params['train']['batch_size'])
        for domain in dnets:
            fetch = sess.run(targets[domain]['lst']['train'],
                feed_dict={
                    ph['x']: x,
                    ph['y']: y,
                    ph['is_training']: True
                }
            )
            update_loss(fetch, train_log[domain])
    print('=' * 32)
    for domain in dnets:
        print_log('Domain Lst {} train'.format(domain), epoch, train_log[domain])
        logger.print(epoch, 'Lst {} train'.format(domain), train_log[domain])

    
def eval_lst(ph, graph, targets, epoch, dsdomain, data_loader):
    #base_lr = train_scheduler.step()

    eval_log = {}
    for d in dnets:
        eval_log[d] = {}
    for batch_idx in range(params[dsdomain]['iter_per_epoch']):
        x, y = data_loader.next_batch(params[dsdomain]['batch_size'])

        for domain in dnets:
            fetch = sess.run(targets[domain]['lst']['eval'],
                feed_dict={
                    ph['x']: x,
                    ph['y']: y,
                    ph['is_training']: False
                }
            )
            update_loss(fetch, eval_log[domain])
    print('=' * 32)
    for domain in dnets:
        print_log('Domain Lst {} {}'.format(domain, dsdomain), epoch, eval_log[domain])
        logger.print(epoch, 'Lst {} {}'.format(domain, dsdomain), eval_log[domain])



gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
sess.run(tf.global_variables_initializer())


saver.restore(sess, os.path.join(model_dir, 'vgg2.ckpt'))

if len(args.model2dir) > 5:
    saver2 = tf.train.Saver(var_list=graph_vars['net1'])
    saver2.restore(sess, os.path.join(args.model2dir, 'vgg2.ckpt'))

for epoch in range(params['hybrid']['num_epoches']):
    train_lst(ph, graph, targets, epoch, train_loader)
    eval_lst(ph, graph, targets, epoch, 'test', test_loader)

for lid in range(params['hybrid']['nlayers'] - 1):
    for epoch in range(params['hybrid']['num_epoches']):
        train_layerwise(lid + 1, ph, graph, targets, epoch, train_loader)
        eval_layerwise(lid + 1, ph, graph, targets, epoch, 'test', test_loader)

for epoch in range(params['hybrid']['num_epoches']):
    train_lst(ph, graph, targets, epoch, train_loader)
    eval_lst(ph, graph, targets, epoch, 'test', test_loader)
