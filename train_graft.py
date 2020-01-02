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
train_scheduler = MultiStepLR(params['grafting']['milestone'], params['grafting']['gamma'])
warmup_scheduler = WarmupLR(iter_per_epoch * params['grafting']['warmup'])
time.sleep(5)

def train(ph, graph, targets, epoch, data_loader, train_scheduler, 
    warmup_scheduler, debug=False):
    base_lr = train_scheduler.step()
    train_log = {}
    print('Epoch {}: lr decay = {}'.format(epoch, base_lr))
    for batch_idx in range(params['train']['iter_per_epoch']):
        if epoch < params['grafting']['warmup']:
            lr = base_lr * warmup_scheduler.step()
        else:
            lr = base_lr
        
        if debug:
            print('Epoch {} Batch {}: Learning Decay = {}'.format(epoch, batch_idx, lr))

        x, y = data_loader.next_batch(params['grafting']['batch_size'])
        fetch = sess.run(targets['grafting']['train'],
            feed_dict={
                ph['x']: x,
                ph['y']: y,
                ph['lr_decay']: lr,
                ph['is_training']: True
            }
        )
        update_loss(fetch, train_log)
    print_log('grafting train', epoch, train_log)
    logger.print(epoch, 'grafting train', train_log)

    
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

def assign_weights(assign_handle, layer_l, layer_r):
    for i in range(layer_r - layer_l + 1):
        layer_id = layer_l + i
        sess.run(assign_handle['l{}'.format(layer_id)])

if len(args.model1dir) > 5:
    saver1 = tf.train.Saver(var_list=graph_vars['net1'])
    saver1.restore(sess, os.path.join(args.model1dir, 'vgg2.ckpt'))
    assign_weights(targets['grafting']['assign_net1'], 1, params['grafting']['nlayers'])

saver.restore(sess, os.path.join(model_dir, 'vgg2.ckpt'))
eval(ph, graph, targets, -1, 'train', train_loader)
eval(ph, graph, targets, -1, 'test', test_loader)

#assign_weights(targets['grafting']['assign_net1'], 1, params['grafting']['nlayers'])
#eval(ph, graph, targets, -1, 'train', train_loader)
#eval(ph, graph, targets, -1, 'test', test_loader)


#assign_weights(targets['grafting']['assign_net1'], 1, args.nanase - 1)
assign_weights(targets['grafting']['assign_net2'], args.nanase, params['grafting']['nlayers'])
eval(ph, graph, targets, -1, 'train', train_loader)
eval(ph, graph, targets, -1, 'test', test_loader)


for epoch in range(params['train']['num_epoches']):
    train(ph, graph, targets, epoch, train_loader,
        train_scheduler, warmup_scheduler)
    eval(ph, graph, targets, epoch, 'test', test_loader)
