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
parser.add_argument('--modeldir', default='../../data/cifar-100-models/')
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

# target log dir
log_dir = os.path.join(args.logdir, datetime.datetime.now().strftime('[%m_%d_%H_%M]') + args.exp_id)
print('Experiment Logs will be written at {}'.format(log_dir))
logger = LogWriter(log_dir, 'main.log')

# load dataset
dataset = load_dataset(params)
train_loader, test_loader = dataset['train'], dataset['test']

# build model
ph, graph, save_vars, graph_vars, targets = build_grafting_onecut_model(params)
saver = tf.train.Saver(var_list=save_vars, max_to_keep=30)
iter_per_epoch = params['train']['iter_per_epoch']
train_scheduler = MultiStepLR(params['train']['milestone'], params['train']['gamma'])
warmup_scheduler = WarmupLR(iter_per_epoch * params['train']['warmup'])

#saver.restore(sess, os.path.join(args.modeldir, 'vgg2.ckpt'))
   
def eval(ph, graph, targets, epoch, dsdomain, data_loader):
    #base_lr = train_scheduler.step()
    eval_log = {'net1': {}, 'net2': {}}
    for batch_idx in range(params[dsdomain]['iter_per_epoch']):
        x, y = data_loader.next_batch(params[dsdomain]['batch_size'])

        for domain in ['net1', 'net2']:
            '''sess.run(targets[domain]['train']['update'], 
                feed_dict={
                    ph['x']: x,
                    ph['y']: y,
                    ph['is_training']: True
                }
            )'''
            fetch = sess.run(targets[domain]['eval'], #+ targets[domain]['train']['update'],
                feed_dict={
                    ph['x']: x,
                    ph['y']: y,
                    ph['is_training']: False
                }
            )
            update_loss(fetch, eval_log[domain])
    for domain in ['net1', 'net2']:
        print_log('Domain {} {}'.format(domain, dsdomain), epoch, eval_log[domain])
        logger.print(epoch, '{} {}'.format(domain, dsdomain), eval_log[domain])


gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
sess.run(tf.global_variables_initializer())
saver.restore(sess, os.path.join(args.modeldir, 'vgg2.ckpt'))
max_valid_acc = 0

for epoch in range(30):
    eval(ph, graph, targets, epoch, 'test', test_loader)

