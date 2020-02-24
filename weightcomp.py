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
parser.add_argument('--bias', default=0, type=int)
parser.add_argument('--outpath', default='', type=str)
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

    
def eval(ph, sess, graph, targets, epoch, domain, dsdomain, data_loader):
    #base_lr = train_scheduler.step()
    eval_log = {}
    for batch_idx in range(params[dsdomain]['iter_per_epoch']):
        x, y = data_loader.next_batch(params[dsdomain]['batch_size'])

        fetch = sess.run(targets[domain]['eval'],
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


#params['train']['save_interval'] = [1, 180]
params['grafting']['nlayers'] = 13

dd1, dd2 = len(params['train']['save_interval']), params['grafting']['nlayers'] - 1
weight_l2 = np.zeros((dd1, dd2))


for e, eid in enumerate(params['train']['save_interval']):
    saver.restore(sess, os.path.join(os.path.join(model_dir, 'epoch'+str(eid+args.bias)), 'vgg2.ckpt'))
    eval(ph, sess, graph, targets, eid, 'net1', 'test', test_loader)
    eval(ph, sess, graph, targets, eid, 'net2', 'test', test_loader)

    for i in range(params['grafting']['nlayers'] - 1):
        net1_weight = find_weight(graph_vars['net1'], 'l{}-'.format(i+1))
        net2_weight = find_weight(graph_vars['net2'], 'l{}-'.format(i+1))
        net1_weight, net2_weight = sess.run([net1_weight, net2_weight])
        print(net1_weight.shape)
        diff = np.mean(np.square(net1_weight - net2_weight)) #* net1_weight.shape[2]
        base = np.mean(np.square(net1_weight)) + np.mean(np.square(net2_weight)) 
        weight_l2[e, i] = diff / (base * 0.5)

output = np.array([weight_l2])
np.save(args.outpath, output)

