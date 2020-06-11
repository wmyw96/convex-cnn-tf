
def generate_params():
    nclass = 10

    data = {
        'rot': False,
        'dataset': 'cifar-10',
        'data_dir': '../../data/cifar-10/',
        'x_size': [32, 32, 3],
        'nclass': nclass,
    }

    train = {
        'lr': 0.01,
        'milestone': [60, 120, 180],
        'gamma': 0.2,
        'warmup': 1,
        'batch_size': 64,
        'num_epoches': 200,
        'iter_per_epoch': 780,
        'save_interval': [1, 2, 3, 4, 5, 8, 10, 20, 30, 40, 50, 60, 80, 100, 120, 180],
    }

    test = {
        'batch_size': 128,
        'iter_per_epoch': 78,
    }

    network = {
        'model': 'vgg16',
        'regw': 1,
        'batch_norm': False,
        'dropout': 0.5,
        'regularizer': 'l12',
        'layer_mask': [True] * 13
    }

    hybrid = {
        'weight_inter': False,
        'batch_size': 64,
        'num_epoches': 7,
        'iter_per_epoch': 780,
        'nlayers': 16,
        'num_nets': 2,
        'fmw': 300.0,
        'layer_mask': [True] * 16,
        'cum': False,
        'pweight': [0.5, 0.5],
        'scaling': 2.0,
        'init': {'net0': 'uniform', 'net1': 'normal', 'hybrid': 'normal'}
    }

    params = {
        'data': data,
        'train': train,
        'test': test, 
        'network': network,
        'hybrid': hybrid
    }

    return params
