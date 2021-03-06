
def generate_params():
    nclass = 100

    data = {
        'rot': False,
        'dataset': 'cifar-100',
        'data_dir': '../../data/cifar-100/',
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
        'save_interval': [5, 10, 20, 30, 40, 50, 80, 100, 120, 180],
    }

    test = {
        'batch_size': 128,
        'iter_per_epoch': 78,
    }

    network = {
        'model': 'vgg16',
        'regw': 2,
        'batch_norm': False,
        'dropout': 0.5,
        'regularizer': 'l12',
        'layer_mask': [True] * 13
    }

    hybrid = {
        'fm_lr': 0.01,
        'milestone': [30, 60, 90],
        'gamma': 0.2,
        'warmup': 1,
        'batch_size': 8,
        'num_epoches': 200,
        'iter_per_epoch': 780 * 4 * 2,
        'nlayers': 16,
        'num_nets': 2,
        'fmw': 10.0,
        'layer_mask': [True] * 16,
        'cum': False,
        'matchk': 100,
        'sigmas': [0.1, 0.5, 1, 2, 4, 8, 16]
    }

    params = {
        'data': data,
        'train': train,
        'test': test, 
        'network': network,
        'hybrid': hybrid
    }

    return params
