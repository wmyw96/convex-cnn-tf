
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
        'lr': 0.001,
        'milestone': [60, 120, 180],
        'gamma': 0.2,
        'warmup': 1,
        'batch_size': 64,
        'num_epoches': 200,
        'iter_per_epoch': 780,
        'save_interval': [1, 2, 5, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 
                          100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
    }

    test = {
        'batch_size': 128,
        'iter_per_epoch': 78,
    }

    network = {
        'model': 'vgg16',
        'regw': 5e-4 * 8,
        'batch_norm': False,
        'dropout': 0.5,
        'regularizer': 'l2',
        'layer_mask': [True] * 13
    }

    graft = {
        'use_adam': True,
        'adam_lr': 1e-3,
        'lr': 0.01,
        'milestone': [60, 120, 180],
        'gamma': 0.2,
        'warmup': 1,
        'batch_size': 64,
        'num_epoches': 200,
        'iter_per_epoch': 780,
        'nlayers': 16,
        'nanase': 5,
        'diffw': 5.0,
        'layer_mask': [True] * 16
    }

    params = {
        'data': data,
        'train': train,
        'test': test, 
        'network': network,
        'grafting': graft
    }

    return params
