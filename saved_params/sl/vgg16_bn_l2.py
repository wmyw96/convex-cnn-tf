
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
        'milestone': [50, 75, 100],
        'gamma': 0.1,
        'warmup': 1,
        'batch_size': 64,
        'num_epoches': 150,
        'iter_per_epoch': 780,
        'save_interval': 10,
    }

    test = {
        'batch_size': 128,
        'iter_per_epoch': 78,
    }

    network = {
        'model': 'vgg16',
        'regw': 5e-4,
        'batch_norm': True,
        'regularizer': 'l2',
        'dropout': 0.8,
    }

    params = {
        'data': data,
        'train': train,
        'test': test, 
        'network': network
    }

    return params
