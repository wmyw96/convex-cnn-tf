
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
        'lr': 0.1,
        'milestone': [60, 120, 180],
        'gamma': 0.2,
        'warmup': 1,
        'batch_size': 128,
        'num_epoches': 200,
        'iter_per_epoch': 390,
        'save_interval': 10,
    }

    test = {
        'batch_size': 128,
        'iter_per_epoch': 78,
    }

    network = {
        'model': 'vgg16',
        'regw': 5e-4,
        'batch_norm': False,
        'regularizer': 'l2'
    }

    params = {
        'data': data,
        'train': train,
        'test': test, 
        'network': network
    }

    return params
