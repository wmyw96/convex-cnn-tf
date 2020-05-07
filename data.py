import numpy as np
import tensorflow as tf
import random


# data augmentation, contains padding, cropping and possible flipping
def augment_image(image, pad):
    init_shape = image.shape
    new_shape = [init_shape[0] + pad * 2, init_shape[1] + pad * 2, init_shape[2]]
    zeros_padded = np.zeros(new_shape)
    zeros_padded[pad:init_shape[0] + pad, pad: init_shape[1] + pad, :] = image
    init_x = np.random.randint(0, pad * 2)
    init_y = np.random.randint(0, pad * 2)
    cropped = zeros_padded[init_x: init_x + init_shape[0], init_y: init_y + init_shape[1], :]
    flip = random.getrandbits(1)
    if flip:
        cropped = cropped[:, ::-1, :]
    return cropped


def augment_all_images(initial_images, pad):
    new_images = np.zeros(initial_images.shape)
    for i in range(initial_images.shape[0]):
        new_images[i] = augment_image(initial_images[i], pad = 4)
    return new_images


class dataset(object):
    def __init__(self, feature, label, nclass=100, aug=False, randomize=True):
        self.inputs = feature
        self.labels = label
        self.nclass = nclass
        if len(self.labels.shape) == 1:
            self.labels = np.reshape(self.labels,
                                     [self.labels.shape[0], 1])
        assert len(self.inputs) == len(self.labels)
        print('Dataset')
        print('- Image Scale = {}'.format(np.max(self.inputs)))
        print('- Image Shape = {}'.format(self.inputs.shape))
        print('- Num Classes = {}'.format(np.max(self.labels) + 1))

        self.num_pairs = self.labels.shape[0]
        self.pointer = self.num_pairs
        self.randomize = randomize
        self.aug = aug

    def len(self):
        return self.num_pairs

    def init_pointer(self):
        self.pointer = 0

        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            if self.aug:
                self.inputs_aug = augment_all_images(self.inputs[idx, :], pad=4)
            else:
                self.inputs_aug = self.inputs[idx, :]
            self.labels_aug = self.labels[idx, :]

    def next_batch(self, batch_size):
        if batch_size < 0:
            return self.inputs, self.labels
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs_aug[self.pointer:end, :]
        labels = self.labels_aug[self.pointer:end, :]
        self.pointer = end
        return inputs, labels.squeeze()
            

def pre_cifar_100(x):
    x = x / 255.0
    xmean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    xstd = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    x_mean = np.reshape(np.array(xmean), (1, 1, 1, 3))
    x_std = np.reshape(np.array(xstd), (1, 1, 1, 3))
    return (x - x_mean) / x_std


def pre_cifar_10(x):
    x = x / 255.0
    xmean = np.mean(x, (0, 1, 2))
    xstd = np.std(x, (0, 1, 2))
    x_mean = np.reshape(np.array(xmean), (1, 1, 1, 3))
    x_std = np.reshape(np.array(xstd), (1, 1, 1, 3))
    return (x - x_mean) / x_std


def load_cifar_100(params):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
    x_train = pre_cifar_100(x_train)
    x_test = pre_cifar_100(x_test)
    if 'noda' in params['data']:
        train = dataset(x_train, y_train, 100, aug=False)
    else:
        train = dataset(x_train, y_train, 100, aug=True)
    test = dataset(x_test, y_test, 100, aug=False)
    return {
        'train': train,
        'test': test
    }


def load_cifar_10(params):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = pre_cifar_10(x_train)
    x_test = pre_cifar_10(x_test)

    if 'p' in params['data']:
        p = params['data']['p']
        concat_images = np.concatenate([x_train, x_test], 0)
        concat_labels = np.concatenate([y_train, y_test], 0)

        train_set_idx = []
        test_set_idx = []
        for i in range(concat_images.shape[0]):
            pv = np.random.uniform(0, 1.0)
            if pv < p[int(concat_labels[i])]:
                train_set_idx.append(i)
            else:
                test_set_idx.append(i)

        x_train, y_train = concat_images[train_set_idx, :], concat_labels[train_set_idx]
        x_test, y_test = concat_images[test_set_idx, :], concat_labels[test_set_idx]

    if 'noda' in params['data']:
        train = dataset(x_train, y_train, 100, aug=False)
    else:
        train = dataset(x_train, y_train, 100, aug=True)
    test = dataset(x_test, y_test, 10, aug=False)
    return {
        'train': train,
        'test': test
    }


def load_dataset(params):
    if params['data']['dataset'] == 'cifar-100':
        return load_cifar_100(params)
    elif params['data']['dataset'] == 'cifar-10':
        return load_cifar_10(params)
    else:
        raise NotImplemented

