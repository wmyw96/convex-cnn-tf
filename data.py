import numpy as np
import tensorflow as tf

class dataset(object):
    def __init__(self, feature, label, nclass=100, randomize=True):
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

    def len(self):
        return self.num_pairs

    def init_pointer(self):
        self.pointer = 0

        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]
            self.labels = self.labels[idx, :]

    def next_batch(self, batch_size):
        if batch_size < 0:
            return self.inputs, self.labels
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        labels = self.labels[self.pointer:end, :]
        self.pointer = end
        return inputs, labels.squeeze()
            

def load_cifar_100(params):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
    train = dataset(x_train, y_train, 100)
    test = dataset(x_test, y_test, 100)
    return {
        'train': train,
        'test': test
    }


def load_dataset(params):
    if params['data']['dataset'] == 'cifar-100':
        return load_cifar_100(params)
    else:
        raise NotImplemented

