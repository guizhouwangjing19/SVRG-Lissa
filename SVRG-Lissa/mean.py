import numpy as np
import pickle, gzip, math
from sklearn.preprocessing import normalize

class Fisher_opti:

    def __init__(self, dataset='MNIST', lam=0):
        self.lam = lam
        self.load_dataset(dataset)


    def load_dataset(self, dataset):
        if dataset == 'MNIST':
            print('-----------------------------------------------')
            print('Loading MNIST 4/9 data...')
            print('-----------------------------------------------\n')
            self.load_mnist_49()
        else:
            raise ValueError('No Dataset exists by that name')

        self.data_dim = self.train_set[0][0].size
        self.num_train_examples = self.train_set[0].shape[0]
        self.num_test_examples = self.test_set[0].shape[0]

    def load_mnist_49(self):
        f = open('../data/mnist49data', 'rb')
        train_set, valid_set, test_set = pickle.load(f)
        f.close()
        self.train_set = [normalize(train_set[0], axis=1, norm='l2'), train_set[1]]
        self.valid_set = [normalize(valid_set[0], axis=1, norm='l2'), valid_set[1]]
        self.test_set = [normalize(test_set[0], axis=1, norm='l2'), test_set[1]]

    def fetch_correct_datamode(self, mode='TRAIN'):
        if mode == 'TRAIN':
            return self.train_set
        elif mode == 'VALIDATE':
            return self.validate_set
        elif mode == 'TEST':
            return self.test_set
        else:
            raise ValueError('Wrong mode value provided')

    def predict(self, data_index, w1, b1, w2, b2, mode = 'TRAIN'):
        data_set = self.fetch_correct_datamode(mode)
        yi = np.dot(w1, data_set[0][data_index]) + b1
        zi = 1/(1 + np.exp(yi))
        mi = np.dot(w2, zi) + b2
        print(yi)
        print(zi)
        print(mi)
        return 1 / (1 + np.exp(mi))