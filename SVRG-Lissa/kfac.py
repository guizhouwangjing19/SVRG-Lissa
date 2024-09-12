import numpy as np
from scipy import linalg
import pickle, gzip, math
from sklearn.preprocessing import normalize


h1 = 5
h2 = 2

class KFAC:
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

    # print(self.num_train_examples)

    ### The following functions implement the logistic function oracles

    ## This is to load the 49 mnist
    def load_mnist_49(self):
        f = open('../data/mnist49data', 'rb')
        train_set, valid_set, test_set = pickle.load(f)
        f.close()
        self.train_set = [normalize(train_set[0], axis=1, norm='l2'), train_set[1]]
        self.valid_set = [normalize(valid_set[0], axis=1, norm='l2'), valid_set[1]]
        self.test_set = [normalize(test_set[0], axis=1, norm='l2'), test_set[1]]


    # print(self.train_set)

    ## First implement the function for individuals

    def fetch_correct_datamode(self, mode='TRAIN'):
        if mode == 'TRAIN':
            return self.train_set
        elif mode == 'VALIDATE':
            return self.validate_set
        elif mode == 'TEST':
            return self.test_set
        else:
            raise ValueError('Wrong mode value provided')
    #定义relu函数
    def relu(self, x):
        l = len(x)
        y = np.random.rand(l, 1)
        for i in range(l):
            if x[i][0]>0:
                y[i][0] = x[i][0]
            else:
                y[i][0] = 0
        return y

    def relu_grand(self, x):
        l = len(x)
        y = np.random.rand(l, 1)
        for i in range(l):
            if x[i][0] > 0:
                y[i][0] = 1
            else:
                y[i][0] = 0
        return y
    #定义神经网络模型，该模型有两个隐藏层，每个隐藏层各有h1,h2个隐藏单元，一个输出节点
    def net(self, data_index, w1, w2, w3, mode='TEST'):

        data_set = self.fetch_correct_datamode(mode)
        a0 = data_set[0][data_index]
        a0 = a0.reshape(-1, 1)
        s1 = np.dot(w1, a0)
        a1 = self.relu(s1)
        s2 = np.dot(w2, a1)
        a2 = self.relu(s2)
        s3 = np.dot(w3, a2)
        return a0, s1, a1, s2, a2, s3
    def net_loss(self, data_batch, w1, w2, w3, mode='TRAIN'):
        data_set = self.fetch_correct_datamode(mode)
        loss = 0
        for data_index in data_batch:
            a0, s1, a1, s2, a2, s3 = self.net(data_index, w1, w2, w3, mode='TRAIN')
            y = data_set[1][data_index]
            loss = loss + (s3 - y)**2
        return loss/len(data_batch)

    def test_error(self,  w1, w2, w3):
        data_set = self.test_set
        i = 0
        for data_index in range(self.num_test_examples):
            a0 = data_set[0][data_index]
            a0 = a0.reshape(-1, 1)
            s1 = np.dot(w1, a0)
            a1 = self.relu(s1)
            s2 = np.dot(w2, a1)
            a2 = self.relu(s2)
            s3 = np.dot(w3, a2)
            if s3 * data_set[1][data_index]>0:
                i = i + 1
            else:
                i = i
        return i/self.num_test_examples

    #定义损失函数的梯度和计算A, G
    def loss_grand_A_G(self, data_index, w1, w2, w3, mode='TRAIN'):
        data_set = self.fetch_correct_datamode(mode)
        y = data_set[1][data_index]
        a0, s1, a1, s2, a2, s3 = self.net(data_index, w1, w2, w3, mode='TRAIN')
        a0 = a0.reshape(-1, 1)

        d_s3 = 2 * (s3 - y)
        d_w3 = d_s3[0] * a2.T
        d_a2 = d_s3 * w3.T
        d_s2 = d_a2 * self.relu_grand(s2)
        d_w2 = np.dot(d_s2, a1.T)
        d_a1 = np.dot(w2.T, d_s2)
        d_s1 = d_a1 * self.relu_grand(s1)
        d_w1 = np.dot(d_s1, a0.T)

        A0 = np.dot(a0, a0.T)
        A1 = np.dot(a1, a1.T)
        A2 = np.dot(a2, a2.T)
        G1 = np.dot(d_s1, d_s1.T)
        G2 = np.dot(d_s2, d_s2.T)
        G3 = np.dot(d_s3, d_s3.T)

        F1 = np.kron(A0, G1)
        F2 = np.kron(A1, G2)
        F3 = np.kron(A2, G3)

        return F1, F2, F3, d_w1, d_w2, d_w3
    def loss_grand_A_G_AVE(self, data_batch, w1, w2, w3):
        F1 = np.random.rand(784*h1, 784*h1)
        F2 = np.random.rand(h1*h2, h1*h2)
        F3 = np.random.rand(h2, h2)
        d_w1 = np.random.rand(h1, 784)
        d_w2 = np.random.rand(h2, h1)
        d_w3 = np.random.rand(1, h2)

        for data_index in data_batch:
            F1_, F2_, F3_, d_w1_, d_w2_, d_w3_ = self.loss_grand_A_G(data_index, w1, w2, w3, mode='TRAIN')

            F1 = F1 + F1_
            F2 = F2 + F2_
            F3 = F3 + F3_
            d_w1 = d_w1 + d_w1_
            d_w2 = d_w2 + d_w2_
            d_w3 = d_w3 + d_w3_

        F1 = F1 /len(data_batch)
        F2 = F2 / len(data_batch)
        F3 = F3 / len(data_batch)
        d_w1 = d_w1 / len(data_batch)
        d_w2 = d_w2 / len(data_batch)
        d_w3 = d_w3 / len(data_batch)

        return F1, F2, F3, d_w1, d_w2, d_w3







