from logistic_oracles import *
from lissa import *
import numpy as np
import pickle
import argparse
from pylab import *
import random, time
mpl.rcParams['font.sans-serif'] = ['SimHei']


data_holder = DataHolder(lam=1e-4, dataset = 'MNIST')
num_examples = data_holder.num_train_examples
h1 = 30
h2 = 20
ite = 5
le = 0.005
w1 = np.random.rand(h1, 785)
w2 = np.random.rand(h2, h1 +1)
w3 = np.random.rand(1, h2 +1)
X1 = np.random.rand(h1, 785)
X2 = np.random.rand(h2, h1 +1)
X3 = np.random.rand(1, h2 +1)
#print(w1)
#print(w2)
a = [0, 1, 2]

print(a[1])