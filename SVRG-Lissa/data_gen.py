import numpy as np

import pickle, gzip, math

print('-----------------------------------------------')
print('Constructing 4/9 MNIST binary classification data')
print('-----------------------------------------------\n')

f = gzip.open('C:/Users/wangjing computer/PycharmProjects/kfac/pythonProject/Data/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='bytes')
f.close()

train_set_49 = train_set[0][np.logical_or(train_set[1] == 4, train_set[1] == 9)]
train_label_set_49 = train_set[1][np.logical_or(train_set[1] == 4, train_set[1] == 9)]
train_label_set_49_corrected = [int(i == 4) for i in train_label_set_49]

test_set_49 = test_set[0][np.logical_or(test_set[1] == 4, test_set[1] == 9)]
test_label_set_49 = test_set[1][np.logical_or(test_set[1] == 4, test_set[1] == 9)]
test_label_set_49_corrected = [int(i == 4)*2 - 1 for i in test_label_set_49]

valid_set_49 = valid_set[0][np.logical_or(valid_set[1] == 4, valid_set[1] == 9)]
valid_label_set_49 = valid_set[1][np.logical_or(valid_set[1] == 4, valid_set[1] == 9)]
valid_label_set_49_corrected = [int(i == 9) for i in valid_label_set_49]



train_set_49_whole = [train_set_49, train_label_set_49_corrected]
test_set_49_whole = [test_set_49, test_label_set_49_corrected]
valid_set_49_whole = [valid_set_49, valid_label_set_49_corrected]

f = open('C:/Users/wangjing computer/PycharmProjects/kfac/pythonProject/Data/mnist49data', 'wb')
pickle.dump([train_set_49_whole, valid_set_49_whole, test_set_49_whole], f)
f.close()

