import numpy as np

def relu(x):
    i = len(x)
    y = np.zeros(i)
    for k in range(i):
        if x[k] >= 0:
            y[k] = x[k]
        else:
            y[k] = 0
    return y

x= [1,2,0,-1,3]
y = relu(x)
print(y)