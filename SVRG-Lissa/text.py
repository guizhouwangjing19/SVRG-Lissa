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
h1 = 5
h2 = 2
ite = 10
le1 = 0.001
le2 = 1.0
m = 10
w1 = np.random.rand(m, 785)
w2 = np.random.rand(h2, h1 +1)
w3 = np.random.rand(1, h2 +1)
x = grand(w1, data_holder,10, 0.001)
#print("x.shape",x.shape)

t1, f1= K_FAC(w1, 0.001, ite, data_holder, 3, 0.4,100)
print(t1, f1)
t2, f2= K_FAC_TR(w1, 0.001, ite, data_holder, 50, 1000, 0.5, 3, 0.4, 10, 0.001)
print(t2, f2)
#t3, f3  = increace_kfac_full_grad(w1, w2, w3, 0.001, ite, le1,data_holder)
#print(t3, f3)
#t4, f4  = grad(w1, w2, w3, ite, le2, data_holder)
#print(t4, f4)
#t5, f5 = increace_grad(w1,w2, w3, ite, le2,data_holder)
#print(t5, f5)
#t6, f6 = increace_kfac_moment(w1, w2, w3, 0.001, ite, data_holder)
#print(t6, f6)
#t7 ,f7 = increace_grad_moment(w1, w2, w3, ite, le2, data_holder)
#print(t7, f7)
#t8 ,f8= kfac_moment(w1, w2, w3, 0.001, ite, data_holder)
#print(t8 ,f8)
#t9, f9 = increace_kfac_moment_jacobi(w1, w2, w3, 0.001, ite, data_holder)
#print(t9, f9)
#t10,f10 = k_fac_TR(w1, w2, w3, 0.1, 0.1, ite, data_holder)
#print(t10,f10)

x = [i for i in range(ite)]

plt.plot(x, t1, marker='o', color="blue",label=u'kfac全梯度 时间')
plt.plot(x, t2, marker='^', color="green",label=u'kfac增量 时间')
#plt.plot(x, t3, marker='+', color="red",label=u'kfac增量全梯度 时间')
#plt.plot(x, t4, marker='*', color="darkorange",label=u'random grad 时间')
#plt.plot(x, t5, marker='x', color="cyan",label=u'random grad增量 时间')
#plt.plot(x, t6, marker='4', color="peru",label=u'kfac增量+动量 时间')
#plt.plot(x, t7, marker='2', color="teal",label=u'random grad增量+动量 时间')
#plt.plot(x, t8, marker='^', color="blue",label=u'kfac+动量 时间')
#plt.plot(x, t9, marker='2', color="lightpink",label=u'kfac+动量+增量+线性方程 时间')
#plt.plot(x, t10, marker='2', color="lightpink",label=u'kfac+信赖域 时间')
plt.legend()  # 让图例生效
#plt.xticks(x, names, rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"迭代次数") #X轴标签
plt.ylabel("时间") #Y轴标签
plt.title("时间对比")                                                                                         #标题

plt.show()

plt.plot(x, f1, marker='o', color="blue",label=u'kfac全梯度 函数值')
plt.plot(x, f2, marker='^', color="green",label=u'kfac增量 函数值')
#plt.plot(x, f3, marker='+', color="red",label=u'kfac增量全梯度 函数值')
#plt.plot(x, f4, marker='*', color="darkorange",label=u'random grad 函数值')
#plt.plot(x, f5, marker='s', color="cyan",label=u'random grad增量 函数值')
#plt.plot(x, f6, marker='4', color="peru",label=u'kfac增量+动量 函数值')
#plt.plot(x, f7, marker='2', color="teal",label=u'random grad增量+动量 函数值')
#plt.plot(x, f8, marker='^', color="blue",label=u'kfac+动量 函数值')
#plt.plot(x, f9, marker='2', color="lightpink",label=u'kfac+动量+增量+线性方程 函数值')
#plt.plot(x, f10, marker='2', color="lightpink",label=u'kfac+信赖域 函数值')
plt.legend()  # 让图例生效
#plt.xticks(x, names, rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"迭代次数") #X轴标签
plt.ylabel("loss 函数值") #Y轴标签
plt.title("loss函数值对比") #标题

plt.show()

