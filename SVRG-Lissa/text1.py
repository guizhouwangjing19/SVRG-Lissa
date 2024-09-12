from mean import *
import numpy as np
from kfac import *
from lissa import *
import numpy as np
import pickle
import argparse
from pylab import *
import random, time
h1 = 5
h2 = 2
ite = 10
x= [3, -1, 4, -2, -1000]
w1 = np.random.rand(h1, 784)
w2 = np.random.rand(h2, h1)
w3 = np.random.rand(1, h2)
#print("w3",w3)

def grand(w1, w2, w3, lam, ite, le, kfac):
	num_examples = KFAC().num_train_examples
	wall_times = []
	func_val = []
	#test_error = []
	#train_err = []
	start_time = time.time()

	for curr_iter in range(ite):

		wall_times += [time.time() - start_time]
		func_val += [KFAC().net_loss(range(0, num_examples), w1, w2, w3, mode='TRAIN')[0][0]]
		#test_error += [KFAC().test_error(w1, w2, w3)]
		batch= random.sample(range(num_examples), 128)
		F1, F2, F3, d_w1, d_w2, d_w3 = KFAC().loss_grand_A_G_AVE(batch, w1, w2, w3)


		w1 = w1 - le * d_w1
		w2 = w2 - le * d_w2
		w3 = w3 - le * d_w3

	#output_data = {'wall_times': wall_times, 'trainerror': trainerror}
	return wall_times, func_val#, test_error#, train_err

def kfac(w1, w2, w3, lam, eta, ite, le, kfac):
	num_examples = KFAC().num_train_examples
	wall_times = []
	func_val = []
	#test_error = []
	#train_err = []
	start_time = time.time()
	i = 0
	for curr_iter in range(ite):
		print("i",i)
		wall_times += [time.time() - start_time]
		func_val += [KFAC().net_loss(range(0, num_examples), w1, w2, w3, mode='TRAIN')[0][0]]
		#test_error += [KFAC().test_error(w1, w2, w3)]
		batch= random.sample(range(num_examples), 128)
		F1, F2, F3, d_w1, d_w2, d_w3 = KFAC().loss_grand_A_G_AVE(batch, w1, w2, w3)

		d_w1 = d_w1.reshape((784*h1, 1))
		d_w2 = d_w2.reshape((h2 * h1, 1))
		d_w3 = d_w3.reshape((h2, 1))
		delta_1 = -1 * np.dot(np.linalg.inv(F1 + (lam + eta)* np.identity(784 * h1)),d_w1)
		delta_2 = -1 * np.dot(np.linalg.inv(F2 + (lam + eta) * np.identity(h2 * h1)), d_w2)
		delta_3 = -1 * np.dot(np.linalg.inv(F3 + (lam + eta) * np.identity(h2)), d_w3)
		if i >0:
			a = np.dot(delta_1.T, np.dot(F1 + (lam + eta)* np.identity(784 * h1), delta_1)) +\
				np.dot(delta_2.T, np.dot(F2 + (lam + eta) * np.identity(h2 * h1), delta_2)) + \
				np.dot(delta_3.T, np.dot(F3 + (lam + eta) * np.identity(h2), delta_3))
			b = np.dot(delta_1.T, np.dot(F1 + (lam + eta) * np.identity(784 * h1), delta_1_0)) + \
				np.dot(delta_2.T, np.dot(F2 + (lam + eta) * np.identity(h2 * h1), delta_2_0)) + \
				np.dot(delta_3.T, np.dot(F3 + (lam + eta) * np.identity(h2), delta_3_0))
			c = np.dot(delta_1_0.T, np.dot(F1 + (lam + eta) * np.identity(784 * h1), delta_1_0)) + \
				np.dot(delta_2_0.T, np.dot(F2 + (lam + eta) * np.identity(h2 * h1), delta_2_0)) + \
				np.dot(delta_3_0.T, np.dot(F3 + (lam + eta) * np.identity(h2), delta_3_0))
			e = np.dot(d_w1.T, delta_1) + np.dot(d_w2.T, delta_2) + np.dot(d_w3.T, delta_3)
			f = np.dot(d_w1.T, delta_1_0) + np.dot(d_w2.T, delta_2_0) + np.dot(d_w3.T, delta_3_0)
			alpha = (b * f - c * e)/(a * c - b * b)
			miu = (b * e - a * f)/(a * c - b * b)
		else:
			alpha = 1
			miu = 0
			delta_1_0 = np.zeros((784 * h1, 1))
			delta_2_0 = np.zeros((h2 * h1, 1))
			delta_3_0 = np.zeros((h2, 1))

		print("alpha",alpha)
		print("miu",miu)
		func_jiu = KFAC().net_loss(range(0, num_examples), w1, w2, w3, mode='TRAIN')[0][0]
		dir_1_1 = alpha * delta_1 + miu * delta_1_0
		dir_2_1 = alpha * delta_2 + miu * delta_2_0
		dir_3_1 = alpha * delta_3 + miu * delta_3_0
		dir_1 = dir_1_1.reshape((h1, 784))
		dir_2 = dir_2_1.reshape((h2, h1))
		dir_3 = dir_3_1.reshape((1, h2))
		w1 = w1  + dir_1
		w2 = w2 + dir_2
		w3 = w3 + dir_3
		func_xin = KFAC().net_loss(range(0, num_examples), w1, w2, w3, mode='TRAIN')[0][0]
		pre = 0.5 * np.dot(dir_1_1.T, np.dot(F1 + (lam + eta)* np.identity(784 * h1), dir_1_1))+ \
			  0.5 * np.dot(dir_2_1.T, np.dot(F2 + (lam + eta) * np.identity(h2 * h1), dir_2_1)) + \
			  0.5 * np.dot(dir_3_1.T, np.dot(F3 + (lam + eta) * np.identity(h2), dir_3_1)) +\
			np.dot(d_w1.T, dir_1_1) + np.dot(d_w2.T, dir_2_1) + np.dot(d_w3.T, dir_3_1)

		rou = (func_xin - func_jiu)/pre
		print("rou",rou)
		if rou>0.75:
			lam = 0.5 * lam
		elif rou < 0.25:
			lam = 2 * lam
		else:
			lam = lam
		print("lam",lam)

		delta_1_0 = dir_1_1
		delta_2_0 = dir_2_1
		delta_3_0 = dir_3_1

		i = i + 1

	return wall_times, func_val#, test_error#, train_err


t1, f1= grand(w1, w2, w3, 3, ite, 0.001, kfac)
print("t1",t1)
print("f1",f1)

t2, f2 = kfac(w1, w2, w3, 0.1, 3, ite, 0.001, kfac)
print("t2",t2)
print("f2",f2)




x = [i for i in range(ite)]

plt.plot(x, t1, marker='o', color="blue",label=u'grand 时间')
plt.plot(x, t2, marker='o', color="red",label=u'kfac 时间')

plt.legend()  # 让图例生效
#plt.xticks(x, names, rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"迭代次数") #X轴标签
plt.ylabel("时间") #Y轴标签
plt.title("时间对比")                                                                                         #标题

plt.show()

plt.plot(x, f1, marker='o', color="blue",label=u'grand 函数值')
plt.plot(x, f2, marker='o', color="red",label=u'kfac 函数值')

plt.legend()  # 让图例生效
#plt.xticks(x, names, rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"迭代次数") #X轴标签
plt.ylabel("loss 函数值") #Y轴标签
plt.title("loss函数值对比") #标题

plt.show()




