from logistic_oracles import *
from lissa import *
import numpy as np
import pickle
import argparse
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

data_holder = DataHolder(lam=1e-4, dataset = 'MNIST')
num_examples = data_holder.num_train_examples

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', default=25, type=int)
parser.add_argument('--num_lissa_iter', default=num_examples, type=int)
parser.add_argument('--outer_grad_size', default=num_examples, type=int)
parser.add_argument('--hessian_batch_size', default=1, type=int)
parser.add_argument('--grande_batch_size', default=1, type=int)
parser.add_argument('--stepsize', default=1.0, type=float)
args = parser.parse_args()

gd_init_x = np.zeros(data_holder.data_dim + 1)
gd_iter = 5
gd_stepsize = 5.0
init_x = grad_descent(gd_iter, gd_init_x, gd_stepsize, num_examples, data_holder)

num_epochs = args.num_epochs
num_lissa_iter = args.num_lissa_iter
outer_grad_size = args.outer_grad_size
hessian_batch_size = args.hessian_batch_size
#grand_batch_size = args.grande_batch_size
stepsize = args.stepsize
iter = 5

print ('-----------------------------------------------')
print ('Training model...')
print ('-----------------------------------------------\n')

#f = open('../data/mnist.pkl.gz','wb')
output_data_1 = lissa_main(init_x, iter, num_epochs, 9847, 9847, outer_grad_size,
                         hessian_batch_size, stepsize, data_holder)
#pickle.dump(output_data, f)
#f.close()
time_1 = output_data_1['wall_times']
fun_val_1 = output_data_1['trainerror']
print("time_1",time_1)
print('fun_val_1',fun_val_1)


output_data_2 = lissa_main(init_x, iter, num_epochs, 1000, 1000, outer_grad_size,
                         hessian_batch_size, stepsize, data_holder)
#pickle.dump(output_data, f)
#f.close()
time_2 = output_data_2['wall_times']
fun_val_2 = output_data_2['trainerror']
print("time_2",time_2)
print('fun_val_2',fun_val_2)

output_data_3 = SSN(init_x, iter, 1000, 1000, data_holder)
time_3 = output_data_3['wall_times']
fun_val_3 = output_data_3['trainerror']
print("time_3",time_3)
print('fun_val_3',fun_val_3)

print ('-----------------------------------------------')
print ('Training complete')
print ('-----------------------------------------------')
x = [i for i in range(iter)]
plt.plot(x, time_1, marker='o', color="blue",label=u'lissa 时间')
plt.plot(x, time_2, marker='o', color="red",label=u'lissa  sub-hes 时间')
plt.plot(x, time_3, marker='o', color="darkorange",label=u'ssn 时间')
plt.legend()  # 让图例生效
#plt.xticks(x, names, rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"迭代次数") #X轴标签
plt.ylabel("时间") #Y轴标签
plt.title("时间对比")#标题

plt.show()
plt.plot(x, fun_val_1, marker='o', color="blue",label=u'lissa 函数值')
plt.plot(x, fun_val_2, marker='o', color="red",label=u'lissa sub-hes 函数值')
plt.plot(x, fun_val_3, marker='o', color="darkorange",label=u'ssn 函数值')
plt.legend()  # 让图例生效
#plt.xticks(x, names, rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"迭代次数") #X轴标签
plt.ylabel("loss 函数值") #Y轴标签
plt.title("loss函数值对比") #标题

plt.show()