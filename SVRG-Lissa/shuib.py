import numpy as np
from logistic_oracles import *
import random, time
def svrg(init_x, ite, sub_ite, eta, data_holder):
    #num_examples = data_holder.num_train_examples
    curr_x = init_x

    wall_times = []
    trainerror = []

    # num_iter = num_epochs*num_examples*1.0/(outer_grad_size + num_lissa_iter*hessian_batch_size)
    start_time = time.time()
    i=0
    for curr_iter in range(ite):
        if i<50:
            eta =0.01
        elif 50<=i<100:
            eta = 0.001
        else:
            eta = 0.0001
        print("i", eta)
        wall_times += [time.time() - start_time]
        trainerror += [data_holder.logistic_batch_func(range(0, 9847), curr_x)]
        rand_index_1 = random.sample(range(9847), 9847)
        curr_grad = data_holder.logistic_batch_grad(rand_index_1, curr_x)
        sub_x = curr_x
        for sub_iter in range(sub_ite):
            rand_index_2 = random.sample(range(9847), 1)
            sub_direction = data_holder.logistic_batch_grad(rand_index_2, sub_x)-\
                            data_holder.logistic_batch_grad(rand_index_2, curr_x)+curr_grad
            sub_x = sub_x - eta * sub_direction

        curr_x = sub_x
        i = i+1

    # output_data = {'epochs': epochs, 'wall_times': wall_times, 'trainerror': trainerror}
    #output_data = {'wall_times': wall_times, 'trainerror': trainerror}
    return wall_times, trainerror
def svrg_LISSA(init_x, ite, sub_ite, eta,T,  data_holder):
    #num_examples = data_holder.num_train_examples
    curr_x = init_x

    wall_times = []
    trainerror = []

    # num_iter = num_epochs*num_examples*1.0/(outer_grad_size + num_lissa_iter*hessian_batch_size)
    start_time = time.time()
    i=0
    for curr_iter in range(ite):
        if i<50:
            eta =0.01
        elif 50<=i<100:
            eta = 0.001
        else:
            eta = 0.0001
        print("i", eta)
        wall_times += [time.time() - start_time]
        trainerror += [data_holder.logistic_batch_func(range(0, 9847), curr_x)]
        rand_index_1 = random.sample(range(9847), 9847)
        curr_grad = data_holder.logistic_batch_grad(rand_index_1, curr_x)
        #用lissa近似估计hessian矩阵的逆矩阵
        E = np.identity(784)
        hes = np.identity(784)
        for ite in range(T):
            rand_index = random.sample(range(9847), 1)
            H = data_holder.logistic_batch_hess_full(rand_index, curr_x)
            hes = E + np.dot((E-H),hes)
        ##########################################
        sub_x = curr_x
        for sub_iter in range(sub_ite):
            rand_index_2 = random.sample(range(9847), 1)
            sub_direction = data_holder.logistic_batch_grad(rand_index_2, sub_x)-\
                            data_holder.logistic_batch_grad(rand_index_2, curr_x)+curr_grad
            sub_x = sub_x - eta * np.dot(hes,sub_direction)

        curr_x = sub_x
        i = i+1

    # output_data = {'epochs': epochs, 'wall_times': wall_times, 'trainerror': trainerror}
    #output_data = {'wall_times': wall_times, 'trainerror': trainerror}
    return wall_times, trainerror
def svrg_Newsamp(init_x, ite, sub_ite, eta,T,  r, data_holder):
    # r--表示特征值分解后取前面r个最大特征值
    #num_examples = data_holder.num_train_examples
    curr_x = init_x

    wall_times = []
    trainerror = []

    # num_iter = num_epochs*num_examples*1.0/(outer_grad_size + num_lissa_iter*hessian_batch_size)
    start_time = time.time()
    i=0
    for curr_iter in range(ite):
        if i<50:
            eta =0.01
        elif 50<=i<100:
            eta = 0.001
        else:
            eta = 0.0001
        print("i", eta)
        wall_times += [time.time() - start_time]
        trainerror += [data_holder.logistic_batch_func(range(0, 9847), curr_x)]
        rand_index_1 = random.sample(range(9847), 9847)
        curr_grad = data_holder.logistic_batch_grad(rand_index_1, curr_x)
        #用newsample近似估计hessian矩阵的逆矩阵##################################################################
        rand_index = random.sample(range(9847), T)
        H = data_holder.logistic_batch_hess_full(rand_index, curr_x)
        #对H进行特征值分解
        eigvals, eigvecs = np.linalg.eig(H)
        eigvals[r:] = eigvals[r]
        #求解特征向量形成的矩阵的逆矩阵
        vec_inv = np.linalg.inv(eigvecs)
        eig_inv = np.linalg.inv(np.diag(eigvals))
        H_inv = vec_inv.T * eig_inv *vec_inv
        #####################################################################################################
        sub_x = curr_x
        for sub_iter in range(sub_ite):
            rand_index_2 = random.sample(range(9847), 1)
            sub_direction = data_holder.logistic_batch_grad(rand_index_2, sub_x)-\
                            data_holder.logistic_batch_grad(rand_index_2, curr_x)+curr_grad
            sub_x = sub_x - eta * np.dot(H_inv, sub_direction)

        curr_x = sub_x
        i = i+1

    # output_data = {'epochs': epochs, 'wall_times': wall_times, 'trainerror': trainerror}
    #output_data = {'wall_times': wall_times, 'trainerror': trainerror}
    return wall_times, trainerror

##############################################################################
def svrg_BFGS(init_x, ite, sub_ite, eta,  data_holder):
    # lam--表示bfgs预测下次结果的步长
    #num_examples = data_holder.num_train_examples
    curr_x = init_x

    wall_times = []
    trainerror = []

    # num_iter = num_epochs*num_examples*1.0/(outer_grad_size + num_lissa_iter*hessian_batch_size)
    start_time = time.time()
    i=0
    for curr_iter in range(ite):
        if i<50:
            eta =0.01
        elif 50<=i<100:
            eta = 0.001
        else:
            eta = 0.0001
        print("i", eta)
        wall_times += [time.time() - start_time]
        trainerror += [data_holder.logistic_batch_func(range(0, 9847), curr_x)]
        rand_index_1 = random.sample(range(9847), 9847)
        curr_grad = data_holder.logistic_batch_grad(rand_index_1, curr_x)
        E = np.identity(784)
        #用newsample近似估计hessian矩阵的逆矩阵##################################################################
        #确定搜索方向d_k=-D_K * g_k
        if i ==0:
            D = E
        else:
            D= D_
        d_k = -1 * np.dot(D, curr_grad)
        up_dic = eta * d_k
        up_x = curr_x + up_dic
        up_grad = data_holder.logistic_batch_grad(rand_index_1, up_x)
        a = np.dot(up_grad, up_x - curr_x)
        A = (-1/a)*np.outer(up_grad, up_x - curr_x)
        B = (1/a)*np.outer(up_x - curr_x, up_x - curr_x)
        D = np.dot((E - A),np.dot(D, (E-A.T))) + B
        D_ = D
        #####################################################################################################
        sub_x = curr_x
        for sub_iter in range(sub_ite):
            rand_index_2 = random.sample(range(9847), 1)
            sub_direction = data_holder.logistic_batch_grad(rand_index_2, sub_x)-\
                            data_holder.logistic_batch_grad(rand_index_2, curr_x)+curr_grad
            sub_x = sub_x - eta * np.dot(D, sub_direction)

        curr_x = sub_x
        i = i+1

    # output_data = {'epochs': epochs, 'wall_times': wall_times, 'trainerror': trainerror}
    #output_data = {'wall_times': wall_times, 'trainerror': trainerror}
    return wall_times, trainerror

#######################################################################################
##########################################################################################
ite =30
sub_ite = 2000
T_1 = 10
T_2 = 100
r = 300
lam = 0.001
data_holder = DataHolder(lam=1e-4, dataset = 'MNIST')
init_x = np.zeros(784)
time1, error1 = svrg(init_x, ite, sub_ite, 0.01, data_holder)
time11 = [i for i in time1]
error11 = [i for i in error1]
print("time",time1)
print("error",error1)
########
time2,error2 = svrg_LISSA(init_x, ite, sub_ite, 0.01 ,T_1,  data_holder)
time22 = [i for i in time2]
error22 = [i for i in error2]
print("time",time22)
print("error",error22)
########
time3,error3 = svrg_Newsamp(init_x, ite, sub_ite, 0.01 ,T_2,  r, data_holder)
time33 = [i for i in time3]
error33 = [i for i in error3]
print("time",time33)
print("error",error33)
############
time4,error4 = svrg_BFGS(init_x, ite, sub_ite, 0.01, data_holder)
time44 = [i for i in time4]
error44 = [i for i in error4]
print("time",time44)
print("error",error44)

#########################################################
#####################################################

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
x = [i+1 for i in range(ite)]
print(x)
plt.plot(x, time11, marker='o', linestyle=':', color="black",label=u'SVRG 时间')
plt.plot(x, time22, marker='^', linestyle='--', color="black",label=u'SVRG_Lissa 时间')
plt.plot(x, time33, marker='*',linestyle='-.',  color="black",label=u'SVRG_newsample时间')
plt.plot(x, time44, marker='x', linestyle='solid', color="black",label=u'SVRG_BFGS 时间')
plt.legend()  # 让图例生效
#plt.xticks(x, names, rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"迭代次数") #X轴标签
plt.ylabel("时间") #Y轴标签
plt.title("时间对比")#标题

plt.show()

x = [i for i in range(ite)]
plt.plot(x, error11, marker='o',linestyle=':', color="black",label=u'svrg 函数值')
plt.plot(x, error22, marker='^', linestyle='--', color="black",label=u'SVRG_Lissa 函数值')
plt.plot(x, error33, marker='*', linestyle='-.', color="black",label=u'SVRG_newsample函数值')
plt.plot(x, error44, marker='x', linestyle='solid', color="black",label=u'SVRG_BFGS 函数值')
plt.legend()  # 让图例生效
#plt.xticks(x, names, rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"迭代次数") #X轴标签
plt.ylabel("函数值") #Y轴标签
plt.title("函数值对比")#标题

plt.show()