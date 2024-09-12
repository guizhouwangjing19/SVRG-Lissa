import numpy as np
from logistic_oracles import *
import random, time 
h1 = 3
h2 = 4
h3 = 5
mo = 0.3
def learning_rate(i, le):
	if i % 2000 == 0:
		le = le * 0.1
	else:
		le = le

	return le

def lissa_main(init_x, iter, num_epochs, num_lissa_iter, outer_grad_size, grande_batch_size, hessian_batch_size,
			   quad_stepsize, data_holder):
	num_examples = data_holder.num_train_examples
	curr_x = init_x

	epochs = []
	wall_times = []
	trainerror = []
	
	#num_iter = num_epochs*num_examples*1.0/(outer_grad_size + num_lissa_iter*hessian_batch_size)
	start_time = time.time()

	for curr_iter in range(iter):
		epochs += [curr_iter*(outer_grad_size + num_lissa_iter*hessian_batch_size)/num_examples]
		wall_times += [time.time() - start_time]
		trainerror += [data_holder.logistic_batch_func(range(0, num_examples), curr_x)]
		rand_index_1 = random.sample(range(num_examples), grande_batch_size)
		curr_grad = data_holder.logistic_batch_grad(rand_index_1, curr_x)
		curr_step = np.zeros(curr_x.size)
		print("num_lissa_iter", num_lissa_iter)
		for lissa_iter in range(num_lissa_iter):
			rand_index_2 = random.sample(range(num_examples), hessian_batch_size)
			sub_step = data_holder.logistic_batch_hess_vec_product(rand_index_2, curr_x, curr_step)
			curr_quad_step = curr_grad - sub_step

			curr_step = curr_step + quad_stepsize*curr_quad_step

		curr_x = curr_x - curr_step

	#output_data = {'epochs': epochs, 'wall_times': wall_times, 'trainerror': trainerror}
	output_data = {'wall_times': wall_times, 'trainerror': trainerror}
	return output_data


def SSN(init_x, iter, grande_batch_size, hessian_batch_size, data_holder):
	num_examples = data_holder.num_train_examples
	curr_x = init_x

	#epochs = []
	wall_times = []
	trainerror = []

	# num_iter = num_epochs*num_examples*1.0/(outer_grad_size + num_lissa_iter*hessian_batch_size)
	start_time = time.time()

	for curr_iter in range(iter):
		#epochs += [curr_iter * (outer_grad_size + num_lissa_iter * hessian_batch_size) / num_examples]
		wall_times += [time.time() - start_time]
		trainerror += [data_holder.logistic_batch_func(range(0, num_examples), curr_x)]
		rand_index_1 = random.sample(range(num_examples), grande_batch_size)
		curr_grad = data_holder.logistic_batch_grad(rand_index_1, curr_x)
		rand_index_2 = random.sample(range(num_examples), hessian_batch_size)
		curr_hess = data_holder.logistic_batch_hess_full(rand_index_2, curr_x)
		curr_hess_inv = np.linalg.inv(curr_hess)
		dir = -1 * np.dot(curr_hess_inv, curr_grad)

		curr_x = curr_x - 0.01 * dir

	# output_data = {'epochs': epochs, 'wall_times': wall_times, 'trainerror': trainerror}
	output_data = {'wall_times': wall_times, 'trainerror': trainerror}
	return output_data


#####################################################################################################
#fisher 更新算法
def fisher_main(init_x, data_holder):
	num_examples = data_holder.num_train_examples
	curr_x = init_x


	wall_times = []
	trainerror = []
	start_time = time.time()

	for curr_iter in range(int(12)):

		wall_times += [time.time() - start_time]
		trainerror += [data_holder.logistic_batch_func(range(0, num_examples), curr_x)]
		curr_step = data_holder.fisher_dir(random.sample(range(num_examples), 1000), curr_x)

		curr_x = curr_x - 0.2* curr_step

	output_data = {'wall_times': wall_times, 'trainerror': trainerror}
	return output_data
######################################################################################################

#####################################################################################################
mini_siae = 128
num = 9600

#fisher 更新算法
def kfac(w1, w2, w3, lam, ite, le, data_holder):
	num_examples = data_holder.num_train_examples


	wall_times = []
	func_val = []
	#train_err = []
	start_time = time.time()

	for curr_iter in range(ite):

		wall_times += [time.time() - start_time]
		func_val += [data_holder.full_loss_func(range(0,num_examples), w1, w2, w3)]
		#train_err += [data_holder.error_01(w1, w2, w3, mode='TRAIN')]
		ave_D_w1, ave_D_w2, ave_D_w3= data_holder.batch_D_wi(range(0, num_examples), w1, w2, w3)
		#ave_F_11_dir,ave_F_22_dir, ave_F_33_dir = data_holder.F((random.sample(range(num_examples), mini_siae)), w1, w2, w3, ave_D_w1, ave_D_w2,ave_D_w3, lam)
		ave_F_11_dir, ave_F_22_dir, ave_F_33_dir = data_holder.F(range(0, num_examples), w1,
																 w2, w3, ave_D_w1, ave_D_w2, ave_D_w3, lam)

		w1 = w1 - le* ave_F_11_dir
		w2 = w2 - le* ave_F_22_dir
		w3 = w3 - le * ave_F_33_dir

	#output_data = {'wall_times': wall_times, 'trainerror': trainerror}
	return wall_times, func_val#, train_err


def kfac_moment(w1, w2, w3, lam, ite,data_holder):
	num_examples = data_holder.num_train_examples


	wall_times = []
	func_val = []
	#train_err = []
	start_time = time.time()
	dir_1 = np.zeros((h1, 785))
	dir_2 = np.zeros((h2, h1 + 1))
	dir_3 = np.zeros((1, h2 + 1))
	j = 0
	le = 1.0
	for curr_iter in range(ite):
		j = j + 1
		le = learning_rate(j, le)
		print(le)

		wall_times += [time.time() - start_time]
		func_val += [data_holder.full_loss_func(range(0,num_examples), w1, w2, w3)]
		#train_err += [data_holder.error_01(w1, w2, w3, mode='TRAIN')]
		batch = random.sample(range(num_examples), 128)
		ave_D_w1, ave_D_w2, ave_D_w3 = data_holder.batch_D_wi(batch, w1, w2, w3)
		ave_F_11_dir, ave_F_22_dir, ave_F_33_dir = data_holder.F(batch, w1, w2, w3, ave_D_w1, ave_D_w2,ave_D_w3, lam)

		w1 = w1 - le * (ave_F_11_dir+ 0.9*dir_1)
		w2 = w2 - le * (ave_F_22_dir+ 0.9*dir_2)
		w3 = w3 - le * (ave_F_33_dir+ 0.9*dir_3)

		dir_1 = ave_F_11_dir
		dir_2 = ave_F_22_dir
		dir_3 = ave_F_33_dir

	#output_data = {'wall_times': wall_times, 'trainerror': trainerror}
	return wall_times, func_val#, train_err
########################################################################################
def increace_kfac(w1, w2, w3, lam, ite, data_holder):
	num_examples = data_holder.num_train_examples


	wall_times = []
	func_val = []
	#train_err = []
	start_time = time.time()
	i= 0
	j=0
	le = 3.0
	for curr_iter in range(ite):
		j = j + 1
		le = learning_rate(j, le)
		print(le)

		wall_times += [time.time() - start_time]
		func_val += [data_holder.full_loss_func(range(0,num_examples), w1, w2, w3)]
		#train_err += [data_holder.error_01(w1, w2, w3, mode='TRAIN')]

		ave_D_w1, ave_D_w2, ave_D_w3 = data_holder.batch_D_wi(range(i,i+mini_siae), w1, w2, w3)
		ave_F_11_dir,ave_F_22_dir, ave_F_33_dir = data_holder.F(range(i,i+mini_siae), w1, w2, w3, ave_D_w1, ave_D_w2, ave_D_w3, lam)

		w1 = w1 - le* ave_F_11_dir
		w2 = w2 - le* ave_F_22_dir
		w3 = w3 - le * ave_F_33_dir
		if i<num:
			i= i+ mini_siae
		else:
			i = 0

	#output_data = {'wall_times': wall_times, 'trainerror': trainerror}
	return wall_times, func_val#, train_err

def increace_kfac_moment(w1, w2, w3, lam, ite, data_holder):
	num_examples = data_holder.num_train_examples


	wall_times = []
	func_val = []
	#train_err = []
	start_time = time.time()
	i= 0
	dir_1 = np.zeros((h1, 785))
	dir_2 = np.zeros((h2, h1 + 1))
	dir_3 = np.zeros((1, h2 + 1))
	j=0
	le = 0.1
	for curr_iter in range(ite):
		j = j + 1
		le = learning_rate(j, le)
		print(le)

		wall_times += [time.time() - start_time]
		func_val += [data_holder.full_loss_func(range(0,num_examples), w1, w2, w3)]
		#train_err += [data_holder.error_01(w1, w2, w3, mode='TRAIN')]

		ave_D_w1, ave_D_w2, ave_D_w3 = data_holder.batch_D_wi(range(i,i+mini_siae), w1, w2, w3)
		ave_F_11_dir,ave_F_22_dir, ave_F_33_dir = data_holder.F(range(i,i+mini_siae), w1, w2, w3, ave_D_w1, ave_D_w2, ave_D_w3, lam)

		w1 = w1 - le* (ave_F_11_dir + mo * dir_1)
		w2 = w2 - le* (ave_F_22_dir + mo * dir_2)
		w3 = w3 - le * (ave_F_33_dir + mo * dir_3)
		if i<num:
			i= i+ mini_siae
		else:
			i = 0
		dir_1 = ave_F_11_dir
		dir_2 = ave_F_22_dir
		dir_3 = ave_F_33_dir

	#output_data = {'wall_times': wall_times, 'trainerror': trainerror}
	return wall_times, func_val#, train_err
#######################################################################################
def increace_kfac_full_grand_moment(w1, w2, w3, lam, ite, data_holder):
	num_examples = data_holder.num_train_examples


	wall_times = []
	func_val = []
	#train_err = []
	start_time = time.time()
	i= 0
	dir_1 = np.zeros((h1, 785))
	dir_2 = np.zeros((h2, h1 + 1))
	dir_3 = np.zeros((1, h2 + 1))
	j=0
	le = 0.1
	for curr_iter in range(ite):
		j = j + 1
		le = learning_rate(j, le)
		print(le)

		wall_times += [time.time() - start_time]
		func_val += [data_holder.full_loss_func(range(0,num_examples), w1, w2, w3)]
		#train_err += [data_holder.error_01(w1, w2, w3, mode='TRAIN')]

		ave_D_w1, ave_D_w2, ave_D_w3 = data_holder.batch_D_wi(range(0,num_examples), w1, w2, w3)
		ave_F_11_dir,ave_F_22_dir, ave_F_33_dir = data_holder.F(range(i,i+mini_siae), w1, w2, w3, ave_D_w1, ave_D_w2, ave_D_w3, lam)

		w1 = w1 - le* (ave_F_11_dir + mo * dir_1)
		w2 = w2 - le* (ave_F_22_dir + mo * dir_2)
		w3 = w3 - le * (ave_F_33_dir + mo * dir_3)
		if i<num:
			i= i+ mini_siae
		else:
			i = 0
		dir_1 = ave_F_11_dir
		dir_2 = ave_F_22_dir
		dir_3 = ave_F_33_dir

	#output_data = {'wall_times': wall_times, 'trainerror': trainerror}
	return wall_times, func_val#, train_err
############################################################################

def increace_kfac_moment_jacobi(w1, w2, w3, lam, ite, data_holder):
	num_examples = data_holder.num_train_examples


	wall_times = []
	func_val = []
	#train_err = []
	start_time = time.time()
	i = 0
	dir_1 = np.zeros((h1, 785))
	dir_2 = np.zeros((h2, h1 + 1))
	dir_3 = np.zeros((1, h2 + 1))
	j = 0
	le = 0.1
	for curr_iter in range(ite):
		j = j + 1
		le = learning_rate(j, le)
		print(le)

		wall_times += [time.time() - start_time]
		func_val += [data_holder.full_loss_func(range(0,num_examples), w1, w2, w3)]
		#train_err += [data_holder.error_01(w1, w2, w3, mode='TRAIN')]

		ave_D_w1, ave_D_w2, ave_D_w3 = data_holder.batch_D_wi(range(i,i+mini_siae), w1, w2, w3)
		print(ave_D_w1.shape)
		x1 = ave_D_w1.reshape((785*h1), order='F')
		x2 = ave_D_w2.reshape((h2 * (h1 + 1)), order='F')
		x3 = ave_D_w3.reshape((h2 + 1), order='F')

		ave_F_11_dir,ave_F_22_dir, ave_F_33_dir = data_holder.kro_lin(range(i,i+mini_siae), w1, w2, w3, x1, x2, x3)

		ave_f_11 = ave_F_11_dir.reshape((h1, 785))
		ave_f_22 = ave_F_22_dir.reshape((h2, h1 + 1))
		ave_f_33 = ave_F_33_dir.reshape((1, h2 + 1))

		w1 = w1 - le * (ave_f_11 + mo * dir_1)
		w2 = w2 - le * (ave_f_22 + mo * dir_2)
		w3 = w3 - le * (ave_f_33 + mo * dir_3)
		if i < num:
			i = i + mini_siae
		else:
			i = 0
		dir_1 = ave_f_11
		dir_2 = ave_f_22
		dir_3 = ave_f_33

	#output_data = {'wall_times': wall_times, 'trainerror': trainerror}
	return wall_times, func_val#, train_err

def k_fac_TR(w1, w2, w3, lam, eta, ite, data_holder):
	num_examples = data_holder.num_train_examples

	wall_times = []
	func_val = []
	# train_err = []
	start_time = time.time()
	i = 0
	dir_1 = np.zeros((h1, 785))
	dir_2 = np.zeros((h2, h1 + 1))
	dir_3 = np.zeros((1, h2 + 1))
	j = 0
	le = 0.1
	delta = 100
	delta_max = 5000
	for curr_iter in range(ite):
		j = j + 1
		le = learning_rate(j, le)
		print(le)

		wall_times += [time.time() - start_time]
		func_val += [data_holder.full_loss_func(range(0, num_examples), w1, w2, w3)]
		# train_err += [data_holder.error_01(w1, w2, w3, mode='TRAIN')]

		ave_D_w1, ave_D_w2, ave_D_w3 = data_holder.batch_D_wi(range(0, num_examples), w1, w2, w3)
		print(ave_D_w1.shape)
		x1 = ave_D_w1.reshape((785 * h1), order='F')
		x2 = ave_D_w2.reshape((h2 * (h1 + 1)), order='F')
		x3 = ave_D_w3.reshape((h2 + 1), order='F')
		delta_k, r_k, d_k_1, d_k_2, d_k_3 = data_holder.delta(delta, delta_max, range(i,i+mini_siae),w1, w2, w3, x1, x2, x3,lam)
		delta = delta_k
		if r_k>eta:
			d_k_1 = d_k_1.reshape((h1, 785))
			d_k_2 = d_k_2.reshape((h2, h1 + 1))
			d_k_3 = d_k_3.reshape((1, h2 + 1))

			w1 = w1 + d_k_1
			w2 = w2 + d_k_2
			w3 = w3 + d_k_3

		else:
			w1 = w1
			w2 = w2
			w3 = w3
		if i < num:
			i = i + mini_siae
		else:
			i = 0

	# output_data = {'wall_times': wall_times, 'trainerror': trainerror}
	return wall_times, func_val  # , train_err



######################################################################################
def increace_kfac_full_grad(w1, w2, w3, lam, ite, le, data_holder):
	num_examples = data_holder.num_train_examples


	wall_times = []
	func_val = []
	#train_err =[]
	start_time = time.time()
	i= 0
	for curr_iter in range(ite):



		wall_times += [time.time() - start_time]
		func_val += [data_holder.full_loss_func(range(0,num_examples), w1, w2, w3)]
		#train_err += [data_holder.error_01(w1, w2, w3, mode='TRAIN')]

		ave_D_w1, ave_D_w2, ave_D_w3 = data_holder.batch_D_wi(range(0,num_examples), w1, w2, w3)
		ave_F_11_dir,ave_F_22_dir, ave_F_33_dir = data_holder.F(range(i,i+mini_siae), w1, w2, w3, ave_D_w1, ave_D_w2, ave_D_w3, lam)

		w1 = w1 - le* ave_F_11_dir
		w2 = w2 - le* ave_F_22_dir
		w3 = w3 - le * ave_F_33_dir
		if i < num:
			i = i + mini_siae
		else:
			i = 0

	#output_data = {'wall_times': wall_times, 'trainerror': trainerror}
	return wall_times, func_val#, train_err


#######################################################################################
def grad(w1, w2, w3, ite, le, data_holder):
	num_examples = data_holder.num_train_examples

	wall_times = []
	func_val = []
	#train_err= []
	start_time = time.time()

	for curr_iter in range(ite):
		wall_times += [time.time() - start_time]
		func_val += [data_holder.full_loss_func(range(0, num_examples), w1, w2, w3)]
		#train_err += [data_holder.error_01(w1, w2, w3, mode='TRAIN')]
		ave_D_w1, ave_D_w2, ave_D_w3 = data_holder.batch_D_wi((random.sample(range(num_examples), mini_siae)), w1, w2, w3)

		w1 = w1 - le * ave_D_w1
		w2 = w2 - le * ave_D_w2
		w3 = w3 - le * ave_D_w3

	#output_data = {'wall_times': wall_times, 'trainerror': trainerror}
	return wall_times, func_val#, train_err

###############################################################################
def increace_grad(w1, w2, w3, ite, le, data_holder):
	num_examples = data_holder.num_train_examples

	wall_times = []
	func_val = []
	#train_err =[]
	start_time = time.time()
	i= 0
	for curr_iter in range(ite):
		wall_times += [time.time() - start_time]
		func_val += [data_holder.full_loss_func(range(0, num_examples), w1, w2, w3)]
		#train_err += [data_holder.error_01(w1, w2, w3, mode='TRAIN')]
		ave_D_w1, ave_D_w2, ave_D_w3 = data_holder.batch_D_wi(range(i, i+mini_siae), w1, w2, w3)

		w1 = w1 - le * ave_D_w1
		w2 = w2 - le * ave_D_w2
		w3 = w3 - le * ave_D_w3
		if i < num:
			i = i + mini_siae
		else:
			i = 0

	#output_data = {'wall_times': wall_times, 'trainerror': trainerror}
	return wall_times, func_val#, train_err

def increace_grad_moment(w1, w2, w3, ite, le, data_holder):
	num_examples = data_holder.num_train_examples

	wall_times = []
	func_val = []
	#train_err =[]
	start_time = time.time()
	i= 0
	dir_1 = np.zeros((h1, 785))
	dir_2 = np.zeros((h2, h1 + 1))
	dir_3 = np.zeros((1, h2 + 1))
	for curr_iter in range(ite):
		wall_times += [time.time() - start_time]
		func_val += [data_holder.full_loss_func(range(0, num_examples), w1, w2, w3)]
		#train_err += [data_holder.error_01(w1, w2, w3, mode='TRAIN')]
		ave_D_w1, ave_D_w2, ave_D_w3 = data_holder.batch_D_wi(range(i, i+mini_siae), w1, w2, w3)

		w1 = w1 - le * (ave_D_w1+ mo * dir_1)
		w2 = w2 - le * (ave_D_w2+ mo * dir_2)
		w3 = w3 - le * (ave_D_w3+ mo * dir_3)
		if i < num:
			i = i + mini_siae
		else:
			i = 0
		dir_1 = ave_D_w1
		dir_2 = ave_D_w2
		dir_3 = ave_D_w3


	#output_data = {'wall_times': wall_times, 'trainerror': trainerror}
	return wall_times, func_val#, train_err
######################################################################################################
def grad_descent(num_iter, init_x, stepsize, batch_size, data_holder):
	num_examples = data_holder.num_train_examples
	curr_x = init_x
	
	for curr_iter in range(num_iter):
		curr_grad = data_holder.logistic_batch_grad(random.sample(range(num_examples), batch_size), curr_x)
		curr_x = curr_x - stepsize*curr_grad

	return curr_x

#######################################################################################
m = 10
#定义一个随机梯度下降算法
def grand(w, data_holder,item, le):
	num_examples = data_holder.num_train_examples
	dir_1 = np.zeros((m , 785))
	for i in range(item):
		batch = random.sample(range(num_examples), 128)
		ave_D_w1 = data_holder.batch_grand(batch, w)
		ave_D_w1 = ave_D_w1.reshape((m, 785))

		w = w - le * (ave_D_w1 + 0.9 *dir_1)
		dir_1 = ave_D_w1

	return w






#一层神经网络K-FAC算法主程序
def K_FAC(w1, lam, ite, data_holder, eta, omega, lam_max):
	num_examples = data_holder.num_train_examples
	#dir_1 = np.zeros(m * 785)


	wall_times = []
	func_val = []
	# train_err = []
	start_time = time.time()
	dir_1 = data_holder.batch_grand(range(0, 128), w1)
	w1 = w1 - 0.01 * dir_1.reshape((m, 785))
	#dir_1 = np.ones(m*785)
	for curr_iter in range(ite):
		print('dir_1',dir_1)
		#dir_1 = dir_1.reshape((m, 785))
		wall_times += [time.time() - start_time]
		func_val += [data_holder.ave_L_x_i(range(0, num_examples), w1)]
		# train_err += [data_holder.error_01(w1, w2, w3, mode='TRAIN')]
		batch = random.sample(range(num_examples), 128)
		ave_D_w1 = data_holder.batch_grand(batch, w1)
		A , G = data_holder.A_G(batch, w1)
		#print("data_holder.trace(G)",data_holder.trace(G))
		#print('np.trace(G)',np.trace(G))
		#PI = ((data_holder.trace(A) / 786) / ((data_holder.trace(G)) / m)) ** 0.5
		#print('PI',PI)
		F_inv = np.linalg.inv(np.kron(A, G) + (lam + eta)* np.identity(m*785))
		dir = -1 *np.dot(F_inv, ave_D_w1)
		dir_1 = dir_1.reshape((785 * m), order='F')
		alpha , miu = data_holder.alpha_miu_(A, G, lam, eta, ave_D_w1, dir, dir_1)
		print("alpha, miu", alpha, miu)
		#alpha = -1 * (np.dot(ave_D_w1, dir))/(np.dot(dir, np.dot(np.kron(A, G)+(lam +eta)*np.identity(m*785),dir)))
		#print('alpha,miu' , alpha, miu)
		#更新方向
		dir_1 = alpha * dir + miu * dir_1
		pred = (0.5) * np.dot(dir_1, np.dot(np.kron(A, G) + (lam + eta) * np.identity(m * 785), dir_1)) + np.dot(
			ave_D_w1, dir_1)
		dir_1 = dir_1.reshape((m, 785))
		# 计算rou，进而更新lam的值
		f_jiu = data_holder.ave_L_x_i(range(0, num_examples), w1)
		w1 = w1 + dir_1

		f_xin = data_holder.ave_L_x_i(range(0, num_examples), w1)
		#f_jiu = data_holder.ave_L_x_i(range(0, num_examples), w1)
		#pred = (0.5 ) * np.dot(dir_1, np.dot(np.kron(A, G) + (lam + eta)* np.identity(m * 785), dir_1)) +  np.dot(ave_D_w1, dir_1)
		rou = (f_xin - f_jiu)/pred
		print("rou", rou)
		if rou > 0.75:
			lam = omega * lam
		elif rou < 0.25:
			if lam/omega < lam_max:
				lam = lam/omega
			else:
				lam = lam_max
		else:
			lam = lam
		print('lam',lam)

	# output_data = {'wall_times': wall_times, 'trainerror': trainerror}
	return wall_times, func_val  # , train_err
############################
#信赖域方法
mini_size = 128
def K_FAC_cluster(w1, lam, ite, data_holder, delta, delta_max, gama, eta, omega, lam_max):
	num_examples = data_holder.num_train_examples
	#dir_1 = np.zeros(m * 785)


	wall_times = []
	func_val = []
	# train_err = []
	dir_1 = data_holder.batch_grand(range(0, 128), w1)
	dir_1 = dir_1.reshape((m, 785))

	w1 = w1 - dir_1
	start_time = time.time()
	#dir_1 = np.ones(m*785)
	i = 0
	for curr_iter in range(ite):
		print('dir_1',dir_1)
		#dir_1 = dir_1.reshape((m, 785))
		wall_times += [time.time() - start_time]
		func_val += [data_holder.ave_L_x_i(range(0, num_examples), w1)]
		# train_err += [data_holder.error_01(w1, w2, w3, mode='TRAIN')]
		if i  ==0:
			my_grand_list = []
			my_A_list = []
			my_G_list = []
			batch = range(0, num_examples)
			for j in range(num_examples):
				my_grand_list[j] = data_holder.grand(j, w1)
				A, G = data_holder.A_G_sample(j, w1)
				my_A_list[j] = A
				my_G_list[j] = G
		else:
			batch = range(i, i + mini_size)
		print("my_dict_grand",my_dict_grand["0"])

		ave_D_w1 = data_holder.batch_grand(batch, w1)
		A , G = data_holder.A_G(batch, w1)
		#计算聚合梯度
		grand = np.zeros(m*785)
		for j in range(num_examples):
			if i <= j <= i+mini_size:
				grand = grand + np.zeros(m*785)
			else:
				grand = grand + my_dict_grand["j"]
		grand = (grand + 128 * ave_D_w1)/num_examples

		#计算聚合矩阵A
		A_ = np.zeros((785,785))
		for j in range(num_examples):
			if i <= j <= i + mini_size:
				A_ = A_ + np.zeros((785, 785))
			else:
				A_ = A_ + my_dict_A["j"]
		A_ = (A_ + mini_siae * A) / num_examples

		#计算聚合矩阵G
		G_ = np.zeros((m, m))
		for j in range(num_examples):
			if i <= j <= i + mini_size:
				G_ = G_ + np.zeros((m, m))
			else:
				G_ = G_ + my_dict_G["j"]
		G_ = (G_ + mini_siae * G) / num_examples

		#准备阶段完成，下一步开始计算更新方向
		#PI = ((data_holder.trace(A_) / 786) / (data_holder.trace(G_) / m)) ** 0.5
		F_inv = np.linalg.inv(np.kron(A_, G_) + (lam + eta)* np.identity(m*785))
		dir = -1 *np.dot(F_inv, grand)
		dir_1 = dir_1.reshape((785 * m), order='F')
		#len = np.dot(dir_1, dir_1)
		alpha , miu = data_holder.alpha_miu_(A_, G_, lam, eta, grand, dir, dir_1)
		# alpha = -1 * (np.dot(ave_D_w1, dir))/(np.dot(dir, np.dot(np.kron(A, G)+(lam +eta)*np.identity(m*785),dir)))
		# print('alpha,miu' , alpha, miu)
		# 更新方向
		dir_1 = alpha * dir + miu * dir_1
		len = np.dot(dir_1, dir_1)
		#dir_1 = dir_1.reshape((m, 785))
		if len <= delta:
			dir_1 = dir_1
		else:
			dir_1 = (delta/len)* dir_1
		# 计算rou，进而更新lam的值
		pred = (0.5) * np.dot(dir_1, np.dot(np.kron(A_,G_) + (lam + eta) * np.identity(m* 785),
											dir_1)) + np.dot(grand, dir_1)
		f_jiu = data_holder.ave_L_x_i(range(0, num_examples), w1)
		dir_1 = dir_1.reshape((m, 785))
		#w1 = w1 + dir_1

		f_xin = data_holder.ave_L_x_i(range(0, num_examples), w1 + dir_1)
		# f_jiu = data_holder.ave_L_x_i(range(0, num_examples), w1)
		#pred = (0.5) * np.dot(dir_1, np.dot(np.kron(A, G) + PI* (lam **0.5) * np.identity(m * 785),
													   #dir)) + np.dot(grand, dir_1)
		rou = (f_xin - f_jiu) / pred
		print("rou", rou)
		if rou > 0.75 and len**0.5 == delta:
			if 2* delta < delta_max:
				delta = 2 * delta
			else:
				delta = delta_max
		elif rou < 0.25:
			delta = 0.25 * delta
		else:
			delta = delta
		print('delta', delta)
		w1 = w1 + dir_1

		#更新lam的值
		if rou > 0.75:
			lam = omega * lam
		elif rou < 0.25:
			if lam/omega < lam_max:
				lam = lam/omega
			else:
				lam = lam_max
		else:
			lam = lam
		print('lam', lam)

		#更新i的取值
		if i < num:
			i = i + mini_siae
		else:
			i = 0
		print("i", i)

	# output_data = {'wall_times': wall_times, 'trainerror': trainerror}
	return wall_times, func_val  # , train_err

mini_size = 128
def K_FAC_TR(w1, lam, ite, data_holder, delta, delta_max, gama, eta, omega, lam_max, delta_min):
	num_examples = data_holder.num_train_examples
	#dir_1 = np.zeros(m * 785)


	wall_times = []
	func_val = []
	# train_err = []

	my_grand_list = [0]*num_examples
	#my_A_list = [0]*num_examples
	my_G_list = [0]*num_examples
	for j in range(num_examples):
		#my_grand_list.insert(j,data_holder.grand(j, w1))
		#print("my_dict_grand", my_grand_list[0])
		my_grand_list[j] = data_holder.grand(j, w1)

	#print("my_dict_grand", my_grand_list[0].shape)
	A_ = np.zeros((785, 785))
	for j in range(num_examples):
		A, G = data_holder.A_G_sample(j, w1)
		my_G_list[j] = G
		A_ = A_ + A
	A_ = A_ / num_examples


	print("my_G_list", my_G_list[0].shape)
	print("my_dict_grand", my_grand_list[77].shape)
	dir_1 = -1 * data_holder.batch_grand(range(0, 128), w1)
	w1 = w1 + 0.01 * dir_1.reshape((m, 785))
	grand_0 = -1 * dir_1

	start_time = time.time()
	#dir_1 = np.ones(m*785)
	i = 0


	for curr_iter in range(ite):
		print('grand_0', grand_0)
		#dir_1 = dir_1.reshape((m, 785))
		wall_times += [time.time() - start_time]
		func_val += [data_holder.ave_L_x_i(range(0, num_examples), w1)]
		# train_err += [data_holder.error_01(w1, w2, w3, mode='TRAIN')]

		batch = range(i, i + 128)
		print("batch",batch)

		#ave_D_w1 = data_holder.batch_grand(batch, w1)
		#A , G = data_holder.A_G(batch, w1)
		#计算聚合梯度

		for j in batch:
			my_grand_list[j] = data_holder.grand(j, w1)

		grand = np.zeros(m * 785)
		for k in range(num_examples):
			#print('my_dict_grand[i]',my_dict_grand["1"])
			grand = grand + my_grand_list[k]

		grand = grand/num_examples

		#计算聚合矩阵A

		for j in batch:
			A_, G_ = data_holder.A_G_sample(j, w1)
			my_G_list[j] = G_


		#计算聚合矩阵G
		G_ = np.zeros((m, m))
		for l in range(num_examples):
			G_ = G_ + my_G_list[l]

		G_ = G_ /num_examples

		#batch = random.sample(range(i, i + 128), 128)
		#grand = data_holder.batch_grand(batch, w1)
		#A_ , G_ = data_holder.A_G(batch, w1)


		#准备阶段完成，下一步开始计算更新方向
		#PI = ((data_holder.trace(A_) / 786) / (data_holder.trace(G_) / m)) ** 0.5
		F_inv = np.linalg.inv(np.kron(A_, G_) + (lam + eta) * np.identity(m*785))
		dir = -1 * np.dot(F_inv, grand)
		dir_1 = dir_1.reshape((785 * m), order='F')
		#len = np.dot(dir_1, dir_1)
		#alpha, miu = data_holder.alpha_miu_(A_, G_, lam, eta, grand, dir, dir_1)
		alpha = -1* np.dot(grand, dir)/(np.dot(dir, np.dot(np.kron(A_, G_), dir)) + (lam + eta)* np.dot(dir, dir))

		print('alpha' , alpha)
		# 更新方向
		dir_1 = alpha * dir + 0.9 * dir_1
		len = np.dot(dir_1, dir_1)
		print("len", len)

		if len <= delta:
			dir_1 = dir_1
		else:
			dir_1 = (delta/len) * dir_1
		# 计算rou，进而更新lam的值
		grand = grand.reshape((m*785), order='F')
		#dir_1 = dir_1 - 0.001 * grand_0

		pred = (0.5) * np.dot(dir_1, np.dot(np.kron(A_, G_) + (lam + eta) * np.identity(m * 785),
											dir_1)) + np.dot(grand, dir_1)
		f_jiu = data_holder.ave_L_x_i(range(0, num_examples), w1)
		dir_1 = dir_1.reshape((m, 785))

		w1 = w1 + dir_1
		f_xin = data_holder.ave_L_x_i(range(0, num_examples), w1)

		rou = (f_xin - f_jiu) / pred
		print("rou", rou)
		if rou > 0.75 and len**0.5 == delta:
			if 2 * delta < delta_max:
				delta = 2 * delta
			else:
				delta = delta_max
		elif rou <= 0.25:
			if 0.25 * delta >=delta_min:
				delta = 0.25 * delta
			else:
				delta = delta_min
		else:
			delta = delta
		print('delta', delta)

		#grand = grand.reshape((m, 785))
		#grand_0 = grand_0.reshape((m, 785))


		#更新lam的值
		if rou > 0.75:
			lam = omega * lam
		elif rou < 0.25:
			if lam/omega < lam_max:
				lam = lam/omega
			else:
				lam = lam_max
		else:
			lam = lam
		print('lam', lam)

		#grand_0 = grand


		i = i + 128

		print("i", i)




	# output_data = {'wall_times': wall_times, 'trainerror': trainerror}
	return wall_times, func_val  # , train_err
#########################################################################
def K_FAC_XIN(w1,w2, w3, w4, lam, ite, data_holder, eta, omega, lam_max):
	num_examples = data_holder.num_train_examples
	#dir_1 = np.zeros(m * 785)


	wall_times = []
	func_val = []
	# train_err = []
	start_time = time.time()
	D_w1, D_w2, D_w3, D_w4, A00, A11, A22, A33, G11, G22, G33, G44 = \
		data_holder.g_A_G_batch(random.sample(range(num_examples), 1), w1, w2, w3, w4, mode='TRAIN')
	PI_1 = ((data_holder.trace(A00) / 786) / ((data_holder.trace(G11)) / (h1 + 1))) ** 0.5
	PI_2 = ((data_holder.trace(A11) / (h1 + 2)) / ((data_holder.trace(G22)) / (h2 + 1))) ** 0.5
	PI_3 = ((data_holder.trace(A22) / (h2 + 2)) / ((data_holder.trace(G22)) / (h3 + 1))) ** 0.5
	PI_4 = ((data_holder.trace(A33) / (h3 + 2)) / (G44 / 2)) ** 0.5
	print("D_w1.shape", D_w1.shape)
	dir_1_0 = -1 * data_holder.matri_mu(np.linalg.inv(G11 + ((lam + eta)**0.5/PI_1)* np.identity(h1)),
								 data_holder.matri_mu(D_w1, np.linalg.inv(
									 A00 + ((lam + eta)**0.5/PI_1)* np.identity(785))))
	dir_2_0 = -1 * data_holder.matri_mu(np.linalg.inv(G22 + ((lam + eta) ** 0.5 / PI_2) * np.identity(h2)),
								 data_holder.matri_mu(D_w2, np.linalg.inv(
									 A11 + ((lam + eta) ** 0.5 / PI_2) * np.identity(h1 + 1))))
	dir_3_0 = -1 * data_holder.matri_mu(np.linalg.inv(G33 + ((lam + eta) ** 0.5 / PI_3) * np.identity(h3)),
								 data_holder.matri_mu(D_w3, np.linalg.inv(
									 A22 + ((lam + eta) ** 0.5 / PI_3) * np.identity(h2 + 1))))
	dir_4_0 = -1 / (G44 + ((lam + eta) ** 0.5 / PI_4)) * data_holder.mu_(np.linalg.inv(
									 A33 + ((lam + eta) ** 0.5 / PI_4) * np.identity(h3 + 1)), w4)
	w1 = w1 + dir_1_0
	w2 = w2 + dir_2_0
	w3 = w3 + dir_3_0
	print("dir_4_0", dir_4_0)
	w4 = w4 + dir_4_0

	for curr_iter in range(ite):

		#dir_1 = dir_1.reshape((m, 785))
		wall_times += [time.time() - start_time]
		jiu_fun = data_holder.loss_fun_ave(range(0, num_examples), w1, w2, w3, w4,  mode='TRAIN')
		func_val += [jiu_fun]
		# train_err += [data_holder.error_01(w1, w2, w3, mode='TRAIN')]
		batch = random.sample(range(num_examples), 1)
		D_w1, D_w2, D_w3, D_w4, A00, A11, A22, A33, G11, G22, G33, G44 = \
			data_holder.g_A_G_batch(batch, w1, w2, w3, w4, mode='TRAIN')
		PI_1 = ((data_holder.trace(A00) / 786) / ((data_holder.trace(G11)) / (h1 + 1))) ** 0.5
		PI_2 = ((data_holder.trace(A11) / (h1 + 2)) / ((data_holder.trace(G22)) / (h2 + 1))) ** 0.5
		PI_3 = ((data_holder.trace(A22) / (h2 + 2)) / ((data_holder.trace(G22)) / (h3 + 1))) ** 0.5
		PI_4 = ((data_holder.trace(A33) / (h3 + 2)) / (G44 / 2)) ** 0.5
		dir_1 = -1 * data_holder.matri_mu(np.linalg.inv(G11 + ((lam + eta) ** 0.5 / PI_1) * np.identity(h1)),
										  data_holder.matri_mu(D_w1, np.linalg.inv(
											  A00 + ((lam + eta) ** 0.5 / PI_1) * np.identity(785))))
		dir_2 = -1 * data_holder.matri_mu(np.linalg.inv(G22 + ((lam + eta) ** 0.5 / PI_2) * np.identity(h2)),
										  data_holder.matri_mu(D_w2, np.linalg.inv(
											  A11 + ((lam + eta) ** 0.5 / PI_2) * np.identity(h1 + 1))))
		dir_3 = -1 * data_holder.matri_mu(np.linalg.inv(G33 + ((lam + eta) ** 0.5 / PI_3) * np.identity(h3)),
										  data_holder.matri_mu(D_w3, np.linalg.inv(
											  A22 + ((lam + eta) ** 0.5 / PI_3) * np.identity(h2 + 1))))
		print("G44", G44)
		dir_4 = -1 / (G44 + ((lam + eta) ** 0.5 / PI_4)) * data_holder.mu_(np.linalg.inv(
									 A33 + ((lam + eta) ** 0.5 / PI_4) * np.identity(h3 + 1)), w4)
		###############################
		#计算alpha，miu
		##############################
		F1 = np.kron(A00 + ((lam + eta) ** 0.5 / PI_1) * np.identity(785),
					 G11 + ((lam + eta) ** 0.5 / PI_1) * np.identity(h1))
		F2 = np.kron(A11 + ((lam + eta) ** 0.5 / PI_2) * np.identity(h1 +1),
					 G22 + ((lam + eta) ** 0.5 / PI_2) * np.identity(h2))
		F3 = np.kron(A22 + ((lam + eta) ** 0.5 / PI_3) * np.identity(h2 + 1),
					 G33 + ((lam + eta) ** 0.5 / PI_3) * np.identity(h3))
		F4 = (G44 + (lam + eta) ** 0.5 / PI_4)* (A33 + ((lam + eta) ** 0.5 / PI_4) * np.identity(h3 + 1))
		dir_1_0 = dir_1_0.reshape((785 * h1), order='F')
		dir_1 = dir_1.reshape((785 * h1), order='F')
		dir_2_0 = dir_2_0.reshape(((h1+1) * h2), order='F')
		dir_2 = dir_2.reshape(((h1 + 1) * h2), order='F')
		dir_3_0 = dir_3_0.reshape(((h2 + 1) * h3), order='F')
		dir_3 = dir_3.reshape(((h2 + 1) * h3), order='F')
		dir_4_0 = dir_4_0.reshape((h3 + 1), order='F')
		#dir_4 = dir_4.reshape((h3 + 1), order='F')
		D_w1 = D_w1.reshape((785 * h1), order='F')
		D_w2 = D_w2.reshape(((h1+1) * h2), order='F')
		D_w3 = D_w3.reshape(((h2 + 1) * h3), order='F')
		#D_w4 = D_w4.reshape((h3 + 1), order='F')
		a = np.dot(dir_1, np.dot(F1, dir_1)) +np.dot(dir_2, np.dot(F2, dir_2)) \
			+np.dot(dir_3, np.dot(F3, dir_3)) + np.dot(dir_4, np.dot(F4, dir_4))
		b = np.dot(dir_1, np.dot(F1, dir_1_0)) + np.dot(dir_2, np.dot(F2, dir_2_0)) \
			+ np.dot(dir_3, np.dot(F3, dir_3_0)) + np.dot(dir_4, np.dot(F4, dir_4_0))
		c = np.dot(dir_1_0, np.dot(F1, dir_1_0)) + np.dot(dir_2_0, np.dot(F2, dir_2_0)) \
			+ np.dot(dir_3_0, np.dot(F3, dir_3_0)) + np.dot(dir_4_0, np.dot(F4, dir_4_0))
		d = -1 * (np.dot(D_w1, dir_1)+np.dot(D_w2, dir_2)+np.dot(D_w3, dir_3)+np.dot(D_w4, dir_4))
		e = -1 * (np.dot(D_w1, dir_1_0)+np.dot(D_w2, dir_2_0)+np.dot(D_w3, dir_3_0)+np.dot(D_w4, dir_4_0))
		alpha = (b * e - c * d) / (a * c - b * b)
		miu = (b * d - a * e) / (a * c - b * b)
		print("alpha, miu", alpha, miu)
		dir_1 = alpha * dir_1 + miu * dir_1_0
		dir_2 = alpha * dir_2 + miu * dir_2_0
		dir_3 = alpha * dir_3 + miu * dir_3_0
		dir_4 = alpha * dir_4 + miu * dir_4_0
		#计算rou
		dir_1 = alpha * dir + miu * dir_1
		pred = (0.5) * np.dot(dir_1, np.dot(F1, dir_1)) + (0.5) * np.dot(dir_2, np.dot(F2, dir_2))
		+ (0.5) * np.dot(dir_3, np.dot(F3, dir_3)) + (0.5) * np.dot(dir_4, np.dot(F4, dir_4))
		+ np.dot(D_w1, dir_1) + np.dot(D_w2, dir_2) + np.dot(D_w3, dir_3) + np.dot(D_w4, dir_4)


		dir_1 = dir_1.reshape((h1, 785))
		dir_2 = dir_2.reshape((h2, h1 + 1))
		dir_3 = dir_3.reshape((h3, h2 + 1))
		dir_4 = dir_4.reshape((1, h3 + 1))

		w1 = w1 + dir_1
		w2 = w2 + dir_2
		w3 = w3 + dir_3
		w4 = w4 + dir_4

		xin_fun = data_holder.loss_fun_ave(range(0, num_examples), w1, w2, w3, w4,  mode='TRAIN')

		rou = (xin_fun - jiu_fun)/pred
		print("rou", rou)
		if rou > 0.75:
			lam = omega * lam
		elif rou < 0.25:
			if lam/omega < lam_max:
				lam = lam/omega
			else:
				lam = lam_max
		else:
			lam = lam
		print('lam',lam)

	# output_data = {'wall_times': wall_times, 'trainerror': trainerror}
	return wall_times, func_val  # , train_err

