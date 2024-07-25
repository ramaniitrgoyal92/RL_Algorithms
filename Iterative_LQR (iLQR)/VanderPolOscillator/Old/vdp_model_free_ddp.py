'''
Model free DDP method for the Van der Pol Oscillator.
'''
#!/usr/bin/env python

import numpy as np
import time
import os
import platform
import math
import json
import matplotlib.pyplot as plt
from pathlib import Path
from POM_model_free_DDP import DDP
from arma_ltv_sys_id import ltv_sys_id_class
from vdp_params import *

class model_free_material_DDP(DDP, ltv_sys_id_class):
	
	def __init__(self, initial_state, final_state, MODEL_XML, alpha, horizon, state_dimension, control_dimension, Q, Q_final, R):
		
		'''
			Declare the matrices associated with the cost function
		'''
		self.Q = Q
		self.Q_final = Q_final
		self.R = R


		DDP.__init__(self, None, state_dimension, control_dimension, alpha, horizon, initial_state, final_state)
		ltv_sys_id_class.__init__(self, None, state_dimension, control_dimension, n_substeps, n_samples=feedback_n_samples)

	def state_output(self, state):
		'''
			Given a state in terms of Mujoco's MjSimState format, extract the state as a numpy array, after some preprocessing
		'''
		
		return state#np.concatenate([state.qpos, state.qvel]).reshape(-1, 1)


	def cost(self, x, u):
		'''
			Incremental cost in terms of state and controls
		'''
		return (((x - self.X_g).T @ self.Q) @ (x - self.X_g)) + (((u.T) @ self.R) @ u)
	
	def cost_final(self, x):
		'''
			Cost in terms of state at the terminal time-step
		'''
		return (((x - self.X_g).T @ self.Q_final) @ (x - self.X_g)) 

	def initialize_traj(self, path=None, init=None):#Add check for replan, and add an input for U_0 as initial guess
		'''
		Initial guess for the nominal trajectory by default is produced by zero controls
		'''
		if path is None:
			
			for t in range(0, self.N):
				#print(np.shape(np.random.normal(0, nominal_init_stddev, (self.n_u, 1))))
				if init is None:
					self.U_p[t, :] = np.random.normal(0, nominal_init_stddev, (self.n_u, 1))	#np.random.normal(0, 0.01, self.n_u).reshape(self.n_u,1)#DM(array[t, 4:6])
				else:
					self.U_p[t, :] = init[t,:]

			np.copyto(self.U_p_temp, self.U_p)
			
			self.forward_pass_sim()
			
			np.copyto(self.X_p_temp, self.X_p)

		else:

			array = np.loadtxt('../rrt_path.txt')
			### INCOMPLETE

	def replan(self, init_state, t1, horizon, ver, cost):

		path_to_model_free_DDP = "/home/raman/POD2C"
		MODEL_XML = path_to_model_free_DDP + "/models/fish_old.xml"
		path_to_exp = path_to_model_free_DDP + "/experiments/Material/feedback_tests/exp_replan"

		path_to_file = path_to_exp + "/material_policy_"+str(ver-1)+".txt"
		path_to_file_re = path_to_exp + "/material_policy_"+str(ver)+".txt"
		training_cost_data_file = path_to_exp + "/training_cost_data.txt"
		path_to_data = path_to_exp + "/material_D2C_DDP_data.txt"

		print(path_to_file)
		
		noise_std_test = 0.1
		n_iterations=5
		alpha = .7
		U_p_rep = np.zeros((horizon, self.n_u, 1))
		xdim = int(math.sqrt(self.n_x))
		check_flag=True
		if horizon==2:
			check_flag=False

		with open(path_to_file) as f:

				Pi = json.load(f)

		for i in range(0, horizon):
			U_p_rep[i, :] = (np.array(Pi['U'][str(i)])).flatten().reshape(np.shape(U_p_rep[i, :]))


		#if ver==1:
		#	print('Umax = '+str(np.max(U_p_rep)))

		#print('Init = '+str(U_p_rep[0,:]))
		# print('Ver = ',ver)
		#print('Init = ',init_state)
		#print('Final = ',self.X_g)
		# print('New horizon =',horizon-1)
		#print('State_dim = ',self.n_x)
		#print('Control_dim = ',self.n_u)
		
		updated_model = model_free_material_DDP(init_state, self.X_g, MODEL_XML, alpha, horizon-1, self.n_x, self.n_u, Q, Q_final, R)
		# print('Check = ', horizon-1)
		updated_model.iterate_ddp(n_iterations, finite_difference_gradients_flag=False, u_init=None)#U_p_rep[t1:, :])
		# print('Check2 = ', horizon-1)
		#print('ctrl = ', updated_model.U_p[0])
		#updated_model.plot_episodic_cost_history(save_to_path=path_to_exp+"/replan_episodic_cost_training_step"+str(ver)+".png")
		updated_model.save_policy(path_to_file_re)
		# print('CheckS = ', horizon-1)

		testCL = updated_model.test_episode(1, path=path_to_file_re, noise_stddev=noise_std_test, check_plan = check_flag, init=init_state, version=ver, cst=cost)
		if check_flag==False:
			print("Ver + "+str(ver)+" Replanned Goal diff = "+str(np.linalg.norm(testCL-final_state)))
			print("Final state = ", testCL.reshape((xdim,xdim)))
			#print('Total rep cost = '+str(cost))


if __name__=="__main__":

	# Path of the model file
	cwd = os.getcwd()
	path_to_model_free_DDP = Path(cwd)/"Iterative_LQR (iLQR)/VanderPolOscillator"
	MODEL_XML = path_to_model_free_DDP/"models/None.xml"

	path_to_exp = path_to_model_free_DDP/"VDP_Experiments/exp_2"
	path_to_file = path_to_exp / "vdp_policy_0.txt"
	training_cost_data_file = path_to_exp / "training_cost_data.txt"
	path_to_data = path_to_exp / "vdp_D2C_DDP_data.txt"

	# Declare other parameters associated with the problem statement
	
	# alpha is the line search parameter during forward pass
	alpha = .7

	# Declare the initial state and the final state in the problem


	initial_state = np.zeros((state_dimension,1))
	initial_state[0] = 2
	
	final_state = np.zeros((state_dimension, 1))

	print('Initial phase : \n', initial_state)
	print('Goal phase : \n', final_state)
	# No. of ILQR iterations to run
	n_iterations = 60#5#40

	# Initiate the above class that contains objects specific to this problem
	model = model_free_material_DDP(initial_state, final_state, MODEL_XML, alpha, horizon, state_dimension, control_dimension, Q, Q_final, R)


	# ---------------------------------------------------------------------------------------------------------
	# -----------------------------------------------Training---------------------------------------------------
	# ---------------------------------------------------------------------------------------------------------
	# Train the policy

	training_flag_on = True
	if training_flag_on:

		with open(path_to_data, 'w') as f:

			f.write("D2C training performed for the required control task:\n\n")

			# f.write("System details : {}\n".format(os.uname().sysname + "--" + os.uname().nodename + "--" + os.uname().release + "--" + os.uname().version + "--" + os.uname().machine))
			# f.write("System details : {}\n".format(
			# 	platform.uname().sysname + "--" + platform.uname().nodename + "--" + platform.uname().release + "--" + platform.uname().version + "--" + platform.uname().machine))
			f.write("-------------------------------------------------------------\n")

		time_1 = time.time()

		# Run the DDP algorithm
		# To run using our LLS-CD jacobian estimation (faster), make 'finite_difference_gradients_flag = False'
		# To run using forward difference for jacobian estimation (slower), make 'finite_difference_gradients_flag = True'

		model.iterate_ddp(n_iterations, finite_difference_gradients_flag=False)
		
		time_2 = time.time()

		D2C_algorithm_run_time = time_2 - time_1

		print("D2C-2 algorithm run time taken: ", D2C_algorithm_run_time)

		# Save the history of episodic costs 
		with open(training_cost_data_file, 'w') as f:
			for cost in model.episodic_cost_history:
				f.write("%s\n" % cost)

		# Test the obtained policy
		model.save_policy(path_to_file)
		#U_nominal = U_p
		#X_nominal = X_p
		#np.save('X_nominal.npy', X_nominal)

		with open(path_to_data, 'a') as f:

				f.write("\nTotal time taken: {}\n".format(D2C_algorithm_run_time))
				f.write("------------------------------------------------------------------------------------------------------------------------------------\n")

		# Display the final state in the deterministic policy on a noiseless system
		print('Final State : ', model.X_p[-1])
		print('control = ', model.U_p)
		np.save('simulator/control_p1.npy', model.U_p)
		# Plot the episodic cost during the training
		model.plot_episodic_cost_history(save_to_path=path_to_exp+"/episodic_cost_training.png")


		# state_history_nominal=model.X_p[-1]
		#for t_time in range(1):#range(np.shape(state_history_nominal)[0]):
		#phi=state_history_nominal[t_time,:].reshape([20, 20])

		''''# UPDATING TO NEW TRAJECTORY
		# n_iterations = 50
		initial_state = model.X_p[-1]
		final_state = 5*np.ones((state_dimension, 1))
		final_state[1, :] = -1*final_state[1, :]

		model = model_free_material_DDP(initial_state, final_state, MODEL_XML, alpha, horizon, state_dimension,
										control_dimension, Q, Q_final, R)

		time_1 = time.time()
		print('Starting new run')
		# Run the DDP algorithm
		# To run using our LLS-CD jacobian estimation (faster), make 'finite_difference_gradients_flag = False'
		# To run using forward difference for jacobian estimation (slower), make 'finite_difference_gradients_flag = True'

		model.iterate_ddp(n_iterations, finite_difference_gradients_flag=False)

		time_2 = time.time()

		D2C_algorithm_run_time = time_2 - time_1

		print("D2C-2 algorithm run time taken: ", D2C_algorithm_run_time)

		print('Final State : ', model.X_p[-1])
		np.save('simulator/control_p2.npy', model.U_p)
		# Plot the episodic cost during the training
		model.plot_episodic_cost_history(save_to_path=path_to_exp + "/episodic_cost_training_2.png")

		# UPDATING TO NEW TRAJECTORY

		initial_state = model.X_p[-1]
		final_state = -2 * np.ones((state_dimension, 1))
		final_state[1, :] = 0 * final_state[1, :]

		model = model_free_material_DDP(initial_state, final_state, MODEL_XML, alpha, horizon, state_dimension,
											control_dimension, Q, Q_final, R)

		time_1 = time.time()
		print('Starting new run')
		# Run the DDP algorithm
		# To run using our LLS-CD jacobian estimation (faster), make 'finite_difference_gradients_flag = False'
		# To run using forward difference for jacobian estimation (slower), make 'finite_difference_gradients_flag = True'

		model.iterate_ddp(n_iterations, finite_difference_gradients_flag=False)

		time_2 = time.time()

		D2C_algorithm_run_time = time_2 - time_1

		print("D2C-2 algorithm run time taken: ", D2C_algorithm_run_time)

		print('Final State : ', model.X_p[-1])
		np.save('simulator/control_p3.npy', model.U_p)
		# Plot the episodic cost during the training
		model.plot_episodic_cost_history(save_to_path=path_to_exp + "/episodic_cost_training_3.png")

		initial_state = model.X_p[-1]
		final_state = np.zeros((state_dimension, 1))

		model = model_free_material_DDP(initial_state, final_state, MODEL_XML, alpha, horizon, state_dimension,
										control_dimension, Q, Q_final, R)

		time_1 = time.time()
		print('Starting new run')
		# Run the DDP algorithm
		# To run using our LLS-CD jacobian estimation (faster), make 'finite_difference_gradients_flag = False'
		# To run using forward difference for jacobian estimation (slower), make 'finite_difference_gradients_flag = True'

		model.iterate_ddp(n_iterations, finite_difference_gradients_flag=False)

		time_2 = time.time()

		D2C_algorithm_run_time = time_2 - time_1

		print("D2C-2 algorithm run time taken: ", D2C_algorithm_run_time)

		print('Final State : ', model.X_p[-1])
		np.save('simulator/control_p4.npy', model.U_p)
		# Plot the episodic cost during the training
		model.plot_episodic_cost_history(save_to_path=path_to_exp + "/episodic_cost_training_4.png")'''

	# ---------------------------------------------------------------------------------------------------------
	# -----------------------------------------------Testing---------------------------------------------------
	# ---------------------------------------------------------------------------------------------------------
	# Test the obtained policy

	test_flag_on = False
	#np.random.seed(1)

	if test_flag_on:

		f = open(path_to_exp + "/material_testing_data.txt", "a")

		def frange(start, stop, step):
			i = start
			while i < stop:
				yield i
				i += step
		
		u_max = 3.5

		try:

			for i in frange(0.0, 1.02, 0.02):

				print(i)
				print("\n")
				terminal_mse = 0
				Var_terminal_mse = 0
				n_samples = 100

				for j in range(n_samples):	

					terminal_state = model.test_episode(render=0, path=path_to_file, noise_stddev=i*u_max)
					terminal_mse += np.linalg.norm(terminal_state[0:3] - final_state[0:3], axis=0)
					Var_terminal_mse += (np.linalg.norm(terminal_state[0:3] - final_state[0:3], axis=0))**2

				terminal_mse_avg = terminal_mse/n_samples
				Var_terminal_mse_avg = (1/(n_samples*(n_samples-1)))*(n_samples*Var_terminal_mse - terminal_mse**2)

				std_dev_mse = np.sqrt(Var_terminal_mse_avg)

				f.write(str(i)+",\t"+str(terminal_mse_avg[0])+",\t"+str(std_dev_mse[0])+"\n")
		except:

			print("Testing failed!")
			f.close()

		f.close()

	xdim=int(math.sqrt(state_dimension))

	# feedback_costs = np.array([])
	# # replan_costs = np.array([])
	# np.save('std8_noise_nfb.npy',feedback_costs)
	# np.save('replan_costs_std8_noise.npy',replan_costs)

	# end_disc = np.array([])
	# np.save('end_disc.npy',end_disc)
	# for i in range(50):
	# 	testCL = model.test_episode(1, path=path_to_file, noise_stddev=noise_std_test, check_plan=False, init=initial_state)
	# 	print("Goal diff = "+str(np.linalg.norm(testCL-final_state)))
	# AA=np.load('end_disc.npy')
	# np.save('end_disc.npy',np.append(AA, np.linalg.norm(testCL-final_state)))
	# # plt.matshow(testCL.reshape((sdim, sdim)))
	#plt.show()
	# for i in range(1):
	# 	testCL2 = model.test_episode(1, path=path_to_file, noise_stddev=noise_std_test, check_plan=True, init=initial_state)
	#
	#print("Goal diff replanned = "+str(np.linalg.norm(testCL2-final_state)))
