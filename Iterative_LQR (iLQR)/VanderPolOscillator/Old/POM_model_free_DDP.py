'''
Python class for model free DDP method.

ASSUMPTIONS :

1) Costs are quadratic functions
2) Default is set to ILQR - by dropping the second order terms of dynamics.

'''
#!/usr/bin/env python

from __future__ import division

# Numerics
import numpy as np
import time
import requests
import sys
import json
# import docker
import math# Parameters
import matplotlib.pyplot as plt
#import params

from vanderpol import *
from arma_ltv_sys_id import ltv_sys_id_class
from progressbar import *
from vdp_params import q, q_u, obs_dimension
# from params.material_params import dt_ch, n_ch, n_ac


class DDP(object):

	def __init__(self, MODEL_XML, n_x, n_u, alpha, horizon, initial_state, final_state, url=None):

		self.X_p_0 = initial_state
		self.X_g   = final_state

		widgets = [Percentage(), '   ', ETA(), ' (', Timer(), ')']
		self.pbar = ProgressBar(widgets=widgets)

		self.n_x = n_x
		self.n_u = n_u
		self.n_z = obs_dimension
		self.N = horizon

		self.alpha = alpha

		
		# Define nominal state trajectory
		self.X_p = np.zeros((self.N, self.n_x, 1))
		self.X_p_temp = np.zeros((self.N, self.n_x, 1))
		self.X_t = np.zeros((100, self.n_x, 1))  #ADDED FOR COST COMP

		# Define nominal control trajectory
		self.U_p  = np.zeros((self.N, self.n_u, 1))
		self.U_p_temp = np.zeros((self.N, self.n_u, 1))
		self.U_t = np.zeros((100, self.n_u, 1))  #ADDED FOR COST COMP

		# Define sensitivity matrices
		# self.K = np.zeros((self.N, self.n_u, self.n_x))
		self.K = np.zeros((self.N, self.n_u, self.n_z*q+self.n_u*(q_u-1)))
		self.k = np.zeros((self.N, self.n_u, 1))
		
		self.V_zz = np.zeros((self.N, self.n_z*q+self.n_u*(q_u-1), self.n_z*q+self.n_u*(q_u-1)))#np.zeros((self.N, self.n_x, self.n_x))
		self.V_z = np.zeros((self.N, self.n_z*q+self.n_u*(q_u-1), 1))#np.zeros((self.N, self.n_x, 1))

		
		# regularization parameter
		self.mu_min = 1e-3
		self.mu = 1e-3	#10**(-6)
		self.mu_max = 10**(8)
		self.delta_0 = 2
		self.delta = self.delta_0
		
		self.c_1 = -6e-1
		self.count = 0
		self.episodic_cost_history = []
		self.control_cost_history = []


	def iterate_ddp(self, n_iterations, finite_difference_gradients_flag=False, u_init=None): #ADD OPTION TO INPUT INITIAL GUESS IN INIT TRAJ()
		
		'''
			Main function that carries out the algorithm at higher level

		'''
		# Initialize the trajectory with the desired initial guess
		self.initialize_traj(init=None)

		# sys.exit()
		
		for j in self.pbar(range(n_iterations)):				

			b_pass_success_flag, del_J_alpha = self.backward_pass(finite_difference_gradients_flag, activate_second_order_dynamics=0)

			if b_pass_success_flag == 1:

				self.regularization_dec_mu()
				f_pass_success_flag = self.forward_pass(del_J_alpha)

				if not f_pass_success_flag:

					# print("Forward pass doomed")
					i = 2

					while not f_pass_success_flag:

						# print("Forward pass-trying %{}th time".format(i))
						self.alpha = self.alpha*0.99  # simulated annealing
						i += 1
						f_pass_success_flag = self.forward_pass(del_J_alpha)
						# print("alpha = ", self.alpha)

			else:

				self.regularization_inc_mu()
				print("This iteration %{} is doomed".format(j))

			if j < 5:
				self.alpha = self.alpha*0.9
			else:
				self.alpha = self.alpha*0.999
			
			self.episodic_cost_history.append(self.calculate_total_cost(self.X_p_0, self.X_p, self.U_p, self.N)[0][0])
			print('cost = ', self.calculate_total_cost(self.X_p_0, self.X_p, self.U_p, self.N))



	def backward_pass(self, finite_difference_gradients_flag=False, activate_second_order_dynamics=0):

		################## defining local functions & variables for faster access ################

		partials_list = self.partials_list_arma
		k = np.copy(self.k)
		K = np.copy(self.K)
		V_z = np.copy(self.V_z)
		V_zz = np.copy(self.V_zz)

		##########################################################################################
		V_z[self.N-1] = self.l_x_f(self.X_p[self.N-1])	

		np.copyto(V_zz[self.N-1], 2*self.Q_final)

		# Initialize before forward pass
		del_J_alpha = 0

		# TEST CODE
		del_J_alpha, b_pass_success_flag = partials_list(self.X_p_0, self.U_p, V_z, V_zz, activate_second_order_dynamics, finite_difference_gradients_flag, del_J_alpha)

		return b_pass_success_flag, del_J_alpha

	def forward_pass(self, del_J_alpha):

		# Cost before forward pass
		J_1 = self.calculate_total_cost(self.X_p_0, self.X_p.reshape(self.N, obs_dimension), self.U_p, self.N)

		np.copyto(self.X_p_temp, self.X_p)
		np.copyto(self.U_p_temp, self.U_p)

		self.forward_pass_sim()
		
		# Cost after forward pass
		J_2 = self.calculate_total_cost(self.X_p_0,  self.X_p.reshape(self.N, obs_dimension), self.U_p, self.N)

		z = (J_1 - J_2)/del_J_alpha


		if z < self.c_1:

			np.copyto(self.X_p, self.X_p_temp)
			np.copyto(self.U_p, self.U_p_temp)
	
			f_pass_success_flag = 0
			# print("f",z, del_J_alpha, J_1, J_2)

		else:

			f_pass_success_flag = 1

		return f_pass_success_flag

	def partials_list_arma(self, x_0, u_nom, V_z, V_zz, activate_second_order_dynamics, finite_difference_gradients_flag=False, del_J_alpha=0, b_pass_success_flag=1):

		################## defining local functions / variables for faster access ################

		n_x = self.n_x
		n_u = self.n_u
		k = self.k
		K = self.K

		##########################################################################################
		if finite_difference_gradients_flag:

			AB, V_z_F_XU_XU = self.sys_id_FD(x_0, u_nom, central_diff=1, activate_second_order=activate_second_order_dynamics, V_z=V_z)

		else:

			A_aug, B_aug, V_z_F_XU_XU, traj = self.sys_id(x_0, u_nom, central_diff=1, activate_second_order=activate_second_order_dynamics, V_z=V_z)
		

		F_x = np.copy(A_aug)
		F_u = np.copy(B_aug)
		

		####################################################
		
		for t in range(self.N-1, max(q, q_u)-1, -1):

			Q_z = self.l_x(traj[:, t].reshape(self.n_x,1)) + (F_x[t].T @ V_z[t])
			Q_u = self.l_u(u_nom[t]) + ((F_u[t].T) @ V_z[t])
			Q_zz = 2*self.Q + (F_x[t].T @ ((V_zz[t])  @ F_x[t]))
			Q_uz = F_u[t].T @ ((V_zz[t] + self.mu*np.eye(V_zz[t].shape[0])) @ F_x[t])
			Q_uu = 2*self.R + F_u[t].T @ ((V_zz[t] + self.mu*np.eye(V_zz[t].shape[0])) @ F_u[t])

			try:
				# If a matrix cannot be positive-definite, that means it cannot be cholesky decomposed
				np.linalg.cholesky(Q_uu)

			except np.linalg.LinAlgError:
				
				print("FAILED! Q_uu is not Positive definite at t=",t)

				b_pass_success_flag = 0

				# If Q_uu is not positive definite, revert to the earlier values 
				np.copyto(k, self.k)
				np.copyto(K, self.K)
				np.copyto(V_z, self.V_z)
				np.copyto(V_zz, self.V_zz)
				
				break

			else:

				b_pass_success_flag = 1
				
				# update gains as follows
				Q_uu_inv = np.linalg.inv(Q_uu)
				k[t] = -(Q_uu_inv @ Q_u)
				K[t] = -(Q_uu_inv @ Q_uz)

				del_J_alpha += -self.alpha*(k[t].T @ Q_u) - 0.5*self.alpha**2 * (k[t].T @ (Q_uu @ k[t]))
				
				if t>0:

					# V_z[t-1] = Q_z + (K[t].T) @ (Q_uu @ k[t]) + ((K[t].T) @ Q_u) + ((Q_uz.T) @ k[t])
					# V_zz[t-1] = Q_zz + ((K[t].T) @ (Q_uu @ K[t])) + ((K[t].T) @ Q_uz) + ((Q_uz.T) @ K[t])
					V_z[t-1] = Q_z + K[t].T @ (Q_uu @ k[t]) + (K[t].T @ Q_u) + (Q_uz.T @ k[t])
					V_zz[t-1] = Q_zz + (K[t].T @ (Q_uu @ K[t])) + (K[t].T @ Q_uz) + (Q_uz.T @ K[t])


		######################### Update the new gains ##############################################

		np.copyto(self.k, k)
		np.copyto(self.K, K)
		np.copyto(self.V_z, V_z)
		np.copyto(self.V_zz, V_zz)
		
		#############################################################################################

		self.count += 1
		####################################################

		return del_J_alpha, b_pass_success_flag #Q_z, Q_u, Q_zz, Q_uu, Q_uz

	def forward_pass_sim(self, render=0, std=0, pr=False):
		
		################## defining local functions & variables for faster access ################

		#sim = self.sim

		##########################################################################################

		# sim.set_state_from_flattened(np.concatenate([np.array([self.sim.get_state().time]), self.X_p_0.flatten()]))

		ctrl = np.zeros(self.n_u)

		# d_z = np.zeros((obs_dimension*q, 1))
		# d_u = np.zeros((self.n_u*q_u, 1))

		for t in range(0, self.N):

			# sim.forward()

			if t == 0:

				self.U_p[t] = self.U_p_temp[t] + self.alpha * self.k[
					t]  # + np.random.normal(np.zeros(np.shape(self.U_p_temp[t])), std)

			else:

				self.U_p[t] = self.U_p_temp[t] + self.alpha * self.k[t] + (
							self.K[t] @ (self.X_p[t - 1] - self.X_p_temp[t - 1]))

			ctrl[:] = self.U_p[t].flatten()

			if t == 0:
				self.X_p[t] = run_vanderpol(yinit=self.X_p_0.flatten(), tfinal=1, forcing=ctrl, mu=2)[
					-1].reshape(np.shape(self.X_p[0]))

			else:
				self.X_p[t] = run_vanderpol(yinit=self.X_p[t - 1].flatten(), tfinal=1, forcing=ctrl, mu=2)[
					-1].reshape(np.shape(self.X_p[0]))


	def cost(self, state, control):

		raise NotImplementedError()



	def initialize_traj(self):
		# initial guess for the trajectory
		pass



	def test_episode(self, render=0, path=None, noise_stddev=0, printch=False, check_plan=True, init=None, version=0, cst=0):
		
		'''
			Test the episode using the current policy if no path is passed. If a path is mentioned, it simulates the controls from that path
		'''
		ctrl = np.zeros(self.n_u)
		u_net = np.zeros((self.n_u,1))
		u1 = np.zeros((self.n_x,1))
		u2 = np.zeros((self.n_x,1))
		ctrl_seq = np.zeros((self.N, 800))
		dx = 1/self.n_x
		dt = 0.1
		nu = 0.1#0.01/np.pi
		#ctrl_seq_temp = np.zeros((self.N, self.n_u))

		xdim = int(math.sqrt(self.n_x))
		N=xdim
		#cst=np.zeros(self.N)
		
		if path is None:
		
			self.forward_pass_sim(render=1, std=noise_stddev, pr=printch)
					
		else:

			#self.X_p[-1] = np.zeros((self.n_x, 1))#+np.random.normal(np.zeros(np.shape(self.X_p[-1])), noise_stddev)
			CTR = 0
			cost = cst
			costZ=0
			control_cost=np.zeros((self.N,1))
			#print(self.X_p[-1].reshape((10,10)))
			# if init is not None:
			# 	self.X_p[-1] = init
			# 	#self.X_p[0] = init
				#CTR=1

			with open(path) as f:

				Pi = json.load(f)

			for i in range(CTR, self.N):
				
				#self.sim.forward()

				if i!=-1:#==0:
					#ctrl_seq[i, :] = np.array(Pi['U'][str(i)]).reshape(np.shape(ctrl_seq[i]))
					# print(np.shape(u))
					# u = np.array(Pi['U'][str(i)])
					# u1 = u[0:4].reshape((2, 2))
					# u1r = np.repeat(u1, 5, axis=0)
					# u1 = np.repeat(u1r, 5, axis=1).reshape((100,))
					# u2 = u[4:].reshape((2, 2))
					# u2r = np.repeat(u2, 5, axis=0)
					# u2 = np.repeat(u2r, 5, axis=1).reshape((100,))

					# ctrl_seq[i,:100]=u1
					# ctrl_seq[i, 100:]=u2
					ctrl[:] = ((np.array(Pi['U'][str(i)])) + np.random.normal(np.zeros(np.shape(Pi['U'][str(i)])), noise_stddev)).flatten()										
					# ctrl_seq[i,:] = ctrl
					'''elif i==3:#To compensate for reduced horizon, h_orig= 10
					
						ctrl[:] = (np.array(Pi['U'][str(i)]) + \
												np.random.normal(np.zeros(np.shape(Pi['U'][str(i)])), noise_stddev) + \
												np.array(Pi['K'][str(i-1)]) @ (self.X_p[i-1] - np.array(Pi['X'][str(i-1)]))).flatten()'''

				else:
					
					
					ctrl[:] = (np.array(Pi['U'][str(i)]) + \
											np.random.normal(np.zeros(np.shape(Pi['U'][str(i)])), noise_stddev) + \
											np.array(Pi['K'][str(i-1)]) @ (self.X_p[i-1] - np.array(Pi['X'][str(i)]))).flatten()#self.state_output(self.sim.get_state()) - np.array(Pi['X'][str(i-1)]))).flatten()'''
					#ctrl_seq[i,:] = ctrl
					
				'''
				if i==0:
					self.X_p[i] = PFM.simulation(n_ac, 0.001, 0.001, ctrl[:int(len(ctrl)/2)].reshape((xdim, xdim)), ctrl[int(len(ctrl)/2):].reshape((xdim, xdim)), init.reshape((xdim, xdim))).reshape((xdim*xdim, 1))
					# self.X_p[i] = sim.burgers_1d(init, dx, dt, nu, ctrl[:int(len(ctrl)/2)], ctrl[int(len(ctrl)/2):])
				else:
					# self.X_p[i] = sim.burgers_1d(self.X_p[i-1], dx, dt, nu, ctrl[:int(len(ctrl)/2)], ctrl[int(len(ctrl)/2):])
					self.X_p[i] = PFM.simulation(n_ac, 0.001, 0.001, ctrl[:int(len(ctrl)/2)].reshape((xdim, xdim)), ctrl[int(len(ctrl)/2):].reshape((xdim, xdim)), self.X_p[i-1].reshape((xdim, xdim))).reshape((xdim*xdim, 1))
				
				#print(self.X_p[i].reshape((10,10)))
				#self.X_p[i] = PFM.simulation(int(n_ac/int(100/self.N)), 0.001, 0.001, ctrl[:int(len(ctrl)/2)].reshape((xdim, xdim)), ctrl[int(len(ctrl)/2):].reshape((xdim, xdim)), self.X_p[i-1].reshape((xdim, xdim))).reshape((xdim*xdim, 1))
				#self.X_p[i] = PFM.simulation(25, 0.001, 0.001, np.zeros((xdim, xdim)), ctrl[:].reshape((xdim, xdim)), self.X_p[i-1].reshape((xdim, xdim))).reshape((xdim*xdim, 1))
				#self.X_p[i] = PFM_CH.simulation(n_ch, 0.001, dt_ch, ctrl[:int(len(ctrl)/2)].reshape((xdim, xdim)), ctrl[int(len(ctrl)/2):].reshape((xdim, xdim)), self.X_p[i-1].reshape((xdim, xdim))).reshape((xdim*xdim, 1))
				#self.sim.plotCMAP(self.X_p[i].reshape((xdim, xdim)), i)
				#print(np.linalg.norm(self.X_p[i-1]-np.array(Pi['X'][str(i)])))
				#print('Simulated = '+str(ctrl[:]))
				# print('PREV FUCKIN VALUE = ', self.X_p[i-1])
				# print('NOMINAL COST = ',np.array(Pi['U'][str(i)]))
				# print('X_nom = ',np.array(Pi['X'][str(i)]))
				# print('Delta Friggin X = ',self.X_p[i-1] - np.array(Pi['X'][str(i)]))
				# print('X_act = ',self.X_p[i-1])
				# print('X_upd = ',self.X_p[i])
				
					#print('X_non_re_term = ',self.X_p[i])
				'''
				
				if i==0:
					costZ+=self.cost(init, ctrl)
					print('Cost = ',self.cost(init, ctrl))
				else:
					costZ+=self.cost(self.X_p[i-1], ctrl)
					print('Cost = ',self.cost(self.X_p[i-1], ctrl))
				control_cost[i]+=(((ctrl_seq[i].T) @ (.05*2*np.eye(800))) @ ctrl_seq[i])
				#print('X_non_re = ',self.X_p[i])
				#print('U_non_re = ',ctrl)
				if check_plan is True:
					if i==0:
						#np.linalg.norm(self.X_p[i]-np.array(Pi['X'][str(i)]))>3:#say
						#print("The divergence occurs at t = "+str(i))
						#print(np.linalg.norm(self.X_p[i]-np.array(Pi['X'][str(i)])))
						cost += self.cost(init, ctrl)
						# print('X here = ', init.reshape((2,2)))
						print('replanning con cost = ', cost)
						print('replanning inc Cost = ',self.cost(init, ctrl))
						#print('X=',self.X_p[i])
						#print('Control = ',ctrl)
						self.replan(self.X_p[i], i+1, self.N, version+1, cost)
						break
				
				#if render:
					#self.sim.render(mode='window')
			#print(self.X_p[-1])
			#print(np.sum(cst))

		if check_plan is not True:
			costZ+=self.cost_final(self.X_p[self.N-1])
			print('Cost_term = ',self.cost_final(self.X_p[self.N-1]))
			print('Total z cost = '+str(costZ))
			print("Total re cost 1 = "+str(cost+self.cost_final(self.X_p[self.N-1])))#self.cost_final(self.X_p[i])))
			# print("X Final = ", self.X_p[self.N-1].reshape((10,10)))
			#temp = np.load('control_cost_totals_disc.npy')
			print("Control cost = ", control_cost)
			np.shape(control_cost)
			# print('Max Control = ', np.max(ctrl_seq))
			#np.save('control_cost_totals_disc.npy', np.append(temp, control_cost))
			#print('Term rep Cost = ',self.cost_final(self.X_p[i]))
			#print('X_re_term = ',self.X_p[i])
			print(i)
			feedback_costs = np.load('std8_noise_nfb.npy')
			np.save("std8_noise_nfb.npy", np.append(feedback_costs, costZ))

			# replan_costs = np.load('replan_costs_std8_noise.npy')
			# np.save("replan_costs_std8_noise.npy", np.append(replan_costs, cost+self.cost_final(self.X_p[self.N-1])))
			# print('Post-save check')

		#print('Total cost = '+str(self.calculate_total_cost(self.X_p[-1], self.X_p, ctrl_seq, self.N)))

		'''if check_plan is False:
			print(ctrl_seq)
		else:
			print(ctrl_seq[i])'''
		return self.X_p[self.N-1]#self.state_output(self.sim.get_state())
			


	def feedback(self, W_x_LQR, W_u_LQR, W_x_LQR_f, finite_difference_gradients_flag=False):
		'''
		AB matrix comprises of A and B as [A | B] stacked at every ascending time-step, where,
		A - f_x
		B - f_u
		'''	

		P = W_x_LQR_f

		for t in range(self.N-1, 0, -1):

			if finite_difference_gradients_flag:

				AB, V_z_F_XU_XU = self.sys_id_FD(self.X_p[t-1], self.U_p[t], central_diff=1)

			else:

				AB, V_z_F_XU_XU = self.sys_id(self.X_p[t-1], self.U_p[t], central_diff=1)

			A = AB[:, 0:self.n_x]
			B = AB[:, self.n_x:]

			S = W_u_LQR + ( (np.transpose(B) @ P) @ B)

			# LQR gain 
			self.K[t] = -np.linalg.inv(S) @ ( (np.transpose(B) @ P) @ A)
			
			# second order equation
			P = W_x_LQR  +  ((np.transpose(A) @ P) @ A) - ((np.transpose(self.K[t]) @ S) @ self.K[t]) 



	def calculate_total_cost(self, initial_state, state_traj, control_traj, horizon):

		# assign the function to a local function variable
		incremental_cost = self.cost
		# initialize total cost
		# initial_state = initial_state.reshape(np.shape(state_traj[0]))
		cost_total = incremental_cost(initial_state, control_traj[0])
		cost_total += sum(incremental_cost(state_traj[t], control_traj[t+1]) for t in range(0, horizon-1)) #CHANGE HERE FOR COMP/ change here
		cost_total += self.cost_final(state_traj[horizon-1])

		return cost_total



	def regularization_inc_mu(self):

		# increase mu - regularization 

		self.delta = np.maximum(self.delta_0, self.delta_0*self.delta)

		self.mu = np.maximum(self.mu_min, self.mu*self.delta)

		if self.mu > self.mu_max:

			self.mu = self.mu_max


		#print(self.mu)



	def regularization_dec_mu(self):

		# decrease mu - regularization 

		self.delta = np.minimum(1/self.delta_0, self.delta/self.delta_0)

		if self.mu*self.delta > self.mu_min:

			self.mu = self.mu*self.delta

		else:
			self.mu = self.mu_min



	def plot_(self, y, save_to_path=None, x=None, show=1):

		if x==None:
			
			plt.figure(figsize=(7, 5))
			plt.plot(y, linewidth=2)
			plt.xlabel('Training iteration count', fontweight="bold", fontsize=12)
			plt.ylabel('Episodic cost', fontweight="bold", fontsize=12)
			#plt.grid(linestyle='-.', linewidth=1)
			plt.grid(color='.910', linewidth=1.5)
			plt.title('Episodic cost vs No. of training iterations')
			if save_to_path is not None:
				plt.savefig(save_to_path, format='png')#, dpi=1000)
			plt.tight_layout()
			plt.show()
		
		else:

			plt.plot(y, x)
			plt.show()



	def plot_episodic_cost_history(self, save_to_path=None):

		try:
			self.plot_(np.asarray(self.episodic_cost_history).flatten(), save_to_path=save_to_path, x=None, show=1)

		except:

			print("Plotting failed")
			pass


	def save_policy(self, path_to_file):

		Pi = {}
		# Open-loop part of the policy
		Pi['U'] = {}
		# Closed loop part of the policy - linear feedback gains
		Pi['K'] = {}
		Pi['X'] = {}

		for t in range(0, self.N):
			
			Pi['U'][t] = np.ndarray.tolist(self.U_p[t])
			Pi['K'][t] = np.ndarray.tolist(self.K[t])
			Pi['X'][t] = np.ndarray.tolist(self.X_p[t])
			
		with open(path_to_file, 'w') as outfile:  

			json.dump(Pi, outfile)



	def l_x(self, x):

		# z = np.zeros((np.shape(self.Q)[0], 1))
		# z[:obs_dimension,:] = (x - self.X_g.reshape(np.shape(x)))
		return 2*self.Q @ (x - self.X_g)#z


	def l_x_f(self, x):
		# z = np.zeros((np.shape(self.Q)[0], 1))
		# z[:obs_dimension,:] = (x - self.X_g.reshape(np.shape(x)))
		return 2*self.Q_final @(x - self.X_g)


	def l_u(self, u):

		return 2*self.R @ u


	def replan(self):
		#Replan when exceeding a threshold from nominal

		pass