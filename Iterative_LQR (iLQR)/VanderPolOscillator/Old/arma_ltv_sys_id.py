
#!/usr/bin/env python

import numpy as np
import scipy.linalg.blas as blas
import math
import requests
# import docker
import time
import sys
#from mujoco_py import load_model_from_path, MjSim, MjViewer
from vanderpol import *
from vdp_params import *

class ltv_sys_id_class(object):

	def __init__(self, model_xml_string,  state_size, action_size, n_substeps=1, n_samples=500):

		self.n_x = state_size
		self.n_u = action_size
		self.N = horizon

		# Standard deviation of the perturbation 
		self.sigma = 1e-3#1e-7
		self.n_samples = n_samples

		#GSo Mask for actuation
		self.inputs = None
		
		#self.sim = ACsolver()#MjSim(load_model_from_path(model_xml_string), nsubsteps=n_substeps)
		


	def sys_id(self, x_0, u_nom, central_diff, activate_second_order=0, V_z=None):

		'''
			system identification for a given nominal state and control
			returns - a numpy array with F_x and F_u horizontally stacked
		'''
		################## defining local functions & variables for faster access ################
		generate_nominal_traj = self.generate_nominal_traj
		simulate = self.simulate
		n_x = self.n_x
		n_u = self.n_u
		n_z = self.n_x#np.shape(C)[0]
		q = 1
		q_u = 1
		# start = self.initInstances(url_list, start_time=0, warmup_period=0, step=300)
		##########################################################################################

		A_aug = np.zeros((self.N, n_x, n_x))#n_z*q+n_u*(q_u-1), n_z*q+n_u*(q_u-1)))
		B_aug = np.zeros((self.N, n_x, n_u))#n_z*q+n_u*(q_u-1), n_u))
		

		# Generating nominal traj
		Z_norm = generate_nominal_traj(x_0, u_nom)
		# print(np.shape(Z_norm))
		# sys.exit()

		u_max = np.max(abs(u_nom))
		# print('u_nom = ',u_nom)
		U_ = np.random.normal(0, self.sigma, (self.n_samples, n_u*(horizon)))
		for t in range(horizon):
			U_[:, n_u*t] = u_max*U_[:, n_u*t]

		delta_z = np.zeros((self.n_samples, n_z*(horizon+1)))
		X = np.zeros((horizon+1, n_x))
		ctrl = np.zeros((n_u, horizon))

		# Generating delta_z for all rollouts
		for j in range(self.n_samples):
			X[0] = x_0.flatten()
			ctrl = u_nom + U_[j, :].reshape((np.shape(u_nom)))
			# for k in range(horizon):
			# 	ctrl[k,0] = min(max(285.15, ctrl[k,0]), 313.15)
			# 	ctrl[k,1] = min(max(0.0, ctrl[k,1]), 1.0)
			# U_[j,:] = (ctrl - u_nom).reshape(np.shape(U_[j,:]))
			# X[i+1] = simulate(X[0], ctrl).reshape(np.shape(X[1:]))

			for i in range(horizon):
				X[i + 1] = simulate(X[i], ctrl[:,i]).reshape(n_x, )
				delta_z[j,n_z*(horizon-i-1):n_z*(horizon-i)] = (X-Z_norm)[i+1,:].reshape((n_x,))#X[:, 1:]-Z_norm.T[:, 1:]


			# for i in range(horizon):
			# 	ctrl[:] = u_nom[i] + U_ [j, n_u*(horizon-i):n_u*(horizon-i+1)].reshape(np.shape(u_nom[i]))
			# 	X[:, i+1] = simulate(None, X[:,i], ctrl).reshape(n_x,)
			# 	testsum1 = np.sum(X[:, i+1])
			# 	if np.isnan(testsum1):
			# 		print('X has nan')
			# 		print('X0 = ', x_0)
			# 		print('X [',i,'] = ', X[:, i+1])
			# 		sys.exit()
			# 	# delta_z[j, n_z*(i+1):n_z*(i+2)] = (C @ X[:,i+1]) - Z_norm[:,i+1]
			# 	delta_z[j, n_z*(horizon-i-1):n_z*(horizon-i)] = (X[:,i+1]) - Z_norm[:,i+1]


		fitcoef=np.zeros((n_z,n_z*q+n_u*q_u,horizon)); # M1 * fitcoef = delta_z


		for i in range(max(q, q_u),horizon):
			
			# M1 = np.hstack([delta_z[:, n_z*(horizon-i+1)+1:n_z*(horizon-i+q+1)], U_[:, n_u*(horizon-i+1)+1:n_u*(horizon-i+q_u+1)]])
			M1 = np.hstack([delta_z[:, n_z*(horizon-i):n_z*(horizon-i+q)], U_[:,n_u*(horizon-i):n_u*(horizon-i+q_u)]])
			delta = delta_z[:, n_z*(horizon-i-1):n_z*(horizon-i)]
			
			mat, res, rank, S = np.linalg.lstsq(M1, delta, rcond=None)
			# print(np.shape(mat))
			fitcoef[:, :, i] = mat.T
			A_aug[i, :, :] = fitcoef[:, :n_z, i]
			B_aug[i, :, :] = fitcoef[:, n_z:n_z+n_u, i]
			# sys.exit()


		V_x_F_XU_XU = None


		###############################################################################
		# print(delta_z[:, n_z*(i-q):n_z*(i)].reverse())
		# sys.exit()

		# TEST_NUM=1; #number of monte-carlo runs to verify the fitting result
		# ucheck=0.01*u_max*np.random.random((n_u*(horizon+1), TEST_NUM))#randn(IN_NUM*(horizon+1),TEST_NUM); % input used for checking
		# y_sim=np.zeros(n_z*(horizon+1),TEST_NUM); # output from real system
		# y_pred=zeros(n_z*(horizon+1),TEST_NUM); # output from arma model
		# y_sim = C @ generate_nominal_traj(x_0, u_nom+ucheck)
		
		# for j in range(TEST_NUM):
		# 	X[:,0] = x_0.reshape((n_x,))
		# 	for i in range(horizon):
		# 		ctrl[:] = u_nom[i] + ucheck[n_u*i:n_u*(i+1), :].reshape(np.shape(u_nom[i]))
		# 		X[:, i+1] = simulate(None, X[:,i], ctrl).reshape(n_x,)
		# 		y_sim[n_z*(i+1):n_z*(i+2), j] = (C @ X[:,i+1])# - Z_norm[:,i+1]

		# y_pred[:,:]=y_sim(n_u*(horizon-q-1)+1:n_z*(horizon+1),:); # manually match the first few steps
		# for i=max(q,qu)+2:1:horizon % start to apply input after having enough data
		#     M2=[y_pred(n_z*(horizon-i+1)+1:n_z*(horizon-i+1+q),:);ucheck(IN_NUM*(horizon-i+1)+1:IN_NUM*(horizon-i+qu+1),:)];
		#     y_pred(n_z*(horizon-i)+1:n_z*(horizon-i+1),:)=fitcoef(:,:,i)*M2;
		# end
		###############################################################################


		# return F_ZU, V_x_F_XU_XU
		return A_aug, B_aug, V_x_F_XU_XU, Z_norm.T




	def sys_id_FD(self, x_t, u_t, central_diff, activate_second_order=0, V_x_=None):

		'''
			system identification by a forward finite-difference for a given nominal state and control
			returns - a numpy array with F_x and F_u horizantally stacked
		'''
		################## defining local functions & variables for faster access ################

		simulate = self.simulate
		n_x = self.n_x
		n_u = self.n_u
		##########################################################################################
		
		z_t = x_t
		XU = np.random.normal(0.0, self.sigma, (n_x, n_x + n_u))
		X_ = np.copy(XU[:, :n_x])
		U_ = np.copy(XU[:, n_x:])

		F_XU = np.zeros((n_x, n_x + n_u))
		V_x_F_XU_XU = None

		x_t_next = simulate(x_t.T, u_t.T)

		if central_diff:
			
			for i in range(0, n_x):
				for j in range(0, n_x):

					delta = np.zeros((1, n_x))
					delta[:, j] = XU[i, j]
					
					F_XU[i, j] = (simulate(x_t.T + delta, u_t.T)[:, i] - x_t_next[:, i])/XU[i, j]

			for i in range(0, n_x):
				for j in range(0, n_u):

					delta = np.zeros((1, n_u))
					delta[:, j] = XU[i, n_x + j]
					F_XU[i, n_x + j] = (simulate(x_t.T , u_t.T + delta)[:, i] - x_t_next[:, i])/XU[i, n_x + j]
					
		else:

			Y = (simulate((x_t.T) + X_, (u_t.T) + U_) - simulate((x_t.T), (u_t.T)))

		return F_XU, V_x_F_XU_XU	

	def forward_simulate(self, sim, x, u):

		'''
			Function to simulate a single input and a single current state
			Note : The initial time is set to be zero. So, this can only be used for independent simulations
			x - append time (which is zero here due to above assumption) before state
		'''
		# sdim = int(math.sqrt(state_dimension))
		# sim.set_state_from_flattened(x)
		# sim.forward()
		# sim.data.
		# ctrl = np.zeros((self.n_u, 1))
		ctrl = u

		# return simulator.simulation(tperiod, nu, dt, ctrl[:int(len(ctrl) / 2)], ctrl[int(len(ctrl) / 2):], x)
		return run_vanderpol(yinit=x, tfinal=1, forcing=ctrl, mu=2)[-1].reshape(np.shape(self.X_p[0]))

	def simulate(self, X, U):

		'''
		Function to simulate a batch of inputs given a batch of control inputs and current states
		X - vector of states vertically stacked
		U - vector of controls vertically stacked
		'''
		################## defining local functions & variables for faster access ################

		# sim = self.sim
		forward_simulate = self.forward_simulate
		state_output = self.state_output

		##########################################################################################

		X_next = []
		# Augmenting X by adding a zero column corresponding to time
		# X = np.hstack((np.zeros((X.shape[0], 1)), X))
		print('X = ', X)
		for i in range(X.shape[0]):
			X_next.append(state_output(forward_simulate(None, X, U[i])))

		return np.asarray(X_next)[:, :, 0]
	

	def vec2symm(self, ):
		pass

			

	def traj_sys_id(self, x_nominal, u_nominal):	
		
		'''
			System identification for a nominal trajectory mentioned as a set of states
		'''
		
		Traj_jac = []
		
		for i in range(u_nominal.shape[0]):
			
			Traj_jac.append(self.sys_id(x_nominal[i], u_nominal[i]))

		return np.asarray(Traj_jac)
		

	
	def state_output(state):

		pass


	def khatri_rao(self, B, C):
		"""
	    Calculate the Khatri-Rao product of 2D matrices. Assumes blocks to
	    be the columns of both matrices.
	 
	    See
	    http://gmao.gsfc.nasa.gov/events/adjoint_workshop-8/present/Friday/Dance.pdf
	    for more details.
	 
	    Parameters
	    ----------
	    B : ndarray, shape = [n, p]
	    C : ndarray, shape = [m, p]
	 
	 
	    Returns
	    -------
	    A : ndarray, shape = [m * n, p]
	 
	    """
		if B.ndim != 2 or C.ndim != 2:
			raise ValueError("B and C must have 2 dimensions")
		
		n, p = B.shape
		m, pC = C.shape

		if p != pC:
			raise ValueError("B and C must have the same number of columns")

		return np.einsum('ij, kj -> ikj', B, C).reshape(m * n, p)

	def generate_nominal_traj(self, x_0, u_nom):
		forward_simulate = self.forward_simulate

		Y = np.zeros((state_dimension, horizon + 1))
		Y[:, 0] = x_0.reshape((state_dimension,))

		for i in range(horizon):
			Y[:, i + 1] = forward_simulate(None, Y[:, i], u_nom[i, :]).reshape((state_dimension,))

		return Y


