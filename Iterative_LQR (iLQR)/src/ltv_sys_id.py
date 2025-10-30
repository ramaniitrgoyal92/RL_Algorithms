import numpy as np
import math
import sys
np.random.seed(42)

class LTV_SysID:

    def __init__(self, MODEL, n_x, n_u, N, n_samples=500, pert_sigma = 1e-7):
        self.model = MODEL
        self.n_x = n_x # State dimension
        self.n_u = n_u # Control dimension
        self.N = N # Horizon length
        self.n_samples = n_samples       
        self.sigma = pert_sigma # Standard deviation of the perturbation 

    def sys_id_state_pertb(self, x_t, u_t):
        '''
            x_t = (nx,1)
            u_t = (nu,1)
			system identification for a given nominal state and control
			returns - a numpy array with F_x and F_u horizantally stacked
		'''
		################## defining local functions & variables for faster access ################
        n_x = self.n_x
        n_u = self.n_u
		##########################################################################################
		
        dXU = np.random.normal(0.0, self.sigma, (self.n_samples, n_x+n_u))
        dX = dXU[:,:n_x]
        dU = dXU[:,n_x:]

        dX_next = self.batch_simulate(x_t.T+dX, u_t.T+dU) - self.batch_simulate(x_t.T, u_t.T)

        AB = np.linalg.solve((dXU.T @ dXU), (dXU.T@dX_next)).T

        return AB

    def batch_simulate(self, X, U):
        '''
        Need a simulate function from the actual example
        forward_simulate(self,x,u)
        '''
        X_next = np.zeros((X.shape[0], self.n_x))
        for i in range(X.shape[0]):
            X_next[i,:] = self.model.simulate(X[i],U[i])

        return X_next

    def traj_sys_id_state_pertb(self,x_nom,u_nom):
        traj_AB = []
        for i in range(u_nom.shape[0]):
            traj_AB.append(self.sys_id_state_pertb(x_nom[i],u_nom[i]))
        
        return np.array(traj_AB)
    
    def traj_sys_id(self, x_nom, u_nom, central_diff=0): #TODO rollout 
        
        
        delta_x = np.zeros(((self.N+1), self.n_x, self.n_samples))
        X, U_ = self.generate_rollouts(x_nom, u_nom)
        traj_AB = []

        # Generating delta_x for all rollouts
        for i in range(self.N+1):
            delta_x[i] = X[i] - x_nom[i]
            if i>0:
                regressor = np.hstack([delta_x[i-1,:].T, (U_[i-1].T).reshape(self.n_samples, self.n_u)])
                AB = np.linalg.lstsq(regressor, delta_x[i].T, rcond=None)[0].T
                traj_AB.append(AB)
                           
        return traj_AB
    
    def generate_rollouts(self, x_nom, u_nom):
        # Taking control perturbations as a function of max control
        u_max = np.max(abs(u_nom))
        U_ = self.sigma*u_max*np.random.normal(0, self.sigma, (self.N+1, self.n_u, self.n_samples))
        X = np.zeros((self.N+1, self.n_x, self.n_samples))
        ctrl = np.zeros((self.n_u, 1))

        for j in range(self.n_samples):
            X[0, :, j] = x_nom[0].flatten()
            for i in range(self.N):
                ctrl[:] = u_nom[i] + U_[i,:,j].reshape(np.shape(u_nom[i]))
                X[i+1, :, j] = self.model.simulate(X[i, :, j], ctrl.flatten()).flatten()

        return X, U_
