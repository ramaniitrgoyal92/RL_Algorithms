import numpy as np
import math

class LTV_SysID:

    def __init__(self, MODEL, n_x, n_u, n_samples=500, pert_sigma = 1e-7):
        self. n_x = n_x
        self.n_u = n_u
        self.n_samples = n_samples      
        self.sigma = pert_sigma # Standard deviation of the perturbation 

    def sys_id(self, x_t, u_t):
        '''
            x_t = (nx,1)
            u_t = (nu,1)
			system identification for a given nominal state and control
			returns - a numpy array with F_x and F_u horizantally stacked
		'''
		################## defining local functions & variables for faster access ################
        simulate = self.simulate
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
        X_next = np.zeros((X.shape[0], self.nx))
        for i in range(X.shape[0]):
            X_next[i,:] = self.simulate(X[i],U[i])

        return X_next

    def traj_sys_id(self,x_nom,u_nom):
        traj_AB = []

        for i in range(u_nom.shape[0]):
            traj_AB.append(self.sys_id(x_nom[i],u_nom[i]))
        
        return np.array(traj_AB)
