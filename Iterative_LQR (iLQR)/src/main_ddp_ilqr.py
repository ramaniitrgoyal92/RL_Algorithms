import numpy as np
import math

from ltv_sys_id import LTV_SysID
from progressbar import *

class DDP_iLQR:

    def __init__(self,MODEL, n_x, n_u, alpha, horizon, init_state, final_state):
        self.X_0 = init_state
        self.X_N = final_state
        self.N = horizon        
        self.n_x = n_x
        self.n_u = n_u
        self.alpha = alpha

        widgets = [Percentage(), '   ', ETA(), ' (', Timer(), ')']
        self.pbar = ProgressBar(widgets=widgets)
        
        self.mu = 1e-3
        self.mu_min = 1e-3
        self.mu_max = (10)**6
        self.delta_0 = 2
        self.delta = self.delta_0

    

    def iterate_DDP(self, n_iter):
        
        for i in self.pbar(range(n_iter)):
            backward_pass_flag = self.backward_pass()

            if backward_pass_flag:
                forward_pass_flag = self.forward_pass()

        pass



    def backward_pass(self):
        #TODO

        for i in range(self.N, -1, -1):

            pass

    def forward_pass(self):
        pass


    def incremental_cost(self,x,u):
        '''
			Incremental cost in terms of state and controls.
            Can be overwritten in actual working example
		'''
        return (((x - self.X_N).T @ self.Q) @ (x - self.X_N)) + (((u.T) @ self.R) @ u)
	
    def terminal_cost(self,x):
        '''
			Terminal cost in terms of state.
            Can be overwritten in actual working example
		'''
        return (((x - self.X_N).T @ self.Q_final) @ (x - self.X_N)) 


    def initialize_traj(self):
        pass

    def forward_simulate(self):
        pass
    
    def calculate_total_cost(self,init_state, state_traj, control_traj, horizon):
        total_cost = 0
        total_cost += self.incremental_cost(init_state,control_traj[0])
        for t in range(horizon-1):
            total_cost += self.incremental_cost(state_traj[t],control_traj[t+1])
        total_cost += self.terminal_cost(state_traj[horizon-1])

    def inc_reg_mu(self):
        '''increase regularization variable mu'''
        self.delta = np.maximum(self.delta_0, self.delta_0*self.delta)
        self.mu *= self.delta
        # self.mu = self.mu_min if self.mu < self.mu_min else self.mu
        self.mu = self.mu_max if self.mu>self.mu_max else self.mu

    def dec_reg_mu(self):
        '''decrease regularization variable mu'''
        self.delta = np.minimum(1/self.delta_0, self.delta/self.delta_0)
        self.mu *= self.delta
        self.mu = self.mu if self.mu > self.mu_min else 0
        # self.mu = self.mu_max if self.mu>self.mu_max else self.mu

    def l_x(self,x):
        return 2*self.Q @ (x - self.X_N)
    
    def l_u(self, u):
        return 2*self.R @ u
    
    def l_x_N(self,x):
        return 2*self.Q_final @ (x - self.X_N)

