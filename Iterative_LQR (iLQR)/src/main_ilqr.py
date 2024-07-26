import numpy as np
import matplotlib.pyplot as plt
import json

from ltv_sys_id import LTV_SysID
from progressbar import *

class iLQR:

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
        self.J_change_eps = -6e-1

        # Initialize nominal trajectory
        self.X = np.zeros((self.N, self.n_x, 1))
        self.X_temp = np.zeros((self.N, self.n_x, 1))

        # Initialize nominal control
        self.U = np.zeros((self.N, self.n_u, 1))
        self.U_temp = np.zeros((self.N, self.n_u, 1))

        # Define Feedback Gain matrices
        self.K = np.zeros((self.N, self.n_u, self.n_x))
        self.k = np.zeros((self.N, self.n_u, 1))

        self.V_xx = np.zeros((self.N, self.n_x, self.n_x))
        self.V_x = np.zeros((self.N, self.n_x, 1))

        self.episodic_cost_history = []

    def iterate_ilqr(self, n_iter, u_init=None):
        '''
			Main function that carries out the algorithm at higher level
		'''

		# Initialize the trajectory with the desired initial guess
        self.initialize_traj(u_init=u_init)


        # Start the iLQR iterations
        for i in self.pbar(range(n_iter)):
            backward_pass_flag, del_J_alpha = self.backward_pass()

            if backward_pass_flag:
                self.dec_reg_mu()

                forward_pass_flag = self.forward_pass(del_J_alpha)

                if not forward_pass_flag:

                    while not forward_pass_flag:
                        # simulated annealing
                        self.alpha = self.alpha*0.99
                        forward_pass_flag = self.forward_pass(del_J_alpha)
            else:
                self.inc_reg_mu()
                print(f"Iteration {i} failed.")

            if i<5:
                self.alpha = self.alpha*0.9
            else:
                self.alpha = self.alpha*0.999

            self.episodic_cost_history.append(self.calculate_total_cost(self.X_0, self.X, self.U, self.N))


    def backward_pass(self):
        ################## defining local functions & variables for faster access ################
        k = self.k
        K = self.K
        V_x = self.V_x
        V_xx = self.V_xx
        ##########################################################################################

        V_x[self.N-1] = self.l_x_N(self.X[self.N-1])	
        V_xx[self.N-1] = 2*self.Q_final

        # Initialize before forward pass
        del_J_alpha = 0

        for t in range(self.N-1, -1, -1):
            if t>0:
                Q_x, Q_u, Q_xx, Q_uu, Q_ux = self.get_gradients(self.X[t-1],self.U[t],V_x[t], V_xx[t])
            else:
                Q_x, Q_u, Q_xx, Q_uu, Q_ux = self.get_gradients(self.X_0,self.U[0],V_x[0], V_xx[0])
            

            try:
                np.linalg.cholesky(Q_uu)

            except np.linalg.LinAlgError:
                print("FAILED! Q_uu is not Positive definite at t=",t)
                backward_pass_flag = 0
                k = self.k
                K = self.K
                V_x = self.V_x
                V_xx = self.V_xx

            else:
                backward_pass_flag = 1
                # update gains as follows
                Q_uu_inv = np.linalg.inv(Q_uu)
                k[t] = -(Q_uu_inv @ Q_u)
                K[t] = -(Q_uu_inv @ Q_ux)

                del_J_alpha += -self.alpha*((k[t].T) @ Q_u) - 0.5*self.alpha**2 * ((k[t].T) @ (Q_uu @ k[t]))
				
                if t>0:
                    V_x[t-1] = Q_x + (K[t].T) @ (Q_uu @ k[t]) + ((K[t].T) @ Q_u) + ((Q_ux.T) @ k[t])
                    V_xx[t-1] = Q_xx + ((K[t].T) @ (Q_uu @ K[t])) + ((K[t].T) @ Q_ux) + ((Q_ux.T) @ K[t])

		######################### Update the new gains ##############################################
        self.k = k
        self.K = K
        self.V_x = V_x
        self.V_xx = V_xx

        return backward_pass_flag, del_J_alpha
    
    def forward_pass(self, del_J_alpha):

        #cost before forward pass
        J1 = self.calculate_total_cost(self.X_0, self.X, self.U, self.N)

        self.X_temp = self.X
        self.U_temp = self.U

        self.forward_pass_simulate()

        #cost after forward pass
        J2 = self.calculate_total_cost(self.X_0, self.X, self.U, self.N)

        if (J1-J2)/del_J_alpha < self.J_change_eps:
            forward_pass_flag = 0
            self.X = self.X_temp
            self.U = self.U_temp
        else:
            forward_pass_flag = 1

        return forward_pass_flag

    def get_gradients(self,x,u,V_x_next, V_xx_next):

        Fx_Fu = self.sys_id(x, u)
        F_x = Fx_Fu[:,:self.n_x]
        F_u = Fx_Fu[:,self.n_x:]

        Q_x = self.l_x(x) + ((F_x.T) @ V_x_next)
        Q_u = self.l_u(u) + ((F_u.T) @ V_x_next)

        Q_xx = 2*self.Q + ((F_x.T) @ ((V_xx_next)  @ F_x)) 
        Q_ux = (F_u.T) @ ((V_xx_next + self.mu*np.eye(V_xx_next.shape[0])) @ F_x)
        Q_uu = 2*self.R + (F_u.T) @ ((V_xx_next + self.mu*np.eye(V_xx_next.shape[0])) @ F_u)

        return Q_x, Q_u, Q_xx, Q_uu, Q_ux


    def forward_pass_simulate(self):

        for t in range(self.N):
            if t==0:
                self.U[t] = self.U_temp[t] + self.alpha*self.k[t]
                self.X[t] = self.forward_simulate(self.X_0.flatten(),self.U[t].flatten()).reshape(np.shape(self.X_0))
            else:
                self.U[t] = self.U_temp[t] + self.alpha*self.k[t] + (self.K[t] @ (self.X[t-1] - self.X_temp[t-1]))
                self.X[t] = self.forward_simulate(self.X[t-1].flatten(),self.U[t].flatten()).reshape(np.shape(self.X_0))



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


    def initialize_traj(self,u_init):
        for t in range(0, self.N):
            if u_init is None:
                self.U[t, :] = np.random.normal(0, self.nominal_init_stddev, (self.n_u, 1))	#np.random.normal(0, 0.01, self.n_u).reshape(self.n_u,1)#DM(array[t, 4:6])
            else:
                self.U[t, :] = u_init[t,:]

        self.U_temp = self.U
        
        self.forward_pass_simulate()
        
        self.X_temp = self.X
        
    def forward_simulate(self):
        pass
    
    def calculate_total_cost(self,init_state, state_traj, control_traj, horizon):
        total_cost = 0
        total_cost += self.incremental_cost(init_state,control_traj[0])
        for t in range(horizon-1):
            total_cost += self.incremental_cost(state_traj[t],control_traj[t+1])
        total_cost += self.terminal_cost(state_traj[horizon-1])

        return total_cost

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

    def plot_episodic_cost_history(self,save_to_path=None):
        data = np.array(self.episodic_cost_history).flatten()
        plt.figure(figsize=(7, 5))
        plt.plot(data, linewidth=2)
        plt.xlabel('Training iteration count', fontweight="bold", fontsize=12)
        plt.ylabel('Episodic cost', fontweight="bold", fontsize=12)
        plt.grid(color='.910', linewidth=1.5)
        plt.title('Episodic cost vs No. of training iterations')
        if save_to_path is not None:
            plt.savefig(save_to_path, format='png')
        plt.tight_layout()
        plt.show()

    def save_policy(self, path_to_file):

        Pi = {}
		# Open-loop part of the policy
        Pi['U'] = {}
		# Closed loop part of the policy - linear feedback gains
        Pi['K'] = {}
        Pi['X'] = {}


        for t in range(self.N):
            Pi['X'][t] = self.X[t].tolist()
            Pi['U'][t] = self.U[t].tolist()
            Pi['K'][t] = self.K[t].tolist()

        with open(path_to_file, 'w') as outfile:
            json.dump(Pi,outfile)

    def save_cost(self,path_to_file):
        data = {}
        data['cost'] = np.array(self.episodic_cost_history).flatten().tolist()
        with open(path_to_file, 'w') as outfile:
            json.dump(data,outfile)