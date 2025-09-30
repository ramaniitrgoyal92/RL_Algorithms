import numpy as np
import math
from main_ilqr import iLQR

class POD_iLQR(iLQR):

    def __init__(self, C, MODEL, n_x, n_u, alpha, horizon, init_state, final_state, n_z, q, q_u, Q, Q_final, R, 
                 nominal_init_stddev, n_sys_id_samples, pert_sys_id_sigma, arma_sys_id_flag = True):
        self.C = C
        self.n_z = n_z
        self.q = q
        self.q_u = q_u
        self.K = np.zeros((self.N, self.n_u, n_z*q+self.n_u*(q_u-1)))
        self.k = np.zeros((self.N, self.n_u, 1))
		
        self.V_zz = np.zeros((self.N, n_z*q+self.n_u*(q_u-1), n_z*q+self.n_u*(q_u-1)))
        self.V_z = np.zeros((self.N, n_z*q+self.n_u*(q_u-1), 1))

        iLQR.__init__(self, MODEL, n_x, n_u, alpha, horizon, init_state, final_state, 
                      Q, Q_final, R, nominal_init_stddev, n_sys_id_samples, pert_sys_id_sigma, arma_sys_id_flag = arma_sys_id_flag)

    def iterate_ilqr(self, n_iter, u_init=None):
    # exactly same from iLQR, will be removed later
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
        V_z = self.V_z
        V_zz = self.V_zz
        ##########################################################################################

        V_z[self.N-1] = self.l_x_N(self.X[self.N-1])	
        V_zz[self.N-1] = 2*self.Q_final

        # Initialize before forward pass
        del_J_alpha = 0

        #TODO: TEST CODE
        # del_J_alpha, b_pass_success_flag = partials_list(self.X_p_0, self.U_p, V_z, V_zz, del_J_alpha)
        A_aug, B_aug, V_z_F_XU_XU, traj = self.ltv_sys_id.traj_sys_id_state_pertb(np.concatenate((self.X_0.reshape(1, self.n_x, 1), self.X), axis=0), self.U)
        # A_aug, B_aug, V_z_F_XU_XU, traj = self.sys_id(x_0, u_nom, central_diff=1, V_z=V_z)

        # for t in range(self.N-1, -1, -1):
        for t in range(self.N-1, max(self.q, self.q_u)-1, -1):
            F_x = A_aug[t]
            F_u = B_aug[t]
            if t>0:
                Q_z, Q_u, Q_zz, Q_uu, Q_uz = self.get_gradients(F_x,F_u,self.X[t-1],self.U[t],V_z[t], V_zz[t])
            else:
                Q_z, Q_u, Q_zz, Q_uu, Q_uz = self.get_gradients(F_x,F_u,self.X_0,self.U[0],V_z[0], V_zz[0])
            

            try:
                np.linalg.cholesky(Q_uu)

            except np.linalg.LinAlgError:
                print("FAILED! Q_uu is not Positive definite at t=",t)
                backward_pass_flag = 0
                k = self.k
                K = self.K
                V_z = self.V_z
                V_zz = self.V_zz
            else:
                backward_pass_flag = 1
                # update gains as follows
                Q_uu_inv = np.linalg.inv(Q_uu)
                k[t] = -(Q_uu_inv @ Q_u)
                K[t] = -(Q_uu_inv @ Q_uz)

                del_J_alpha += -self.alpha*((k[t].T) @ Q_u) - 0.5*self.alpha**2 * ((k[t].T) @ (Q_uu @ k[t]))
				
                if t>0:
                    V_z[t-1] = Q_z + (K[t].T) @ (Q_uu @ k[t]) + ((K[t].T) @ Q_u) + ((Q_uz.T) @ k[t])
                    V_zz[t-1] = Q_zz + ((K[t].T) @ (Q_uu @ K[t])) + ((K[t].T) @ Q_uz) + ((Q_uz.T) @ K[t])

		######################### Update the new gains ##############################################
        self.k = k
        self.K = K
        self.V_z = V_z
        self.V_zz = V_zz

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

    def get_gradients(self,F_x,F_u,x,u, V_z, V_zz):
        # Exactly same from iLQR, will be removed later
        # Q_z = self.l_x(traj[:,t].reshape(self.n_x,1)) + ((F_x.T) @ V_z)
        Q_z = self.l_x(x) + ((F_x.T) @ V_z)
        Q_u = self.l_u(u) + ((F_u.T) @ V_z)

        Q_zz = 2*self.Q + ((F_x.T) @ (V_zz @ F_x)) 
        Q_uz = (F_u.T) @ ((V_zz + self.mu*np.eye(V_zz.shape[0])) @ F_x)
        Q_uu = 2*self.R + (F_u.T) @ ((V_zz + self.mu*np.eye(V_zz.shape[0])) @ F_u)

        return Q_z, Q_u, Q_zz, Q_uu, Q_uz
        
    
    def forward_pass_simulate(self):
        for t in range(self.N):
            if t==0:
                self.U[t] = self.U_temp[t] + self.alpha*self.k[t] #TODO check for K(x-X_0)
                self.X[t] = self.model.simulate(self.X_0.flatten(),self.U[t].flatten()).reshape(np.shape(self.X_0))
            else:
                self.U[t] = self.U_temp[t] + self.alpha*self.k[t] + (self.K[t] @ (self.X[t-1] - self.X_temp[t-1]))
                self.X[t] = self.model.simulate(self.X[t-1].flatten(),self.U[t].flatten()).reshape(np.shape(self.X_0))

    
    def l_x(self, x):
        z = np.zeros((np.shape(self.Q)[0], 1))
        z[:self.n_z,:] = self.C @(x - self.X_N)
        return 2*self.Q @ z

    def l_x_f(self, x):
        z = np.zeros((np.shape(self.Q)[0], 1))
        z[:self.n_z,:] = self.C @(x - self.X_N)
        return 2*self.Q_final @ z