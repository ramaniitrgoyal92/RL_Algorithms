import numpy as np
import math
from main_ilqr import iLQR

class Main_POD_iLQR(iLQR):

    def __init__(self, MODEL, n_x, n_u, alpha, horizon, init_state, final_state, n_z, q, q_u):
        self.n_z = n_z
        self.q = q
        self.q_u = q_u
        self.K = np.zeros((self.N, self.n_u, n_z*q+self.n_u*(q_u-1)))
        self.k = np.zeros((self.N, self.n_u, 1))
		
        self.V_zz = np.zeros((self.N, n_z*q+self.n_u*(q_u-1), n_z*q+self.n_u*(q_u-1)))
        self.V_z = np.zeros((self.N, n_z*q+self.n_u*(q_u-1), 1))

        iLQR.__init__(self,MODEL, n_x, n_u, alpha, horizon, init_state, final_state)

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

        # TEST CODE
        del_J_alpha, b_pass_success_flag = partials_list(self.X_p_0, self.U_p, V_z, V_zz, del_J_alpha)
        A_aug, B_aug, V_z_F_XU_XU, traj = self.sys_id(x_0, u_nom, central_diff=1, V_z=V_z)

        # for t in range(self.N-1, -1, -1):
        for t in range(self.N-1, max(self.q, self.q_u)-1, -1):
            if t>0:
                Q_z, Q_u, Q_zz, Q_uu, Q_uz = self.get_gradients(self.X[t-1],self.U[t],V_z[t], V_zz[t])
            else:
                Q_z, Q_u, Q_zz, Q_uu, Q_uz = self.get_gradients(self.X_0,self.U[0],V_z[0], V_zz[0])
            

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

    def get_gradients(self,x,u, V_z, V_zz):

        Fx_Fu = self.sys_id(x, u)
        F_x = Fx_Fu[:,:self.n_x]
        F_u = Fx_Fu[:,self.n_x:]

        Q_z = self.l_x(traj[:,t].reshape(self.n_x,1)) + ((F_x[t].T) @ V_z[t])
        Q_u = self.l_u(u_nom[t]) + ((F_u[t].T) @ V_z[t])
        Q_zz = 2*self.Q + ((F_x[t].T) @ ((V_zz[t])  @ F_x[t])) 
        Q_uz = (F_u[t].T) @ ((V_zz[t] + self.mu*np.eye(V_zz[t].shape[0])) @ F_x[t])
        Q_uu = 2*self.R + (F_u[t].T) @ ((V_zz[t] + self.mu*np.eye(V_zz[t].shape[0])) @ F_u[t])

        return Q_x, Q_u, Q_xx, Q_uu, Q_ux
    
    

			