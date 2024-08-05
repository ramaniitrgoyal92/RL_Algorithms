import numpy as np
import math
from ltv_sys_id import LTV_SysID

class ARMA_LTV_SysID(LTV_SysID):

    def __init__(self, MODEL, n_x, n_u, n_z, n_samples=500, pert_sigma = 1e-7):
        self.n_z = n_z
        
        LTV_SysID.__init__(self, MODEL, n_x, n_u, n_samples, pert_sigma)


    def sys_id(self, u_nom):

        '''
            system identification for a given nominal state and control
            returns - a numpy array with F_x and F_u horizantally stacked
		'''
		################## defining local functions & variables for faster access ################
        n_x = self.n_x
        n_u = self.n_u
        n_z = self.n_z
        q = self.q
        q_u = self.q_u
        N = self.N
		##########################################################################################
		
        A_aug = np.zeros((self.n_samples, n_z*q+n_u*(q_u-1), n_z*q+n_u*(q_u-1)))
        B_aug = np.zeros((self.n_samples, n_z*q+n_u*(q_u-1), n_u))
        

        # Generating nominal traj
        Z_norm = self.C @ self.generate_nominal_traj(u_nom)
		
		# # Generating delta_z for all rollouts
		# for j in range(self.n_samples):
		# 	X[:,0] = x_0.reshape((n_x,))
		# 	for i in range(N):
		# 		ctrl[:] = u_nom[i] + U_ [j, n_u*(N-i)+1:n_u*(N-i+1)]
		# 		X[:, i+1] = simulate(None, X[:,i], ctrl).reshape(n_x,)
		# 		delta_z[j, n_z*(N-i)+1:n_z*(N-i+1)+1] = (C @ X[:,i+1]) - Z_norm[:,i+1]

        u_max = np.max(abs(u_nom))
        U_ = 0.1*u_max*np.random.normal(0, self.sigma, (self.n_samples, n_u*(N+1)))
        
        delta_z = np.zeros((self.n_samples, n_z*(N+1)))
        
        X = np.zeros((n_x, N+1))
        ctrl = np.zeros((n_u, 1))

        # Generating delta_z for all rollouts
        for j in range(self.n_samples):
            X[:,0] = self.X_0.reshape((n_x,))
            for i in range(N):
                ctrl[:] = u_nom[i] + U_ [j, n_u*(N-i):n_u*(N-i+1)].reshape(np.shape(u_nom[i]))
                forward_simulate(self.X_0.flatten(),self.U[t].flatten()).reshape(np.shape(self.X_0))
                X[:, i+1] = self.simulate(None, X[:,i], ctrl).reshape(n_x,)

                testsum1 = np.sum(X[:, i+1])
                if np.isnan(testsum1):
                    print('X has nan')
                    print('X0 = ', self.X_0)
                    # print('Nominal Control : ',u_nom)
                    # print('Control = ', ctrl)
                    print('X [',i,'] = ', X[:, i+1])
                    # sys.exit()
                # delta_z[j, n_z*(i+1):n_z*(i+2)] = (C @ X[:,i+1]) - Z_norm[:,i+1]
                delta_z[j, n_z*(N-i-1):n_z*(N-i)] = (self.C @ X[:,i+1]) - Z_norm[:,i+1]



        fitcoef=np.zeros((n_z,n_z*q+n_u*q_u,N)); # M1 * fitcoef = delta_z

        
        for i in range(max(q, q_u),N):
            
            # M1 = np.hstack([delta_z[:, n_z*(N-i+1)+1:n_z*(N-i+q+1)], U_[:, n_u*(N-i+1)+1:n_u*(N-i+q_u+1)]])
            M1 = np.hstack([delta_z[:, n_z*(N-i):n_z*(N-i+q)], U_[:,n_u*(N-i):n_u*(N-i+q_u)]])
            delta = delta_z[:, n_z*(N-i-1):n_z*(N-i)]

            mat, res, rank, S = np.linalg.lstsq(M1, delta, rcond=None)
            # print(np.shape(mat))
            fitcoef[:, :, i] =  mat.T
            A_aug[i, :np.shape(fitcoef)[0],:] =  np.hstack((fitcoef[:, :n_z*q,i], fitcoef[:, n_z*q+n_u:, i]))
            A_aug[i, np.shape(fitcoef)[0]:np.shape(fitcoef)[0]+n_z*(q-1),:] = np.hstack((np.eye(n_z*(q-1)), np.zeros((n_z*(q-1), n_z+n_u*(q_u-1)))))
            # print(np.shape(A_aug[i, np.shape(fitcoef)[0]+n_z*(q-1):,:]))
            A_aug[i, np.shape(fitcoef)[0]+n_z*(q-1):,:] = np.zeros(np.shape(A_aug[i, np.shape(fitcoef)[0]+n_z*(q-1):,:]))
            A_aug[i, np.shape(fitcoef)[0]+n_z*(q-1)+n_u:,n_z*q+1:n_z*q+1+n_u*(q_u-2)] = np.eye(n_u*(q_u-2)) 

            # print(np.shape(fitcoef[:,n_z*q:n_z*q+n_u,i]))
            B_aug[i, :np.shape(fitcoef)[0]+n_z*(q-1), :] = np.vstack([fitcoef[:,n_z*q:n_z*q+n_u,i], np.zeros((n_z*(q-1), n_u))])
            B_aug[i, np.shape(fitcoef)[0]+n_z*(q-1):, :] = np.vstack([np.eye(n_u*min(1, q_u-1)), np.zeros((n_u*(q_u-2), n_u))])
            # sys.exit()

		###############################################################################
		# print(delta_z[:, n_z*(i-q):n_z*(i)].reverse())
		# sys.exit()

		# TEST_NUM=1; #number of monte-carlo runs to verify the fitting result
		# ucheck=0.01*u_max*np.random.random((n_u*(N+1), TEST_NUM))#randn(IN_NUM*(N+1),TEST_NUM); % input used for checking
		# y_sim=np.zeros(n_z*(N+1),TEST_NUM); # output from real system
		# y_pred=zeros(n_z*(N+1),TEST_NUM); # output from arma model
		# y_sim = C @ generate_nominal_traj(x_0, u_nom+ucheck)
		
		# for j in range(TEST_NUM):
		# 	X[:,0] = x_0.reshape((n_x,))
		# 	for i in range(N):
		# 		ctrl[:] = u_nom[i] + ucheck[n_u*i:n_u*(i+1), :].reshape(np.shape(u_nom[i]))
		# 		X[:, i+1] = simulate(None, X[:,i], ctrl).reshape(n_x,)
		# 		y_sim[n_z*(i+1):n_z*(i+2), j] = (C @ X[:,i+1])# - Z_norm[:,i+1]

		# y_pred[:,:]=y_sim(n_u*(N-q-1)+1:n_z*(N+1),:); # manually match the first few steps
		# for i=max(q,qu)+2:1:N % start to apply input after having enough data
		#     M2=[y_pred(n_z*(N-i+1)+1:n_z*(N-i+1+q),:);ucheck(IN_NUM*(N-i+1)+1:IN_NUM*(N-i+qu+1),:)];
		#     y_pred(n_z*(N-i)+1:n_z*(N-i+1),:)=fitcoef(:,:,i)*M2;
		# end
		###############################################################################

		
		# return F_ZU, V_x_F_XU_XU
        return A_aug, B_aug, X

    def generate_nominal_traj(self, u_nom):

        Y = np.zeros((self.n_x, self.N+1))
        Y[:, 0] = self.X_0.reshape((self.n_x,))
		
        for i in range(self.N):
            Y[:, i+1] = self.forward_simulate(None, Y[:,i], u_nom[i, :]).reshape((self.n_x,))

        return Y

    def state_output(state):
        pass