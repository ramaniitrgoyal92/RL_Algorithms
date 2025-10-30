import numpy as np
import math
from ltv_sys_id import LTV_SysID
import sys
class ARMA_LTV_SysID(LTV_SysID):

    def __init__(self, MODEL, n_x, n_u, n_z, C, q, q_u, N, n_samples=500, pert_sigma = 1e-3):
        """
        ARMA LTV System Identification
        Args:
            MODEL: dynamics model
            n_x: full state dimension
            n_u: control dimension  
            n_z: observation dimension (e.g., 1 for pendulum, 2 for cartpole)
            q: state history length
            q_u: control history length 
            n_samples: number of perturbed trajectories
            pert_sigma: perturbation standard deviation for actions
        """
        super().__init__(MODEL, n_x, n_u, n_samples, pert_sigma)
        self.model = MODEL
        self.n_x = n_x
        self.n_u = n_u
        self.n_z = n_z
        self.C = C
        self.q = q
        self.q_u = q_u
        self.N = N
        self.sigma = pert_sigma
        self.n_samples = n_samples
        self.pert_sigma = pert_sigma
        
        #LTV_SysID.__init__(self, MODEL, n_x, n_u, n_samples, pert_sigma)

    def traj_sys_id(self, x_nom, u_nom):

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
        self.X_0 = (x_nom[0]).reshape(n_z,1)
        Z_nom = x_nom
		##########################################################################################
        # Generating perturbations
        X, U_ = self.generate_rollouts(x_nom, u_nom)
        Z = self.C @ X
        U_ = U_.reshape(self.N+1*n_u, self.n_samples).T       
        delta_z = np.zeros((self.n_samples, n_z*(N+1)))
        
        # Generating delta_z for all rollouts
        for j in range(self.n_samples):
            for i in range(N):
                delta_z[j, n_z*(N-i-1):n_z*(N-i)] = Z[i+1,j] - Z_nom[i+1]
        
        return self.arma_fit(delta_z, U_, self.n_z, self.n_u, self.q, self.q_u, self.N, self.n_samples)
 

    def arma_fit(self, delta_z, U_, n_z, n_u, q, q_u, N, n_samples):
        """
        ARMA LTV fitting with A_aug and B_aug construction
        """
        fitcoef = np.zeros((n_z, n_z*q + n_u*q_u, N))
    
        # Handle edge case: when q_u=1, there's no control history to store
        aug_dim = n_z*q + n_u*max(0, q_u-1)  # Ensure non-negative dimension
        A_aug = np.zeros((N, aug_dim, aug_dim))
        B_aug = np.zeros((N, aug_dim, n_u))
        
        # Pre-allocate reusable arrays
        M1 = np.zeros((n_samples, n_z*q + n_u*q_u))
        delta = np.zeros((n_samples, n_z))
        
        # Pre-compute constant blocks - handle edge cases
        state_shift_block = None
        if q > 1:
            state_eye = np.eye(n_z*(q-1))
            state_zeros = np.zeros((n_z*(q-1), n_z + n_u*max(0, q_u-1)))
            state_shift_block = np.hstack([state_eye, state_zeros])
        
        ctrl_eye = None
        if q_u > 2:
            ctrl_eye = np.eye(n_u*(q_u-2))
        
        # B_aug constant blocks - handle edge cases
        b_state_zeros = np.zeros((n_z*max(0, q-1), n_u))  # Handle q=1 case
        
        # Handle control history blocks
        if q_u > 1:
            b_ctrl_eye = np.eye(n_u)
            if q_u > 2:
                b_ctrl_zeros = np.zeros((n_u*(q_u-2), n_u))
                b_ctrl_block = np.vstack([b_ctrl_eye, b_ctrl_zeros])
            else:
                b_ctrl_block = b_ctrl_eye
        else:
            # q_u = 1: no control history to store
            b_ctrl_block = np.zeros((0, n_u))  # Empty block
        
        # Main loop
        for i in range(max(q, q_u), N):
            # Build regressors - same as before
            M1[:, :n_z*q] = delta_z[:, n_z*(N-i):n_z*(N-i+q)]
            M1[:, n_z*q:] = U_[:, n_u*(N-i):n_u*(N-i+q_u)]
            delta[:, :] = delta_z[:, n_z*(N-i-1):n_z*(N-i)]
            
            # Solve least squares
            mat, res, rank, S = np.linalg.lstsq(M1, delta, rcond=None)
            fitcoef[:, :, i] = mat.T
            
            # === A_aug construction with edge case handling ===
            
            # Top row: ARMA coefficients
            if q_u > 1:
                # Standard case: include control history terms
                A_aug[i, :n_z, :] = np.hstack((fitcoef[:, :n_z*q, i], fitcoef[:, n_z*q+n_u:, i]))
            else:
                # q_u=1 case: only state terms (no control history)
                A_aug[i, :n_z, :n_z*q] = fitcoef[:, :n_z*q, i]
                # No control history terms to add
            
            # State shifting block (only if q > 1)
            if q > 1 and state_shift_block is not None:
                A_aug[i, n_z:n_z+n_z*(q-1), :] = state_shift_block
            
            # Zero out remaining sections if they exist
            if aug_dim > n_z + n_z*max(0, q-1):
                A_aug[i, n_z+n_z*max(0, q-1):, :] = 0
            
            # Control shifting (only if q_u > 2)
            if q_u > 2 and ctrl_eye is not None:
                row_start = n_z + n_z*max(0, q-1) + n_u
                col_start = n_z*q + 1
                row_end = row_start + n_u*(q_u-2)
                col_end = col_start + n_u*(q_u-2)
                
                # Check bounds to avoid indexing errors
                if row_end <= aug_dim and col_end <= aug_dim:
                    A_aug[i, row_start:row_end, col_start:col_end] = ctrl_eye
            
            # === B_aug construction with edge case handling ===
            
            # Top part: current control and state history zeros
            top_rows = n_z + n_z*max(0, q-1)
            if top_rows > 0:
                if q > 1:
                    B_aug[i, :top_rows, :] = np.vstack([fitcoef[:, n_z*q:n_z*q+n_u, i], b_state_zeros])
                else:
                    # q=1 case: only current control, no state history
                    B_aug[i, :n_z, :] = fitcoef[:, n_z*q:n_z*q+n_u, i]
            
            # Bottom part: control history storage (only if q_u > 1)
            if q_u > 1 and top_rows < aug_dim:
                remaining_rows = aug_dim - top_rows
                if remaining_rows > 0 and b_ctrl_block.shape[0] > 0:
                    # Make sure dimensions match
                    rows_to_fill = min(remaining_rows, b_ctrl_block.shape[0])
                    B_aug[i, top_rows:top_rows+rows_to_fill, :] = b_ctrl_block[:rows_to_fill, :]
        
        return A_aug, B_aug