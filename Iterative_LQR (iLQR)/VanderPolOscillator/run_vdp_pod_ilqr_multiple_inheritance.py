import numpy as np
import math
import sys
import os
from pathlib import Path


# Add src directory to sys.path
src_path = Path(__file__).resolve().parent.parent / 'src'
sys.path.append(str(src_path))

from main_ilqr import iLQR
from ltv_sys_id import LTV_SysID
from sim_vdp import SimulateVDP
from main_pod_ilqr import POD_iLQR
from arma_ltv_sys_id import ARMA_LTV_SysID
from vdp_params import *

class RunVdp(iLQR,LTV_SysID,SimulateVDP):

    def __init__(self, mu, state_dimension, control_dimension, dt, horizon, init_state, final_state, Q, Q_final, R, alpha, nominal_init_stddev):

        self.Q = Q
        self.R = R
        self.Q_final = Q_final
        self.nominal_init_stddev = nominal_init_stddev
        
        iLQR.__init__(self, None, state_dimension, control_dimension, alpha, horizon, init_state, final_state)
        LTV_SysID.__init__(self, None, state_dimension, control_dimension, n_samples=100, pert_sigma = 1e-7)
        SimulateVDP.__init__(self, mu, state_dimension, control_dimension, dt)

    def simulate(self,x,u,part_obs_flag=False):
        if not part_obs_flag:
            return self.simulate_vdp(x, u)[-1]
        else:
            return self.C*self.simulate_vdp(x, u)[-1]

class RunPodVdp(RunVdp,POD_iLQR,ARMA_LTV_SysID,SimulateVDP):
    def __init__(self, mu, state_dimension, control_dimension, dt, horizon, init_state, final_state, Q, Q_final, R, alpha, nominal_init_stddev, C):       
        self.C = C
        n_z = C.shape[0]

        RunVdp.__init__(self, mu, state_dimension, control_dimension, dt, horizon, init_state, final_state, Q, Q_final, R, alpha, nominal_init_stddev)
        POD_iLQR.__init__(self, None, state_dimension, control_dimension, alpha, horizon, init_state, final_state, n_z, q, q_u)
        ARMA_LTV_SysID.__init__(self, None, state_dimension, control_dimension, n_z, n_samples=500, pert_sigma = 1e-7)
        SimulateVDP.__init__(self, mu, state_dimension, control_dimension, dt)


if __name__=="__main__":
    cwd = os.getcwd()
    path_to_vdp = Path(cwd)/"Iterative_LQR (iLQR)/VanderPolOscillator"
    MODEL = path_to_vdp/"models/None.xml"

    path_to_export = path_to_vdp/"VDP_Experiments/exp_3"
    path_to_policy_file = path_to_export/"vdp_policy.txt"
    path_to_cost_file = path_to_export / "training_cost_data.txt"
    path_to_training_cost_fig = path_to_export/"episodic_cost_training.png"
    path_to_traj_fig = path_to_export/"optimal_traj.png"
    # path_to_data = path_to_export / "vdp_D2C_data.txt"

    init_state = np.zeros((state_dimension,1))
    init_state[0] = .02
    final_state = np.zeros((state_dimension, 1))

    print('Initial phase : \n', init_state)
    print('Goal phase : \n', final_state)
	
    # No. of ILQR iterations to run
    n_iterations = 20
    C = np.array([1.0, 0])
    
    model = RunPodVdp(mu, state_dimension, control_dimension, dt, horizon, init_state, final_state, Q, Q_final, R, alpha, nominal_init_stddev, C)
    model.iterate_ilqr(n_iterations)
    model.plot_episodic_cost_history(path_to_training_cost_fig)
    model.save_policy(path_to_policy_file)
    model.save_cost(path_to_cost_file)

    #Check and Simulate the obtained policy
    model.simulate_vdp(y_init = init_state.flatten(), u = model.U.flatten(), horizon = horizon)
    model.draw_figure(path_to_traj_fig)

    # Test sys_id
    """ x_t = np.array([2.0,0.0]).reshape(state_dimension,1)
    u_t = np.array([0]).reshape(control_dimension,1)
    AB = model.sys_id(x_t,u_t)
    print(AB) """