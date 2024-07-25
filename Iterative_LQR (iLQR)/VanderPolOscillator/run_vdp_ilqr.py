import numpy as np
import math
import sys
import os
from pathlib import Path


# Add src directory to sys.path
# src_path = Path(__file__).resolve().parent.parent / 'src'
# sys.path.append(str(src_path))

from main_ddp_ilqr import DDP_iLQR
from ltv_sys_id import LTV_SysID
from sim_vdp import SimulateVDP
from vdp_params import *

class RunVdpDDP(DDP_iLQR,LTV_SysID,SimulateVDP):

    def __init__(self, mu, state_dimension, control_dimension, alpha, horizon, init_state, final_state, Q, Q_final, R):

        self.Q = Q
        self.R = R
        self.Q_final = Q_final
        
        DDP_iLQR.__init__(self, None, state_dimension, control_dimension, alpha, horizon, init_state, final_state)
        LTV_SysID.__init__(self, None, state_dimension, control_dimension, n_samples=500, pert_sigma = 1e-7)
        SimulateVDP.__init__(self, mu, state_dimension, control_dimension)

    def forward_simulate(self,x,u):
        return self.simulate(x, u)[-1]


if __name__=="__main__":
    cwd = os.getcwd()
    path_to_vdp = Path(cwd)/"Iterative_LQR (iLQR)/VanderPolOscillator"
    MODEL = path_to_vdp/"models/None.xml"

    path_to_export = path_to_vdp/"VDP_Experiments/exp_2"
    path_to_file = path_to_export/"vdp_policy_0.txt"
    training_cost_data_file = path_to_export / "training_cost_data.txt"
    path_to_data = path_to_export / "vdp_D2C_DDP_data.txt"

    init_state = np.zeros((state_dimension,1))
    init_state[0] = 2
    final_state = np.zeros((state_dimension, 1))

    print('Initial phase : \n', init_state)
    print('Goal phase : \n', final_state)
	
    # No. of ILQR iterations to run
    n_iterations = 60

    model = RunVdpDDP(mu, state_dimension, control_dimension, alpha, horizon, init_state, final_state, Q, Q_final, R)
    

    # Test sys_id
    '''x_t = np.array([2.0,0.0]).reshape(state_dimension,1)
    u_t = np.array([0]).reshape(control_dimension,1)
    AB = model.sys_id(x_t,u_t)
    print(AB)'''

    #Test Simulate
    '''model.simulate()
    model.draw_figure()'''