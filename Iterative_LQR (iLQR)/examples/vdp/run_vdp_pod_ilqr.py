import numpy as np
import math
import sys
import os
from pathlib import Path

import sys
import os


# import Iterative_LQR
# from Iterative_LQR.src.main_ilqr import iLQR

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from sim_vdp import SimulateVDP
from vdp_params import *
from main_ilqr import iLQR
from ltv_sys_id import LTV_SysID

class RunVdp(SimulateVDP):

    def __init__(self, mu, state_dimension, control_dimension, dt, C):
        SimulateVDP.__init__(self, mu, state_dimension, control_dimension, dt)
        self.C = C

    def simulate(self,x,u):
        return self.C*self.simulate_vdp(x, u)[-1]


if __name__=="__main__":

    cwd = os.getcwd()
    path_to_vdp = Path(cwd)/"examples/vdp"
    MODEL = path_to_vdp/"models/None.xml"

    path_to_export = path_to_vdp/"VDP_Experiments/exp_1"
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
    n_iterations = 10
    C = np.array([1.0, 0])

    # Create model instance
    run_vdp = RunVdp(mu, state_dimension, control_dimension, dt, C)

    # Create iLQR instance
    ilqr = iLQR(run_vdp, state_dimension, control_dimension, alpha, horizon, init_state, final_state, Q, Q_final, R, 
                nominal_init_stddev, n_sys_id_samples=500, pert_sys_id_sigma=1e-7)
    ilqr.iterate_ilqr(n_iterations)

    ilqr.plot_episodic_cost_history(path_to_training_cost_fig)
    ilqr.save_policy(path_to_policy_file)
    ilqr.save_cost(path_to_cost_file)

    # Check and Simulate the obtained policy
    print
    run_vdp.simulate_vdp(y_init = init_state.flatten(), u = ilqr.U.flatten(), horizon=horizon)
    run_vdp.draw_figure(path_to_traj_fig)

    # Test sys_id
    """ x_t = np.array([2.0,0.0]).reshape(state_dimension,1)
    u_t = np.array([0]).reshape(control_dimension,1)
    AB = model.sys_id(x_t,u_t)
    print(AB) """