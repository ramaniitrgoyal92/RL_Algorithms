import numpy as np

mu = 0.2

state_dimension = 2
control_dimension = 1
obs_dimension = state_dimension

# Cost parameters for nominal design
Q = 9*np.eye(state_dimension)
Q_terminal = 9000*np.eye(state_dimension)
Q_final = Q_terminal
R = .05*np.eye(control_dimension)

# Number of substeps in simulation
ctrl_state_freq_ratio = 1
dt = 0.01
horizon = 10 #800
nominal_init_stddev = 0.1

q = 2
q_u = 2
alpha = 0.7

# Cost parameters for feedback design
W_x_LQR = 10*np.eye(state_dimension*state_dimension)
W_u_LQR = 2*np.eye(2*control_dimension*control_dimension)
W_x_LQR_f = 100*np.eye(state_dimension*state_dimension)

# D2C parameters
feedback_n_samples = 20
n_substeps = 1