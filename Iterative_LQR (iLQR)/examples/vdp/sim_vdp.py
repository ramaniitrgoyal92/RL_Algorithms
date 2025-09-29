"""
Van der Pol oscillator
-----------------------

         y''-mu*(1-y^2)*y'+y=0    y(0)=2,  y'(0)=0
The first step is to rewrite as a system of equations:
         y1'=y2      y2'=mu*(1-y^2)*y'+y    y1(0)=2,  y2(0)=0
The function vanderpol() returns the RHS vector for this system.

The function run_vanderpol() sets up parameters, runs the solver,
and returns the solution. 
The function draw_figures() plots a few figures with this data.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

class SimulateVDP:

    def __init__(self,mu,nx,nu,dt):
        self.mu = mu
        self.nx = nx
        self.nu = nu
        self.dt = dt

    def vanderpol(self,y,u):
        """ Return the derivative vector for the van der Pol equations."""
        y1= y[0]
        y2= y[1]
        dy1=y2
        dy2=self.mu*(1-y1**2)*y2-y1+u
        return dy1, dy2
    
    def onestep_rk4(self,y_init, n_per_step, u = 0):
        h = self.dt/n_per_step
        y1 = y_init[0]
        y2 = y_init[1]
        y = np.zeros((n_per_step,self.nx))
        for i in range(n_per_step):
            k1_x1, k1_x2 = self.vanderpol(np.asarray([y1, y2]), u)
            k2_x1, k2_x2 = self.vanderpol(np.asarray([y1+(k1_x1*h)/2, y2+(k1_x2*h)/2]), u)
            k3_x1, k3_x2 = self.vanderpol(np.asarray([y1+(k2_x1*h)/2, y2+(k2_x2*h)/2]), u)
            k4_x1, k4_x2 = self.vanderpol(np.asarray([y1+(k3_x1*h), y2+(k3_x2*h)]), u)
            y1 += h/6*(k1_x1 + 2*k2_x1 + 2*k3_x1 + k4_x1)
            y2 += h/6*(k1_x2 + 2*k2_x2 + 2*k3_x2 + k4_x2)
            y[i,:] = np.array([y1, y2])
        return y

    def simulate_vdp(self, y_init = np.array([2.0,0.0]), u = np.array([0.0]), horizon=1, n_per_step = 20):
        if u.shape[0] !=horizon:
            u = np.zeros([horizon])
        total_steps = horizon*n_per_step
        self.T = np.linspace(0, horizon, total_steps)
        self.Y = np.zeros((total_steps,self.nx))
        for i in range(horizon):
            y = self.onestep_rk4(y_init, n_per_step, u[i])
            self.Y[n_per_step*i:n_per_step*(i+1), :] = y
            y_init = y[-1]
        return self.Y
    
    def draw_figure(self, save_to_path=None):
        plt.subplot(2,2,1)
        plt.plot(self.T, self.Y[:,0],'-ob')
        plt.title("Time Profile of y1")
        plt.subplot(2,2,3)
        plt.plot(self.T, self.Y[:,1],'-og')
        plt.title("Time Profile of y2")
        # plt.subplot2grid((2,2),(0,1),rowspan=2)
        # in 2x2 grid of subplots, draw in (0,1)-->upper right, 
        # and make the plot span two rows of plots (height is twice normal)
        plt.subplot2grid((2,2),(0,1),rowspan=2)
        plt.plot(self.Y[:,0], self.Y[:,1],'-ok')
        plt.title("Phase Portrait")
        if save_to_path is not None:
            plt.savefig(save_to_path, format='png')
        plt.show()


if __name__ == '__main__':

    cwd = os.getcwd()
    file_loc = Path(cwd)/"Iterative_LQR (iLQR)/examples/vdp"
    file = file_loc/"policies/control.npy"
    

    mu, nx, nu, dt = 2, 2, 1, 1
    y_init = np.array([2.0,0.0])

    time_horizon = 10
    control = np.array([0.33999999999999986, 1.9, 0.14000000000000012, 0.28000000000000025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # control = np.load(file).flatten()
    # control = np.zeros((time_horizon))

    sim_module = SimulateVDP(mu, nx, nu, dt)
    sim_module.simulate_vdp(u=control, horizon=time_horizon)
    # sim_module.simulate_vdp(y_init, control)
    sim_module.draw_figure()

    
