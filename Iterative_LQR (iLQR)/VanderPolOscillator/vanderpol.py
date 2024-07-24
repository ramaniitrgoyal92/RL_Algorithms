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
import sys
import os
from pathlib import Path
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def vanderpol(y,u,mu):
    """ Return the derivative vector for the van der Pol equations."""
    y1= y[0]
    y2= y[1]
    dy1=y2
    dy2=mu*(1-y1**2)*y2-y1+u
    return dy1, dy2

def rk4(yinit=np.asarray([2, 0]), tfinal=1, n_steps = 20, u=0, mu=2):
    h = tfinal/n_steps
    y1 = yinit[0]
    y2 = yinit[1]
    y = np.zeros((n_steps, 2))
    for i in range(n_steps):
        k1_x1, k1_x2 = vanderpol(np.asarray([y1, y2]), u, mu)
        k2_x1, k2_x2 = vanderpol(np.asarray([y1+(k1_x1*h)/2, y2+(k1_x2*h)/2]), u, mu)
        k3_x1, k3_x2 = vanderpol(np.asarray([y1+(k2_x1*h)/2, y2+(k2_x2*h)/2]), u, mu)
        k4_x1, k4_x2 = vanderpol(np.asarray([y1+(k3_x1*h), y2+(k3_x2*h)]), u, mu)
        y1 += h/6*(k1_x1 + 2*k2_x1 + 2*k3_x1 + k4_x1)
        y2 += h/6*(k1_x2 + 2 * k2_x2 + 2 * k3_x2 + k4_x2)
        y[i,0] = y1
        y[i,1] = y2
    return y

def run_vanderpol(yinit=np.asarray([2,0]), tfinal=1, n_steps = 20, forcing=0, mu=2):
    """ Example for how to run odeint.
    More info found in the doc_string. In ipython type odeint?
    """
    # times = np.linspace(0,tfinal,10)
    # rtol=1e-6
    # atol=1e-10
    # y = odeint(vanderpol, yinit, times, args= (forcing, mu,), rtol=rtol, atol=atol)
    
    y = rk4(yinit, tfinal, n_steps, u=forcing, mu=mu)
    return y#,times



def draw_figure(y,t):
    plt.subplot(2,2,1)
    plt.plot(t, y[:,0],'-ob')
    plt.title("Time Profile of y1")
    plt.subplot(2,2,3)
    plt.plot(t, y[:,1],'-og')
    plt.title("Time Profile of y2")
    # plt.subplot2grid((2,2),(0,1),rowspan=2)
    # in 2x2 grid of subplots, draw in (0,1)-->upper right, 
    # and make the plot span two rows of plots (height is twice normal)
    plt.subplot2grid((2,2),(0,1),rowspan=2)
    plt.plot(y[:,0], y[:,1],'-ok')
    plt.title("Phase Portrait")

if __name__ == "__main__":
    
    cwd = os.getcwd()
    file = Path(cwd)/"Iterative_LQR (iLQR)/VanderPolOscillator/policies/control.npy"  
    time_horizon = 10
    n_steps_per_second = 20
    init = np.asarray([2, 0])

    control = np.load(file).flatten()
    control = [0.33999999999999986, 1.9, 0.14000000000000012, 0.28000000000000025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    total_n_steps = n_steps_per_second*time_horizon
    T = np.linspace(0, time_horizon, total_n_steps)
    Y = np.zeros((total_n_steps, 2))
    for i in range(time_horizon):
        y = run_vanderpol(init, forcing=control[i], mu=2)
        Y[n_steps_per_second*i:n_steps_per_second*(i+1), :] = y
        init = y[-1]

    # y = run_vanderpol(init, tfinal=1, forcing=u[0], mu=2)
    draw_figure(Y,T)
    plt.show()
    plt.plot()
