import numpy as np
import math
from main_ddp_ilqr import DDP_iLQR

class Main_POD_iLQR(DDP_iLQR):

    def __init__(self,nx,nu,Q,R):
        self.nx = nx
        self.nu = nu
        self.Q = Q
        self.R = R

    def 
