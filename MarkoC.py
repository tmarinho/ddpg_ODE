import numpy as np
from scipy.linalg import expm

class StochSwitch:
    #r1 : rate of transition
    def _init_(self,r1=0.1,r2=0.01):
        self.P      = np.eye(2)
        self.x      = 0
        self.Q      = np.array([[-r1, r1],[r2 -r2]])
    def reset(self):
        self.P = np.eye(2)
        self.x      = 0

    def sample(self):
        j = int(not self.x)
        s = np.random.rand()
        if s < self.P[self.x,j]:
            self.x = j

    def update_trans(self,t):
        self.P = expm(self.Q*t)
