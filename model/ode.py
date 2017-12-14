import numpy as np
import scipy.integrate as integrate
from math import sin, cos
class ODE:
    def __init__(self, state_0):
        self.state = state_0
        self.mag = np.ones(state_0.shape)
        self.freq = np.zeros(state_0.shape)
        self.bias = np.ones(state_0.shape)
        self.time = 0.0
        self.ref = self.mag*np.sin(self.freq*self.time) + self.bias
        self.x_dot = np.zeros(2)
        self.hist = []
        self.ref_hist = []
        self.u_hist = []
        self.reset()
        control_frequency = 35 # Hz for  control loop
        self.dt = 1.0 / control_frequency
        self.noise_rand = np.zeros(5)
        self.a1 = 1
        self.a2 = 1
        self.Q      = np.array([[10, 0],[0, 0]])

    def reset(self):
        self.state = np.random.randn(*self.state.shape)*0
        self.mag = np.random.rand(*self.state.shape)*3
        self.freq = np.random.rand(*self.state.shape)*1 +0.5
        if np.random.rand()>0.5:
            self.bias = np.array([1, 0])
        else:
            self.bias = np.array([-1, 0])
        self.time = 0.0
        self.ref = self.mag*np.sin(self.freq*self.time) + self.bias
        self.ref[1] = 0
        self.hist = []
        self.ref_hist = []
        self.u_hist = []
        self.noise_rand = np.random.rand(5)

        return self.state - self.ref

    def disturbance(self,time):
        w = self.noise_rand
        return 0.1*sin(time*10*w[1]) + 0.3*w[2]*cos(time*w[3]*9) - 0.5*sin(w[4]*time*6 + 3)

    def state_dot(self, state, t, u, time):
        x1 = state[0]
        x2 = state[1]
        x1_dot = x2
        x2_dot = -x1 -x2 + u[0]

        self.x_dot  = np.array([x1_dot, x2_dot])
        return self.x_dot

    def reward(self,e,u):
        #if np.linalg.norm(e[0]) < 0.01:
        #    return 1
        #return -1
        return -e.dot(self.Q.dot(e))- 0.1*np.linalg.norm(u)**2

    def update(self, u):
        #saturate u
        #u = np.clip(u,-5,5)
        out = integrate.odeint(self.state_dot, self.state, [0,self.dt], args = (u,self.time))
        self.state = out[1]
        self.time += self.dt
        #self.state = self.d_sys(self.state,u)
        self.hist.append(np.array(self.state))
        self.u_hist.append(u[0])

    def step(self, action):
        self.ref = self.mag*np.sin(2*self.freq*self.time) + self.bias
        self.ref[1]= 0
        done = False
        self.update(action)
        error = self.state - self.ref
        reward = self.reward(error, action)
        self.ref_hist.append(np.array(self.ref))
        if abs(np.linalg.norm(error[0])) > 5:
            done = True

        return error, reward, done, {}

    def d_sys(self,state,u):
        x1 = state[0]
        x2 = state[1]
        x1n = x1 + 0.051200000000000*x2 + u[0]*1.25e-05
        x2n = x2 + 0.007812500000000*u[0]

        return np.array([x1n, x2n])
