import numpy as np
import scipy.integrate as integrate
from math import sin, cos
class LTI:
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
        self.reset()
        control_frequency = 100 # Hz for attitude control loop
        self.dt = 1.0 / control_frequency
        self.noise_rand = np.zeros(5)
        self.a1 = 1
        self.a2 = 1
        self.Q      = np.array([[10, 0],[0, 0]])

    def reset(self):
        self.state = np.random.randn(*self.state.shape)*1.5
        self.mag = np.random.randn(*self.state.shape)*0.5
        self.freq = np.random.rand(*self.state.shape)*1 +0.5
        self.bias = np.random.randn(*self.state.shape)
        self.time = 0.0
        self.ref = self.mag*np.sin(self.freq*self.time) + self.bias
        self.ref[1] = 0
        self.hist = []
        self.ref_hist = []
        self.noise_rand = np.random.rand(5)

        return self.state - self.ref

    def disturbance(self,time):
        w = self.noise_rand
        return 0.1*sin(time*10*w[1]) + 0.3*w[2]*cos(time*w[3]*9) - 0.5*sin(w[4]*time*6 + 3)

    def state_dot(self, state, t, u, time):
        x1 = state[0]
        x2 = state[1]
        #x1_dot = x2 + u[0] + x2**2
        #x2_dot = -self.a1*x1-self.a2*x2 + u[1]# + self.disturbance(time)
        x1_dot =  u[0]  +0.*sin(self.disturbance(self.time))
        x2_dot =  u[1] + 0*self.disturbance(self.time)

        self.x_dot  = np.array([x1_dot, x2_dot])

        return self.x_dot

    def reward(self,e,u):
        #if e < 1.2*np.exp(-0.2*self.time)+0.1:
        #    return 1
        #return 0
        return -e.dot(self.Q.dot(e))- 0.001*np.linalg.norm(u)**2
        #return 1/(e.dot(self.Q.dot(e))+1)

    def update(self, u):
        #saturate u
        #u = np.clip(u,-5,5)
        out = integrate.odeint(self.state_dot, self.state, [0,self.dt], args = (u,self.time))
        self.time += self.dt
        self.state = out[1]
        self.hist.append(np.array(self.state))

    def step(self, action):
        self.ref =  self.mag*np.sin(self.freq*self.time) + self.bias
        self.ref[1] = 0
        #self.ref[0] = 0
        done = False
        self.update(action)
        error = self.state - self.ref
        reward = self.reward(error, action)
        self.ref_hist.append(np.array(self.ref))
        if abs(np.linalg.norm(error[0])) > 10:
            done = True

        return error, reward, done, {}
