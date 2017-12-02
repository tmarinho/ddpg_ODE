import numpy as np
import scipy.integrate as integrate
from math import sin, cos
class LTI:
    def __init__(self, state_0):
        """ pos = [x,y,z] attitude = [rool,pitch,yaw]
            """
        self.state = state_0
        self.desired_state = 0.0
        self.s_dot = 0
        self.hist = []
        self.time = 0.0
        control_frequency = 100 # Hz for attitude control loop
        self.dt = 1.0 / control_frequency
        self.mag = 0
        self.freq = 1
        self.bias = 0
        self.noise_rand = np.zeros(5)


    def reset(self):
        self.state = np.random.rand()*2-0.5
        self.hist = []
        self.time = 0.0
        self.mag = np.random.randn()*1.2
        self.freq = 3*np.random.randn() + 2
        self.bias = np.random.rand()
        self.noise_rand = np.random.rand(5)

        return self.state

    def state_dot(self, state, t, u, time):
        x1 = state
        b = 1
        a = 3
        self.s_dot = 0.0
        w = self.noise_rand
        d =0.1**w[0]*sin(time*10*w[1]) + 0.3*w[2]*cos(time*w[3]*9) - 0.5*sin(w[4]*time*6 + 3)
        self.s_dot  = -a*x1 + b*u + d

        return self.s_dot

    def reward(self,s,u):
        return -12*(s)**2 - 0.2*u**2

    def update(self, u):
        #saturate u
        #u = np.clip(u,-5,5)
        out = integrate.odeint(self.state_dot, self.state, [0,self.dt], args = (u,self.time))
        self.time += self.dt
        #print out
        self.state = out[1]
        self.hist.append(np.array(self.state[0]))

    def step(self, action):
        desired_state = 0
        done = False
        #action = agent.do_action(env.state)
        self.update(action)
        state = self.state[0] - self.mag*np.sin(self.freq*self.time) -self.bias
        reward = self.reward(state, action)
        if abs(self.state) > 3 or abs(self.s_dot)>50:
            done = True

        #return self.state[0], reward, done, {}
        return state, reward, done, {}
