import numpy as np
import scipy.integrate as integrate
from utils.quaternion import Quaternion
from utils.utils import RPYToRot, RotToQuat, RotToRPY
import model.params as params

class Quadcopter:
    """ Quadcopter class

    state  - 1 dimensional vector but used as 13 x 1. [x, y, z, xd, yd, zd, qw, qx, qy, qz, p, q, r]
             where [qw, qx, qy, qz] is quternion and [p, q, r] are angular velocity [roll_dot, pitch_dot, yaw_dot]
    F      - 1 x 1, thrust output from controller
    M      - 3 x 1, moments output from controller
    params - system parameters struct, arm_length, g, mass, etc.
    """

    def __init__(self, pos=np.zeros(3), attitude=np.zeros(3)):
        """ pos = [x,y,z] attitude = [rool,pitch,yaw]
            """
        self.state = np.zeros(13)
        roll, pitch, yaw = attitude
        rot    = RPYToRot(roll, pitch, yaw)
        quat   = RotToQuat(rot)
        self.state[0] = pos[0]
        self.state[1] = pos[1]
        self.state[2] = pos[2]
        self.state[6] = quat[0]
        self.state[7] = quat[1]
        self.state[8] = quat[2]
        self.state[9] = quat[3]
        self.hist = []
        self.R = rot
        # Mods
        self.time = 0.0
        self.ref = np.zeros(self.state.shape)
        self.reset()
        self.Q      = np.zeros((13,13))
        self.Q[0,0] = 1
        self.Q[1,1] = 1
        self.Q[2,2] = 5
        self.Ru      = np.eye(3)
        self.Ru[0,0] = 0
        #SetupSim
        control_frequency = 250 # Hz for attitude control loop
        self.dt = 1.0 / control_frequency

    def reset(self):
        self.state = np.zeros(13)
        roll, pitch, yaw = 0.0,0.0,0.0
        rot    = RPYToRot(roll, pitch, yaw)
        quat   = RotToQuat(rot)
        self.state[0] = 0
        self.state[1] = 0
        self.state[2] = 0
        self.state[6] = quat[0]
        self.state[7] = quat[1]
        self.state[8] = quat[2]
        self.state[9] = quat[3]
        self.hist = []
        self.R = rot
        # Mods
        self.time = 0.0
        self.ref = np.zeros(self.state.shape)
        return self.state - self.ref
    def world_frame(self):
        """ position returns a 3x6 matrix
            where row is [x, y, z] column is m1 m2 m3 m4 origin h
            """
        origin = self.state[0:3]
        quat = Quaternion(self.state[6:10])
        rot = quat.as_rotation_matrix()
        wHb = np.r_[np.c_[rot,origin], np.array([[0, 0, 0, 1]])]
        quadBodyFrame = params.body_frame.T
        quadWorldFrame = wHb.dot(quadBodyFrame)
        world_frame = quadWorldFrame[0:3]
        return world_frame

    def position(self):
        return self.state[0:3]

    def velocity(self):
        return self.state[3:6]

    def attitude(self):
        rot = Quaternion(self.state[6:10]).as_rotation_matrix()
        return RotToRPY(rot)

    def omega(self):
        return self.state[10:13]

    def state_dot(self, state, t, F, M):
        x, y, z, xdot, ydot, zdot, qw, qx, qy, qz, p, q, r = state
        quat = np.array([qw,qx,qy,qz])
        bRw = Quaternion(quat).as_rotation_matrix() # world to body rotation matrix
        self.R = bRw
        wRb = bRw.T # orthogonal matrix inverse = transpose
        # acceleration - Newton's second law of motion
        accel = 1.0 / params.mass * (wRb.dot(np.array([[0, 0, F]]).T)
                    - np.array([[0, 0, params.mass * params.g]]).T)
        # angular velocity - using quternion
        # http://www.euclideanspace.com/physics/kinematics/angularvelocity/
        K_quat = 2.0; # this enforces the magnitude 1 constraint for the quaternion
        quaterror = 1.0 - (qw**2 + qx**2 + qy**2 + qz**2)
        qdot = (-1.0/2) * np.array([[0, -p, -q, -r],
                                    [p,  0, -r,  q],
                                    [q,  r,  0, -p],
                                    [r, -q,  p,  0]]).dot(quat) + K_quat * quaterror * quat;

        # angular acceleration - Euler's equation of motion
        # https://en.wikipedia.org/wiki/Euler%27s_equations_(rigid_body_dynamics)
        omega = np.array([p,q,r])
        pqrdot = params.invI.dot( M.flatten() - np.cross(omega, params.I.dot(omega)) )
        state_dot = np.zeros(13)
        state_dot[0]  = xdot
        state_dot[1]  = ydot
        state_dot[2]  = zdot
        state_dot[3]  = accel[0]
        state_dot[4]  = accel[1]
        state_dot[5]  = accel[2]
        state_dot[6]  = qdot[0]
        state_dot[7]  = qdot[1]
        state_dot[8]  = qdot[2]
        state_dot[9]  = qdot[3]
        state_dot[10] = pqrdot[0]
        state_dot[11] = pqrdot[1]
        state_dot[12] = pqrdot[2]
        return state_dot

    def reward(self,e,u):
        return -e.dot(self.Q.dot(e))- 3*u.dot(self.Ru.dot(u))

    def update(self, F, M):
        # limit thrust and Moment
        self.time += self.dt
        L = params.arm_length
        r = params.r
        prop_thrusts = params.invA.dot(np.r_[np.array([[F]]), M])
        prop_thrusts_clamped = np.maximum(np.minimum(prop_thrusts, params.maxF/4), params.minF/4)
        F = np.sum(prop_thrusts_clamped)
        M = params.A[1:].dot(prop_thrusts_clamped)
        self.state = integrate.odeint(self.state_dot, self.state, [0,self.dt], args = (F, M))[1]
        self.hist.append(np.array([self.state[0],self.state[1],self.state[2]]))

    def step(self, action):
        F = action[0];
        M = np.array([[action[1],action[2],action[3]]]).T
        #M = np.array([[action[1],action[2],0.0]]).T
        done = False
        self.ref[2] = 10
        self.update(F,M)
        error = self.state - self.ref
        reward = self.reward(error, action)
        phi, theta, psi = self.attitude()
        #if abs(phi)> 1.5 or abs(theta) >1.5:
        #    done = True
        if abs(np.linalg.norm(error[2])) > 30:
            done = True
        return error, reward, done, {}
