#Import Lubraies
from model.ode import ODE
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
from time import sleep
import json
import model.params as params
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit
import matplotlib.pyplot as plt



noise_toggle = 1
hist_reward =[]
hist_rt = []
klqr = np.array([11.049875621120901, 3.806220057617194])
def RunSim(train_toggle):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 1000000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001
    LRA = 0.0002   #Actor Learning
    LRC = 0.002    #Critic Learning

    #Sim options
    lqr_toggle = 0
    action_dim = 1
    state_dim  = 2
    hist_rt = []
    hist_reward =[]

    d_exploration = 200000.
    num_max_episodes = 250
    max_seconds = 15
    reward =-100
    done = False
    step = 0
    epsilon = 1
    Noise_magnitude = 15


    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    # Initialize Actor and Critic Networks
    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)

    #Create replay buffer
    buff = ReplayBuffer(BUFFER_SIZE)

    #Load the ODE environment
    env = ODE(np.zeros(state_dim))

    # Load network parameters from previous training
    try:
        actor.model.load_weights("actormodel.h5")
        critic.model.load_weights("criticmodel.h5")
        actor.target_model.load_weights("actormodel.h5")
        critic.target_model.load_weights("criticmodel.h5")
    except:
        print("Weight Error")

    print("Taining state is ",train_toggle)


    for i in range(num_max_episodes):
        print("Current episode : " + str(i))
        # Before every episote completly reset the envrionment
        ob = env.reset()
        s_t = np.asarray(ob)[:, None].T
        total_reward = 0.
        max_steps = int(max_seconds/env.dt)

        # Run the episode
        for j in range(max_steps):
            loss = 0
            epsilon -= 1.0 / d_exploration
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])
            a_t_original = actor.model.predict(s_t)
            noise_t[0] = noise_toggle* train_toggle * max(epsilon, 0.05) * Noise_magnitude * np.random.randn(action_dim)
            a_t[0] = a_t_original[0] + noise_t[0]
            #a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            if lqr_toggle ==1:
                a_t[0] = -klqr.dot(np.asarray(s_t[0]))
            ob, r_t, done, info = env.step(a_t[0])
            s_t1 = np.asarray(ob)[:, None].T
            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer
            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0][0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3][0] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            #Output from target Networks
            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])
            #TD
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]

            if (train_toggle):
                loss += critic.model.train_on_batch([states,actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            hist_rt.append(r_t)
            s_t = s_t1

            step += 1
            if done:
                break
        hist_reward.append(np.mean(hist_rt))
        hist_rt = []
        if np.mod(i, 1) == 0:
            plt.close()
            hist = np.asarray(env.hist)
            u = np.asarray(env.u_hist)
            fig = plt.figure(figsize=(15,5))
            plt.suptitle('Episode '+str(i))
            ax1 = fig.add_subplot(121)
            rhist = np.asarray(env.ref_hist)
            time = np.linspace(0,j*env.dt,num=j+1)
            ax1.plot(time,hist[:,0],'b',label='Output')
            ax1.plot(time,rhist[:,0], 'b-.',label='Reference')
            ax1.set_ylabel('y(t)')
            ax1.set_xlabel('t')
            ax2 = fig.add_subplot(122)
            ax2.plot(time,u[:],'g')
            ax2.set_ylabel('Control Signal u(t)')
            ax2.set_xlabel('t')

            #plt.show(block=False)
            fig.savefig('figures/results'+str(i)+'.pdf')

            fig2 = plt.figure(figsize=(5,5))
            plt.title('Mean Reward Per Espisode')
            plt.plot(hist_reward,'go')
            ymax = 100
            #plt.ylim(-100, 0)
            #plt.show(block=False)
            fig2.savefig('figures/reward.pdf')
            print 'reward',total_reward

        if np.mod(i, 10) == 0:
            if (train_toggle):
                print("Now we save model")
                actor.model.save_weights("actormodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)



if __name__ == "__main__":
    RunSim(1)
