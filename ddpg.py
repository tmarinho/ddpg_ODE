from model.quadcopter import Quadcopter
from model.lti import LTI

from MarkoC import StochSwitch
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
from time import sleep
#from keras.engine.training import collect_trainable_weights
import json
import model.params as params

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit
import matplotlib.pyplot as plt

mode = 0        #Quadrotor = 1 LTI = 0
OU = OU()       #Ornstein-Uhlenbeck Process
noise_toggle = 1

klqr = np.array([13.1774, 4.2302])
def playGame(train_indicator=1):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 1000000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0002   #Learning rate for Actor
    LRC = 0.002     #Lerning rate for Critic
    lqr_toggle = 0
    if mode==1:
        action_dim = 3  #Steering/Acceleration/Brake
        state_dim  = 13  #of sensors input
    else:
        action_dim = 2  #Steering/Acceleration/Brake
        state_dim  = 2  #of sensors input

    #np.random.seed(1337)

    vision = False

    EXPLORE = 100000.
    episode_count = 4000
    max_steps = 5000
    reward =-100
    done = False
    step = 0
    epsilon = 1
    indicator = 0

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer


    if mode==1:
        env =  Quadcopter()
    else:
        env = LTI(np.zeros(state_dim))
    #Now load the weight
    try:
        actor.model.load_weights("actormodel.h5")
        critic.model.load_weights("criticmodel.h5")
        actor.target_model.load_weights("actormodel.h5")
        critic.target_model.load_weights("criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("I wish you good luck!")
    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

#        if np.mod(i, 3) == 0:
#            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
#        else:
        ob = env.reset()
        s_t = np.asarray(ob)[:, None].T #TODO increase for more states
        #print s_t
        #print  actor.model.predict(s_t)
#        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        #CTMC.reset()
        total_reward = 0.
        for j in range(max_steps):
            loss = 0
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])

            #a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            #s_t2 =np.expand_dims(s_t,axis = 1)
            a_t_original = actor.model.predict(s_t)
            #print a_t_original[0], s_t
            noise_t[0][0] = noise_toggle* train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0], 0 , 0.0,10 )
            noise_t[0][1] = noise_toggle*train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.2, 1.00, 10)
            if mode==1:
                noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.2, 1.00, 0.1)
                noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], 0 , 1.00, 0.1)
                noise_t[0][3] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][3], 0, 1.00, 500)

            #The following code do the stochastic brake
            #if random.random() <= 0.1:
            #    print("********Now we apply the brake***********")
            #    noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)
            # if np.mod(j, 50) == 0:
            #     CTMC.update_trans(env.time)
            #     CTMC.sample()
            #     if CTMC.x == 0:
            #         a_t = a_t_original/a_t_original + noise
            #     else:
            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            if mode==1:
                a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
                a_t[0][2] = a_t_original[0][2] + noise_t[0][2]
            if lqr_toggle ==1:
                a_t[0][0] = -klqr.dot(np.asarray(s_t[0]))
            #a_t[0][3] = a_t_original[0][1] + noise_t[0][3]

            ob, r_t, done, info = env.step(a_t[0])
            s_t1 = np.asarray(ob)[:, None].T #TODO increase for more states
            #r_t = np.asarray([r_t])
            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer

            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0][0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3][0] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]

            if (train_indicator):
                loss += critic.model.train_on_batch([states,actions], y_t)
                a_for_grad = actor.model.predict(states)
                #print states
                #print a_for_grad
                if lqr_toggle:
                    a_for_grad[0] = klqr.dot(states[0])
                    #print a_for_grad
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1

            if 0:
                epsilon = 1
                lqr_toggle = 0
                print 'I am here'
            #print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)

            step += 1
            if done:
                break
        if np.mod(i, 1) == 0:
            plt.close()
            hist = np.asarray(env.hist)
            #print(hist[0])


            if mode ==1:
                fig = plt.figure()
                ax1 = fig.add_subplot(121)
                ax1.plot(hist[:,0],'b')
                ax1.plot(hist[:,1],'r')
                        #plt.plot(hist[:,1],'r')
                ax2 = fig.add_subplot(122)
                ax2.plot(hist[:,2],'g')
            else:
                rhist = np.asarray(env.ref_hist)
                plt.plot(hist[:,0],'b')
                plt.plot(rhist[:,0], 'b-.')
                #plt.plot(rhist[:,1], 'r-.')
                #plt.ylim([-1, 2])
            plt.show(block=False)
            #plt.draw()
        if np.mod(i, 3) == 0:

            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("actormodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        #print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        #print("Total Step: " + str(step))
        #print("")

    #env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    playGame()
