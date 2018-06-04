import random
import numpy as np
from random import choice
from collections import deque

class A2CAgent:

    def __init__(self, state_size, action_size, trace_length):
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.trace_length = trace_length
        self.value_size = 1
        self.observe = 0
        self.frame_per_action = 4

        # These are hyper parameters for the Policy Gradient
        self.gamma = 0.99
        self.learning_rate = 0.0001

        # create model for actor critic network
        self.model = None

        # lists for the states, actions and rewards
        self.states, self.actions, self.rewards = [], [], []

        # Performance Statistics
        self.stats_window_size= 50 # window size for computing rolling statistics
        self.mavg_score = [] # Moving Average of Survival Time
        self.var_score = [] # Variance of Survival Time
        self.mavg_ammo_left = [] # Moving Average of Ammo used
        self.mavg_kill_counts = [] # Moving Average of Kill Counts

    # using the output of policy network, pick action stochastically (Stochastic Policy)
    def get_action(self, state):
        policy = self.model.predict(state)[0].flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0], policy

    # Instead agent uses sample returns for evaluating policy
    # Use TD(1) i.e. Monte Carlo updates
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save <s, a ,r> of each step
    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

    # update policy network every episode
    def train_model(self):
        episode_length = len(self.states)

        discounted_rewards = self.discount_rewards(self.rewards)
        # Standardized discounted rewards
        discounted_rewards -= np.mean(discounted_rewards)
        if np.std(discounted_rewards):
            discounted_rewards /= np.std(discounted_rewards)
        else:
            self.states, self.actions, self.rewards = [], [], []
            print ('std = 0!')
            return 0

        state_inputs = np.zeros(((episode_length,) + self.state_size)) # Episode_lengthx4x64x64x3

        # Episode length is like the minibatch size in DQN
        for i in range(episode_length):
            state_inputs[i,:,:,:,:] = self.states[i]

        # Prediction of state values for each state appears in the episode
        values = self.model.predict(state_inputs)[1]

        # Similar to one-hot target but the "1" is replaced by Advantage Function i.e. discounted_rewards R_t - Value
        advantages = np.zeros((episode_length, self.action_size))

        for i in range(episode_length):
            advantages[i][self.actions[i]] = discounted_rewards[i] - values[i]

        loss = self.model.fit(state_inputs, [advantages, discounted_rewards], nb_epoch=1, verbose=0)

        self.states, self.actions, self.rewards = [], [], []

        return loss.history['loss']


    def shape_reward(self, r_t, misc, prev_misc, t):

        #Check any kill count
        # if (misc[0] > prev_misc[0]):
            #print ("Kill somebody")
            # r_t = r_t + 1

        # if (misc[1] < prev_misc[1]): #use ammo
            #print ("Use ammo")
            # r_t = r_t - 0.1

        if (misc[0] < prev_misc[0]): #loss HEALTH
            #print ("Loss Health")
            r_t = r_t - 0.1

        return r_t
