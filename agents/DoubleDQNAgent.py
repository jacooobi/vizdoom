import random
import os
import numpy as np
from random import choice
from collections import deque
from keras.callbacks import CSVLogger


class DoubleDQNAgent:

    def __init__(self, state_size, action_size):

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # these is hyper parameters for the Double DQN
        self.gamma = 0.99
        self.learning_rate = 0.0001
        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.0001
        self.batch_size = 32
        self.observe = 0
        self.explore = 5000
        self.frame_per_action = 4
        self.update_target_freq = 3000
        self.timestep_per_train = 100  # Number of timesteps between training interval

        # create replay memory using deque
        self.memory = deque(maxlen=2000)
        self.max_memory = 50000  # number of previous transitions to remember

        # create main model and target model
        self.model = None
        self.target_model = None

        # Performance Statistics
        self.stats_window_size = 50  # window size for computing rolling statistics
        self.mavg_score = []  # Moving Average of Survival Time
        self.var_score = []  # Variance of Survival Time
        self.mavg_ammo_left = []  # Moving Average of Ammo used
        self.mavg_kill_counts = []  # Moving Average of Kill Counts

    def update_target_model(self):
        """
        After some time interval update the target model to be same with model
        """
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        """
        Get action from model using epsilon-greedy policy
        """
        if np.random.rand() <= self.epsilon:
            action_idx = random.randrange(self.action_size)
        else:
            q = self.model.predict(state)
            action_idx = np.argmax(q)
        return action_idx

    def shape_reward(self, r_t, misc, prev_misc, t):
        # Check any kill count
        if misc[0] < 100:  # Loss HEALTH
            r_t = -100.0

        # Calculate negative reward for distance from center
        map_width = 768
        map_center = map_width / 2.0
        pos_y = misc[1]
        r_t = r_t - abs(map_center - pos_y) / map_center * 5.0

        # print('Distance from center negative reward: {:2.4f}'.format(
        #     abs(map_center - pos_y) / map_center * 10.0))

        return r_t

    def shape_reward_simple(self, r_t, misc, prev_misc, t):
        return r_t

    # Save trajectory sample <s,a,r,s'> to the replay memory
    def replay_memory(self, s_t, action_idx, r_t, s_t1, is_terminated, t):
        self.memory.append((s_t, action_idx, r_t, s_t1, is_terminated))

        if self.epsilon > self.final_epsilon and t > self.observe:
            self.epsilon -= (self.initial_epsilon -
                             self.final_epsilon) / self.explore

        if len(self.memory) > self.max_memory:
            self.memory.popleft()

        # Update the target model to be same with model
        if t % self.update_target_freq == 0:
            self.update_target_model()

    # Pick samples randomly from replay memory (with batch_size)
    def train_replay(self):
        num_samples = min(self.batch_size *
                          self.timestep_per_train, len(self.memory))
        replay_samples = random.sample(self.memory, num_samples)

        update_input = np.zeros(((num_samples,) + self.state_size))
        update_target = np.zeros(((num_samples,) + self.state_size))
        action, reward, done = [], [], []

        for i in range(num_samples):
            update_input[i, :, :, :] = replay_samples[i][0]
            action.append(replay_samples[i][1])
            reward.append(replay_samples[i][2])
            update_target[i, :, :, :] = replay_samples[i][3]
            done.append(replay_samples[i][4])

        target = self.model.predict(update_input)
        target_val = self.model.predict(update_target)
        target_val_ = self.target_model.predict(update_target)

        for i in range(num_samples):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(target_val[i])
                target[i][action[i]] = reward[i] + \
                    self.gamma * (target_val_[i][a])

        loss = self.model.fit(update_input, target,
                              batch_size=self.batch_size, epochs=1, verbose=0)

        # Q_max, loss
        return np.max(target[-1]), loss.history['loss'][0]

    # load the saved model
    def load_model(self, name):
        self.model.load_weights(name)

    # save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)
