#!/usr/bin/env python
from __future__ import print_function

import skimage as skimage
from skimage import transform, color, exposure
from skimage.viewer import ImageViewer
import random
from random import choice
import numpy as np
from collections import deque
import time

import json
from keras.models import model_from_json
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Dense, Flatten, merge, MaxPooling2D, Input, AveragePooling2D, Lambda, Merge, Activation, Embedding
from keras.optimizers import SGD, Adam, rmsprop
from keras import backend as K

from vizdoom import DoomGame, ScreenResolution
from vizdoom import *
import itertools as it
from time import sleep
import tensorflow as tf

from networks import a2c_lstm
from A2CAgent import A2CAgent

skip_learning = True
load_model = True
model_path = './model/a2c_lstm_preprocessed.h5'

def preprocess_img(img, size):
    x, y = size
    img = np.rollaxis(img, 0, 3)    # It becomes (640, 480, 3)
    img = skimage.transform.resize(img[150:400, :], size) # Ucinamy sufit i UI gry
    img = np.resize(img[:, :, 0], (x, y, 1)) # wycinamy tylko kanał RED, bo na nim najlepiej widać pociski
    return img

if __name__ == "__main__":

    # Avoid Tensorflow eats up GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    game = DoomGame()
    game.load_config("./scenarios/take_cover.cfg")
    game.set_sound_enabled(False)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(skip_learning)
    game.init()
    print("Doom initialized!")

    # Maximum number of episodes
    max_episodes = 100000

    game.new_episode()
    game_state = game.get_state()
    misc = game_state.game_variables  # [KILLCOUNT, AMMO, HEALTH]
    prev_misc = misc

    action_size = game.get_available_buttons_size()

    img_rows , img_cols = 64, 64
    img_channels = 1 # Color Channels
    # Convert image into Black and white
    trace_length = 4 # RNN states

    state_size = (trace_length, img_rows, img_cols, img_channels)
    agent = A2CAgent(state_size, action_size, trace_length)
    agent.model = a2c_lstm(state_size, action_size, agent.value_size, agent.learning_rate)

    if load_model:
      agent.load_model(model_path)
      print("Model loaded!")

    if not skip_learning:

    # Start training
      GAME = 0
      t = 0
      max_life = 0 # Maximum episode life (Proxy for agent performance)

      # Buffer to compute rolling statistics
      life_buffer, ammo_buffer, kills_buffer = [], [], []

      for i in range(max_episodes):

          game.new_episode()
          game_state = game.get_state()
          misc = game_state.game_variables
          prev_misc = misc

          x_t = game_state.screen_buffer # 480 x 640
          x_t = preprocess_img(x_t, size=(img_rows, img_cols))
          s_t = np.stack(tuple([x_t]*trace_length), axis=0)  # It becomes 4x64x64x3
          s_t = np.expand_dims(s_t, axis=0)  # 1x4x68x64x3

          life = 0 # Episode life

          while not game.is_episode_finished():

              loss = 0 # Training Loss at each update
              r_t = 0 # Initialize reward at time t
              a_t = np.zeros([action_size]) # Initialize action at time t

              x_t = game_state.screen_buffer
              x_t = preprocess_img(x_t, size=(img_rows, img_cols))
              x_t = np.reshape(x_t, (1, 1, img_rows, img_cols, img_channels))
              s_t = np.append(s_t[:, 1:, :, :, :], x_t, axis=1) # 1x4x68x64x3

              position = game_state.game_variables[1]
              action_idx, policy  = agent.get_action(s_t)
              a_t[action_idx] = 1

              # TUTAJ pomysł, żeby ograniczać jego ruchy uniemożliwiając skręcenie w lewo/prawy gdy blisko ściany - nie działą
              # if position > 720:
              #   action_idx = 1
              #   a_t[action_idx] = 1

              # elif position < 30:
              #   action_idx = 0
              #   a_t[action_idx] = 1

              a_t = a_t.astype(int) # Sample action from stochastic softmax policy

              game.set_action(a_t.tolist())
              skiprate = agent.frame_per_action # Frame Skipping = 4
              game.advance_action(skiprate)

              r_t = game.get_last_reward()  # Each frame we get reward of 0.1, so 4 frames will be 0.4
              # Check if episode is terminated
              is_terminated = game.is_episode_finished()

              if (is_terminated):
                  # Save max_life
                  if (life > max_life):
                      max_life = life
                  life_buffer.append(life)
                  # ammo_buffer.append(misc[1])
                  # kills_buffer.append(misc[0])
                  print ("Episode Finish ", prev_misc, policy)
              else:
                  life += 1
                  game_state = game.get_state()  # Observe again after we take the action
                  misc = game_state.game_variables

              # Reward Shaping
              r_t = agent.shape_reward(r_t, misc, prev_misc, t)

              # Save trajactory sample <s, a, r> to the memory
              agent.append_sample(s_t, action_idx, r_t)

              # Update the cache
              t += 1
              prev_misc = misc

              if (is_terminated and t > agent.observe):
                  # Every episode, agent learns from sample returns
                  loss = agent.train_model()

              # Save model every 10000 iterations
              if t % 2000 == 0:
                  print("Save model")
                  agent.model.save_weights(model_path, overwrite=True)

              state = ""
              if t <= agent.observe:
                  state = "Observe mode"
              else:
                  state = "Train mode"

              if (is_terminated):

                  # Print performance statistics at every episode end
                  print("Episode", i, "/ ACTION", action_idx, "/ total reward", game.get_total_reward(), "/ LOSS", loss)

                  # Save Agent's Performance Statistics
                  if GAME % agent.stats_window_size == 0 and t > agent.observe:
                      print("Update Rolling Statistics")
                      agent.mavg_score.append(np.mean(np.array(life_buffer)))
                      agent.var_score.append(np.var(np.array(life_buffer)))
                      # agent.mavg_ammo_left.append(np.mean(np.array(ammo_buffer)))
                      # agent.mavg_kill_counts.append(np.mean(np.array(kills_buffer)))

                      # Reset rolling stats buffer
                      life_buffer, ammo_buffer, kills_buffer = [], [], []

                      # Write Rolling Statistics to file
                      with open("statistics/a2c_lstm_stats.txt", "w") as stats_file:
                          stats_file.write('Game: ' + str(GAME) + '\n')
                          stats_file.write('Max Score: ' + str(max_life) + '\n')
                          stats_file.write('mavg_score: ' + str(agent.mavg_score) + '\n')
                          stats_file.write('var_score: ' + str(agent.var_score) + '\n')
                          # stats_file.write('mavg_ammo_left: ' + str(agent.mavg_ammo_left) + '\n')
                          # stats_file.write('mavg_kill_counts: ' + str(agent.mavg_kill_counts) + '\n')

          # Episode Finish. Increment game count
          GAME += 1

    # kod do obejrzenia agenta
    if skip_learning:
      for _ in range(20):
        game.new_episode()

        while not game.is_episode_finished():
          game_state = game.get_state()

          x_t = game_state.screen_buffer # 480 x 640
          x_t = preprocess_img(x_t, size=(img_rows, img_cols))
          s_t = np.stack(tuple([x_t]*trace_length), axis=0)  # It becomes 4x64x64x3
          s_t = np.expand_dims(s_t, axis=0)

          a_t = np.zeros([action_size]) # Initialize action at time t

          x_t = game_state.screen_buffer
          position = game_state.game_variables[1]

          x_t = preprocess_img(x_t, size=(img_rows, img_cols))
          x_t = np.reshape(x_t, (1, 1, img_rows, img_cols, img_channels))
          s_t = np.append(s_t[:, 1:, :, :, :], x_t, axis=1) # 1x4x68x64x3

          action_idx, policy  = agent.get_action(s_t)
          a_t[action_idx] = 1

          a_t = a_t.astype(int)
          game.set_action(a_t.tolist())
          # skiprate = agent.frame_per_action # Frame Skipping = 4
          # for _ in range(10):
          game.advance_action()

        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)

