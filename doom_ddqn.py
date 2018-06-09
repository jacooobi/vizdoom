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
from keras import backend as K

from vizdoom import DoomGame, ScreenResolution
from vizdoom import *
import itertools as it
from time import sleep
import tensorflow as tf

from networks import dqn
from DoubleDQNAgent import DoubleDQNAgent


def preprocess_img(img, size):
    img = np.rollaxis(img, 0, 3)    # It becomes (640, 480, 3)
    img = skimage.transform.resize(img, size)
    img = skimage.color.rgb2gray(img)
    print("shape", img.shape)
    return img


if __name__ == "__main__":
    # Avoid Tensorflow eats up GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    game = DoomGame()
    game.load_config("./scenarios/defend_the_center.cfg")
    game.set_sound_enabled(False)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(False)
    game.init()

    game.new_episode()
    game_state = game.get_state()
    misc = game_state.game_variables  # [KILLCOUNT, AMMO, HEALTH]
    prev_misc = misc

    action_size = game.get_available_buttons_size()

    img_rows, img_cols = 64, 64
    # Convert image into Black and white
    img_channels = 4  # We stack 4 frames

    state_size = (img_rows, img_cols, img_channels)
    agent = DoubleDQNAgent(state_size, action_size)

    agent.model = dqn(state_size, action_size, agent.learning_rate)
    agent.target_model = dqn(state_size, action_size, agent.learning_rate)

    x_t = game_state.screen_buffer  # 480 x 640
    x_t = preprocess_img(x_t, size=(img_rows, img_cols))
    s_t = np.stack(([x_t] * 4), axis=2)  # It becomes 64x64x4
    s_t = np.expand_dims(s_t, axis=0)  # 1x64x64x4

    is_terminated = game.is_episode_finished()

    # Start training
    epsilon = agent.initial_epsilon
    GAME = 0
    t = 0
    max_life = 0  # Maximum episode life (Proxy for agent performance)
    life = 0

    # Buffer to compute rolling statistics
    life_buffer, ammo_buffer, kills_buffer = [], [], []

    while not game.is_episode_finished():

        loss = 0
        Q_max = 0
        r_t = 0
        a_t = np.zeros([action_size])

        # Epsilon Greedy
        action_idx = agent.get_action(s_t)
        a_t[action_idx] = 1

        a_t = a_t.astype(int)
        game.set_action(a_t.tolist())
        skiprate = agent.frame_per_action
        game.advance_action(skiprate)

        game_state = game.get_state()  # Observe again after we take the action
        is_terminated = game.is_episode_finished()

        # each frame we get reward of 0.1, so 4 frames will be 0.4
        r_t = game.get_last_reward()

        if (is_terminated):
            if (life > max_life):
                max_life = life
            GAME += 1
            life_buffer.append(life)
            ammo_buffer.append(misc[1])
            kills_buffer.append(misc[0])
            print("Episode Finish ", misc)
            game.new_episode()
            game_state = game.get_state()
            misc = game_state.game_variables
            x_t1 = game_state.screen_buffer

        x_t1 = game_state.screen_buffer
        misc = game_state.game_variables

        x_t1 = preprocess_img(x_t1, size=(img_rows, img_cols))
        x_t1 = np.reshape(x_t1, (1, img_rows, img_cols, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        r_t = agent.shape_reward_dqn(r_t, misc, prev_misc, t)

        if (is_terminated):
            life = 0
        else:
            life += 1

        # Update the cache
        prev_misc = misc[:]

        # save the sample <s, a, r, s'> to the replay memory and decrease epsilon
        agent.replay_memory(s_t, action_idx, r_t, s_t1, is_terminated, t)

        # Do the training
        if t > agent.observe and t % agent.timestep_per_train == 0:
            Q_max, loss = agent.train_replay()

        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            print("Now we save model")
            agent.model.save_weights("model/ddqn.h5", overwrite=True)

        # print info
        state = ""
        if t <= agent.observe:
            state = "observe"
        elif t > agent.observe and t <= agent.observe + agent.explore:
            state = "explore"
        else:
            state = "train"

        if (is_terminated):
            print("TIME", t, "/ GAME", GAME, "/ STATE", state,
                  "/ EPSILON", agent.epsilon, "/ ACTION", action_idx, "/ REWARD", r_t,
                  "/ Q_MAX %e" % np.max(Q_max), "/ LIFE", max_life, "/ LOSS", loss)

            # Save Agent's Performance Statistics
            if GAME % agent.stats_window_size == 0 and t > agent.observe:
                print("Update Rolling Statistics")
                agent.mavg_score.append(np.mean(np.array(life_buffer)))
                agent.var_score.append(np.var(np.array(life_buffer)))
                agent.mavg_ammo_left.append(np.mean(np.array(ammo_buffer)))
                agent.mavg_kill_counts.append(np.mean(np.array(kills_buffer)))

                # Reset rolling stats buffer
                life_buffer, ammo_buffer, kills_buffer = [], [], []

                # Write Rolling Statistics to file
                with open("statistics/ddqn_stats.txt", "w") as stats_file:
                    stats_file.write('Game: ' + str(GAME) + '\n')
                    stats_file.write('Max Score: ' + str(max_life) + '\n')
                    stats_file.write('mavg_score: ' +
                                     str(agent.mavg_score) + '\n')
                    stats_file.write(
                        'var_score: ' + str(agent.var_score) + '\n')
                    stats_file.write('mavg_ammo_left: ' +
                                     str(agent.mavg_ammo_left) + '\n')
                    stats_file.write('mavg_kill_counts: ' +
                                     str(agent.mavg_kill_counts) + '\n')
