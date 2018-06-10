import random
import numpy as np
import argparse

import skimage as skimage
from skimage import transform, color, exposure
from skimage.io import imsave

import tensorflow as tf
from keras import backend as K

from vizdoom import DoomGame, ScreenResolution
from vizdoom import *

from DoubleDQNAgent import DoubleDQNAgent
from networks import dueling_dqn


MODEL_SAVING_INTERVAL = 5000


def preprocess_img(img, size):
    x, y = size
    img = np.rollaxis(img, 0, 3)    # It becomes (640, 480, 3)
    img = skimage.transform.resize(
        img[150:400, :], size)  # Ucinamy sufit i UI gry

    # wycinamy tylko kanał RED, bo na nim najlepiej widać pociski
    img = np.resize(img[:, :, 0], (x, y, 1))
    img = np.squeeze(img, axis=2)

    # imsave('./test.png', img)

    return img


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./model/dueling_ddqn.h5',
                        help='Save model path')
    parser.add_argument(
        '--scenario', default='./scenarios/take_cover_slow_imps.cfg')
    parser.add_argument('--load-model', action='store_true', help='Load model')
    parser.add_argument('--show-window', action='store_true',
                        help='Force showing window')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Avoid Tensorflow eats up GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    args = get_args()
    model_path = args.model
    load_model = args.load_model
    show_window = args.show_window
    scenario = args.scenario

    game = DoomGame()
    game.load_config(scenario)
    game.set_sound_enabled(False)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(show_window)
    game.init()

    game.new_episode()
    game_state = game.get_state()
    misc = game_state.game_variables  # [HEALTH] - for take_cover
    prev_misc = misc

    action_size = game.get_available_buttons_size()

    # Convert image into Black and white
    img_rows, img_cols = 64, 64
    img_channels = 4  # We stack 4 frames

    state_size = (img_rows, img_cols, img_channels)
    agent = DoubleDQNAgent(state_size, action_size)

    agent.model = dueling_dqn(
        state_size, action_size, agent.learning_rate)
    agent.target_model = dueling_dqn(
        state_size, action_size, agent.learning_rate)

    if load_model:
        print('Loading model: {}'.format(model_path))
        agent.load_model(model_path)

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

        # each frame we get reward of 2.5, so 4 frames will be 10.0
        r_t = game.get_last_reward()

        if is_terminated:
            # It's just for agent performance
            if life > max_life:
                max_life = life

            GAME += 1
            life_buffer.append(life)
            print('Episode Finish. r_t = {} '.format(r_t), misc)

            game.new_episode()
            game_state = game.get_state()
            misc = game_state.game_variables
            x_t1 = game_state.screen_buffer

        x_t1 = game_state.screen_buffer
        misc = game_state.game_variables

        x_t1 = preprocess_img(x_t1, size=(img_rows, img_cols))
        x_t1 = np.reshape(x_t1, (1, img_rows, img_cols, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        r_t = agent.shape_reward(r_t, misc, prev_misc, t)

        if is_terminated:
            life = 0
        else:
            life += 1

        # Update the cache
        prev_misc = misc

        # save the sample <s, a, r, s'> to the replay memory and decrease epsilon
        agent.replay_memory(s_t, action_idx, r_t, s_t1, is_terminated, t)

        # Do the training
        if t > agent.observe and t % agent.timestep_per_train == 0:
            Q_max, loss = agent.train_replay()
            print('[Training] Q_max = {}, Loss = {}'.format(Q_max, loss))

        s_t = s_t1
        t += 1

        # save progress every 5000 iterations
        if t % MODEL_SAVING_INTERVAL == 0:
            print('Saving model at {}'.format(model_path))
            agent.model.save_weights(model_path, overwrite=True)

        # print info
        state = ""
        if t <= agent.observe:
            state = "observe"
        elif t > agent.observe and t <= agent.observe + agent.explore:
            state = "explore"
        else:
            state = "train"

        if is_terminated:
            # print("TIME", t, "/ GAME", GAME, "/ STATE", state,
            #       "/ EPSILON", agent.epsilon, "/ ACTION", action_idx, "/ REWARD", r_t,
            #       "/ Q_MAX %e" % np.max(Q_max), "/ LIFE", max_life, "/ LOSS", loss)

            # Save Agent's Performance Statistics
            if GAME % agent.stats_window_size == 0 and t > agent.observe:
                print("Update Rolling Statistics")
                agent.mavg_score.append(np.mean(np.array(life_buffer)))
                agent.var_score.append(np.var(np.array(life_buffer)))
                agent.mavg_ammo_left.append(np.mean(np.array(ammo_buffer)))
                agent.mavg_kill_counts.append(np.mean(np.array(kills_buffer)))

                # Reset rolling stats buffer
                life_buffer = []

                # Write Rolling Statistics to file
                with open("statistics/dueling_ddqn_stats.txt", "w") as stats_file:
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
