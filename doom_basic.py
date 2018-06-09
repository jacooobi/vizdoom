import numpy as np
import skimage.color
import skimage.transform

from vizdoom import *
from random import sample, random, randint
from networks import dqn
from itertools import product
from tqdm import trange

learning_rate = 0.00025
discount_factor = 0.99
epochs = 20
learning_steps_per_epoch = 2000
replay_memory_size = 10000

batch_size = 32
frame_repeat = 12
# resolution = (30, 45)
resolution = (64, 64)
episodes_to_watch = 10

model_savefile = '/tmp/model.ckpt'
save_model = True
load_model = False
skip_learning = False
config_file_path = './scenarios/take_cover.cfg'


def resize_img(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img


class ReplayMemory:
    def __init__(self, capacity):
        channels = 1
        state_shape = (capacity, resolution[0], resolution[1], channels)

        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)

        self.is_terminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, is_terminal, reward):
        self.s1[self.pos, :, :, 0] = s1
        self.a[self.pos] = action

        if not is_terminal:
            self.s2[self.pos, :, :, 0] = s2

        self.is_terminal[self.pos] = is_terminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return (self.s1[i],
                self.a[i],
                self.s2[i],
                self.is_terminal[i],
                self.r[i])


def learn(model, s1, target_q):
    loss = model.fit(inputs=s1, outputs=target_q, epochs=1)
    return loss


# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game


def get_q_values(model, s):
    return model.predict(s)


def learn_from_memory(memory, model):
    if memory.size > batch_size:
        s1, a, s2, is_terminal, r = memory.get_sample(batch_size)

        print(s1.shape, s2.shape)

        q2 = np.max(model.predict(s2), axis=1)
        target_q = model.predict(s1)

        # target differs from q only for the selected action.
        # The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if is_terminal else r
        target_q[np.arange(target_q.shape[0]), a] = r + \
            discount_factor * (1 - is_terminal) * q2

        learn(model, s1, target_q)


def perform_learning_step(epoch, memory, model, game):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * epochs  # 10% of learning time
        eps_decay_epochs = 0.6 * epochs  # 60% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * \
                (start_eps - end_eps)
        else:
            return end_eps

    s1 = resize_img(game.get_state().screen_buffer)

    print(s1.shape)

    # With probability eps make a random action.
    eps = exploration_rate(epoch)
    if random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        # Choose the best action according to the network.
        a = get_best_action(s1)
    reward = game.make_action(actions[a], frame_repeat)

    is_terminal = game.is_episode_finished()
    s2 = resize_img(
        game.get_state().screen_buffer) if not is_terminal else None

    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, is_terminal, reward)

    learn_from_memory(memory, model)


def main():
    global actions

    game = initialize_vizdoom(config_file_path)

    n = game.get_available_buttons_size()
    actions = [list(a) for a in product([0, 1], repeat=n)]

    input_shape = (resolution[0], resolution[1], 1)
    print('Input shape = {}'.format(input_shape))
    memory = ReplayMemory(capacity=replay_memory_size)
    model = dqn(input_shape, n, learning_rate)

    print('Start training')
    for epoch in range(epochs):
        game.new_episode()

        train_episodes_finished = 0
        train_scores = []

        for learning_step in trange(learning_steps_per_epoch, leave=False):
            perform_learning_step(epoch, memory, model, game)

            if game.is_episode_finished():
                score = game.get_total_reward()
                train_scores.append(score)
                game.new_episode()
                train_episodes_finished += 1

        train_scores = np.array(train_scores)

        print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()),
              "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

    print('Training finished!')
    game.close()


if __name__ == '__main__':
    main()
