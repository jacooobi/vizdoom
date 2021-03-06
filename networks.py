from keras.models import model_from_json
from keras.models import load_model, Model
from keras.layers.wrappers import TimeDistributed
from keras.layers.merge import add
from keras.layers import Flatten, Dense, Lambda, Input, Conv2D
from keras.optimizers import Adam
from keras.layers.recurrent import LSTM, GRU
from keras import backend as K


def dueling_dqn(input_shape, action_size, learning_rate):
    state_input = Input(shape=(input_shape))
    x = Conv2D(32, (8, 8), strides=(4, 4),
               activation='relu')(state_input)
    x = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Flatten()(x)

    # state value tower - V
    state_value = Dense(256, activation='relu')(x)
    state_value = Dense(1, kernel_initializer='uniform')(state_value)
    state_value = Lambda(lambda s: K.expand_dims(
        s[:, 0], axis=-1), output_shape=(action_size,))(state_value)

    # action advantage tower - A
    action_advantage = Dense(256, activation='relu')(x)
    action_advantage = Dense(action_size)(action_advantage)
    action_advantage = Lambda(lambda a: a[:, :] - K.mean(
        a[:, :], keepdims=True), output_shape=(action_size,))(action_advantage)

    # merge to state-action value function Q
    state_action_value = add([state_value, action_advantage])

    model = Model(inputs=state_input, outputs=state_action_value)
    # model.compile(rmsprop(lr=learning_rate), "mse")
    adam = Adam(lr=learning_rate)
    model.compile(loss='mse', optimizer=adam)

    return model


def a2c_lstm(input_shape, action_size, value_size, learning_rate):
    """Actor and Critic Network share convolution layers with LSTM
    """

    state_input = Input(shape=(input_shape))  # 4x64x64x3
    x = TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4),
                               activation='relu'))(state_input)
    x = TimeDistributed(
        Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))(x)
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu'))(x)
    x = TimeDistributed(Flatten())(x)

    x = LSTM(512, activation='tanh')(x)

    # Actor Stream
    actor = Dense(action_size, activation='softmax')(x)

    # Critic Stream
    critic = Dense(value_size, activation='linear')(x)

    model = Model(inputs=state_input, outputs=[actor, critic])

    adam = Adam(lr=learning_rate, clipnorm=1.0)
    model.compile(loss=['categorical_crossentropy', 'mse'],
                  optimizer=adam, loss_weights=[1., 1.])

    return model
