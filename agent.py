import random
from collections import deque

import numpy as np
from keras import Sequential
from keras.layers import LeakyReLU
from keras.layers.core import Dense


class Agent:
    def __init__(self, player_number, board_size):
        self.number = player_number  # player number from game
        self.board_size = board_size
        self.number_of_fields = board_size[0] * board_size[1]
        self.epsilon = 1
        self.epsilon_decay = 0.9999
        self.model = self.construct_model()
        self.memory = deque(maxlen=10_000)
        self.discount_rate = 0.9
        self.lr = 0.0001

    def parse_state_for_model(self, state):
        if self.number != 0:
            state = [[element if element is None else abs(element - 1) for element in row] for row in state]

        state = [[[1, 0, 0] if element == 0 else element for element in row] for row in state]
        state = [[[0, 1, 0] if element == 1 else element for element in row] for row in state]
        state = [[[0, 0, 1] if element is None else element for element in row] for row in state]
        return np.array(state).reshape(1, -1)

    def random_move(self, state):
        return random.choice(self.legal_moves(state))

    @staticmethod
    def legal_moves(state):
        legal_moves = []
        for row_num in range(len(state)):
            for col_num in range(len(state[0])):
                if state[row_num][col_num] is None:
                    legal_moves.append((col_num, row_num))
        return legal_moves

    def model_move(self, state):
        parsed_state = self.parse_state_for_model(state)

        if random.random() > self.epsilon:
            # We are choosing legal move with highest score in q-table
            action = self.model.predict(parsed_state)[0]
            action = [(idx, value) for idx, value in enumerate(action)]
            action = sorted(action, key=lambda x_: x_[1], reverse=True)
            legal_moves = self.legal_moves(state)
            action = [element for element in action
                      if (element[0] % self.board_size[0], element[0] // self.board_size[0]) in legal_moves]

            action = action[0][0]
            x = action % self.board_size[0]
            y = action // self.board_size[0]
            action = x, y
        else:
            action = self.random_move(state)

        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay

        return action

    def action(self, state):
        return self.model_move(state)

    def construct_model(self):
        model = Sequential()
        model.add(Dense(32, activation='linear', input_shape=(3 * self.number_of_fields,)))
        model.add(LeakyReLU())
        model.add(Dense(32, activation='linear'))
        model.add(LeakyReLU())
        model.add(Dense(self.number_of_fields, activation='elu'))
        model.compile(optimizer='adam', loss='mean_squared_error')

        return model

    def remember(self, state, action, next_state, reward, done):
        if state is not None and action is not None and reward is not None and done is not None:
            self.memory.append([state, action, next_state, reward, done])

    def learn(self):
        for state, action, next_state, reward, done in random.choices(self.memory, k=200):
            state = self.parse_state_for_model(state)
            q_pred = self.model.predict(state)
            if done:
                q_pred = np.zeros((1, len(q_pred[0])))
                q_pred[0][action] = reward
            else:
                next_state = self.parse_state_for_model(next_state)
                q_pred[0][action] = (1 - self.lr) * q_pred[0][action] \
                                    + self.lr * (reward + self.discount_rate * max(self.model.predict(next_state)[0]))
            self.model.fit(state, q_pred, verbose=0)

    def save_model(self, filename):
        self.model.save(f'{filename}.hdf5')
