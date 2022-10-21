import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, Input
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from collections import deque
import random

from JobShopEnv import JobShopEnv

MEMORY_SIZE = 200  # the amount of steps to keep for training a model
DISCOUNT_FACTOR = 0.95  # discount rate
LEARNING_RATE = 0.0005
MINIBATCH_SIZE = 30  # number of samples to use for training
UPDATE_TARGET_EVERY = 5  # states that are terminal
EPISODES = 1000

# exploration rate
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

class DQNAgent:
    # class for deep q learning agent

    def __init__(self, state_size, action_size, number_of_jobs, number_of_features):
        self.state_size = state_size
        self.action_size = action_size
        self.number_of_jobs = number_of_jobs
        self.number_of_features = number_of_features
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = 0.9  # not a constant, going to be decayed

        # Main model which we .fit() the model for every step the agent takes
        self.model = self.create_model()

        # Target network which we .predict() for future q values
        # handling the initial chaos with randomness
        self.target_model = self.create_model()

        # Updating the model weights every some number of steps, not allowing the model to make predition every step,
        # instead for example every 5 episodes or so,
        # in order to have some kind of stability, so the model can actually learn and not overfit to
        self.target_model.set_weights(self.model.get_weights())

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_basic_model(self):

        model = Sequential(name='basic_model')
        model.add(Dense(24, input_dim=self.number_of_features, activation='relu'))
        model.add(Dense(24, input_dim=self.number_of_features, activation='relu'))
        model.add(Dense(24, input_dim=self.number_of_features, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))

        return model

    def create_model(self):

        basic_model = self.create_basic_model()

        output_list = []
        input_list = []
        for i in range(self.number_of_jobs):
            input_list.append(Input(shape=(self.number_of_features,)))  # putting all states of the jobs together
            output_list.append(basic_model(input_list[i]))  # forward pass to calculate the q values for each state

        concatenated = concatenate(output_list)     # concatenating all models that each state was passed through
                                                    # since they all lead to the same classification
        out = Dense(self.action_size, activation='linear')(concatenated)  # action_size = number_of_jobs
        model = Model(input_list, out)
        model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))  # new model that we use to .predict() and .fit()
        return model

    def act(self, current_state):
        if np.random.rand() > self.epsilon:
            action = np.argmax(self.model.predict(current_state))  # exploit, initially 10%
        else:
            action = random.randrange(self.action_size)  # explore, initially 90%
        return action  # returns action

    def update_memory(self, current_state, action, reward, next_state, done):

        # Remember the information of current_state
        self.memory.append((current_state, action, reward, next_state, done))

    def train(self, done):

        minibatch = random.sample(self.memory, MINIBATCH_SIZE)

        # If not a terminal state, get new q from future states, otherwise set it to 0
        for current_state, action, reward, next_state, done in minibatch:
            # Update q value for future state
            future_qs = np.max(self.target_model.predict(next_state)[0])  # we get [0] element bc .predict() returns [[]]
            if not done:
                new_q = reward + DISCOUNT_FACTOR * future_qs
            else:
                new_q = reward
            # Update Q value for given state
            current_qs = self.model.predict(current_state)
            current_qs[0][action] = new_q

            self.model.fit(current_state, current_qs, epochs=1, verbose=0)

        if self.epsilon > EPSILON_MIN:      # let the epsilon decay until it reaches the minimum
            self.epsilon *= EPSILON_DECAY    # less exploring more exploiting, epsilon decays to an optimal one

        # Update target network counter every episode
        if done:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network (target_model)
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
