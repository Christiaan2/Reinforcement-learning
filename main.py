import numpy as np
import gym
import random
from collections import deque
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from score_logger import ScoreLogger

DO_TRAINING = False

ENV_NAME = "CartPole-v1"

MODEL_PATH = "./scores/model.HDF5"

MEMORY_SIZE = 1000000

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_DECAY = 0.995

BATCH_SIZE = 32
LEARNING_RATE = 0.001
GAMMA = 0.98

class DQNAgent:
    def __init__(self, model_path=None):
        self.env = gym.make(ENV_NAME)
        self.observation_space_size = self.env.observation_space.shape[0]
        self.action_space_size = self.env.action_space.n
        self.exploration_rate = EXPLORATION_MAX
        self.memory = deque(maxlen=MEMORY_SIZE)

        if model_path is None:
            self.model = Sequential()
            self.model.add(Dense(256, input_shape=(self.observation_space_size,), activation='relu'))
            self.model.add(Dense(self.action_space_size, activation='linear'))
            self.model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))

            self.score_logger = ScoreLogger(ENV_NAME)
        else:
            self.model = load_model(model_path)
        
        #self.env.seed(5)

    def train(self):
        episode = 0
        while True:
            episode += 1
            state = self.env.reset()
            state = np.reshape(state, (1, self.observation_space_size))
            score = 0
            while True:
                action = self.act(state, off_policy=True)
                state_new, reward, done, info = self.env.step(action)
                score += int(reward)
                state_new = np.reshape(state_new, (1, self.observation_space_size))
                self.memory.append((state, action, reward, state_new, done))
                self.experience_replay()
                state = state_new
                if done:
                    self.model.save(MODEL_PATH)
                    print(f"Episode: {episode}, exploration: {self.exploration_rate}, score: {score}")
                    self.score_logger.add_score(score, episode)
                    break
        self.env.close()
    
    def act(self, state, off_policy=False):
        if off_policy:
            if np.random.rand() < self.exploration_rate:
                return self.env.action_space.sample()
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])
    
    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_new, done in batch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + GAMMA*np.amax(self.model.predict(state_new)[0])
            self.model.fit(state, target, verbose=0)
        self.exploration_rate = np.amax((self.exploration_rate*EXPLORATION_DECAY, EXPLORATION_MIN))

    def simulate(self):
        state = self.env.reset()
        state = np.reshape(state, (1, self.observation_space_size))
        score = 0
        while True:
            self.env.render()
            action = self.act(state, off_policy=False)
            state, reward, done, info = self.env.step(action)
            score += int(reward)
            state = np.reshape(state, (1, self.observation_space_size))
            time.sleep(0.05)
            if done:
                print(f"Episode finished, score: {score}")
                break
        self.env.close()

def train_model():
    dqn_agent = DQNAgent()
    dqn_agent.train()

def simulate_model():
    dqn_agent = DQNAgent(MODEL_PATH)
    dqn_agent.simulate()

if __name__ == "__main__":
    if DO_TRAINING:
        train_model()
    
    simulate_model()


# ToDo:
#manual play
#estimated action-value function Q