import numpy as np
import gym
import random
from collections import deque
import time
import os
import json
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from score_logger import ScoreLogger

DIR_PATH = "./experiments/CartPole-v1_10"
DO_TRAINING = True

ENV_NAME = "CartPole-v1"

MEMORY_SIZE = 1000000
MEMORY_MIN = 500000

FRAMES_PER_STEP = 4

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_DECAY = 0.999982

BATCH_SIZE = 32
LEARNING_RATE = 0.00025
GAMMA = 0.98

WINDOW_SIZE = 100
AVG_SCORE_TO_SOLVE = 480

SEED = 0

UPDATE_TARGET_Q_AFTER_N_STEPS = 1250

class DQNAgent:
    def __init__(self, dir_path=None):

        def initialize():
            # create environment and initial parameters
            self.env = gym.make(self.env_name)
            if DO_TRAINING:
                self.env.seed(self.seed)
            self.observation_space_size = self.env.observation_space.shape[0]
            self.action_space_size = self.env.action_space.n
            self.exploration_rate = self.exloration_max
            self.memory = deque(maxlen=self.memory_size)
            self.tnet_counter = 0

            # create ScoreLogger
            self.score_logger = ScoreLogger(self.dir_path, self.window_size, self.avg_score_to_solve)

        if dir_path is None:
            # settings
            self.env_name = ENV_NAME
            self.exloration_max = EXPLORATION_MAX
            self.exploration_min = EXPLORATION_MIN
            self.exploration_decay = EXPLORATION_DECAY
            self.memory_size = MEMORY_SIZE
            self.batch_size = BATCH_SIZE
            self.learning_rate = LEARNING_RATE
            self.gamma = GAMMA
            self.window_size = WINDOW_SIZE
            self.avg_score_to_solve = AVG_SCORE_TO_SOLVE
            self.seed = SEED
            self.update_target_q_after_n_steps = UPDATE_TARGET_Q_AFTER_N_STEPS
            self.memory_min = MEMORY_MIN
            self.frames_per_step = FRAMES_PER_STEP

            # create new directory to store settings and results
            run = 0
            while True:
                run += 1
                if not os.path.exists(f"./experiments/{ENV_NAME}_{run}"):
                    self.dir_path = f"./experiments/{ENV_NAME}_{run}"
                    os.mkdir(self.dir_path)
                    break

            # save settings
            with open(os.path.join(self.dir_path, "settings.json"), "w") as file:
                json.dump(self.__dict__, file)

            initialize()
            self.score_logger.log(f"Results of experiments stored in: {self.dir_path}")

            # create model and store model and visualization
            # self.qnet is online model, self.tnet is target model
            self.qnet = Sequential()
            self.qnet.add(Dense(256, input_shape=(self.observation_space_size,), activation='relu'))
            self.qnet.add(Dense(self.action_space_size, activation='linear'))
            self.qnet.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
            
            self.qnet.save(os.path.join(self.dir_path, "model.HDF5"))
            plot_model(self.qnet, to_file=os.path.join(self.dir_path, "model.png"), show_shapes=True)

            self.tnet = clone_model(self.qnet)
            self.tnet.set_weights(self.qnet.get_weights())
        else:
            with open(os.path.join(dir_path, "settings.json"), "r") as file:
                self.__dict__ = json.load(file)
            
            initialize()

            model_name = "model_best.HDF5"
            self.qnet = load_model(os.path.join(self.dir_path, model_name))
            self.score_logger.log(f"{os.path.join(self.dir_path, model_name)} loaded")

    def train(self):        
        episode = 0
        i = 0
        j = 0
        while True:
            episode += 1
            state = self.env.reset()
            state = np.reshape(state, (1, self.observation_space_size))
            score = 0
            done = False
            while not done:
                action = self.act(state, off_policy=True)
                state_new, reward, done, info = self.env.step(action)
                score += reward
                state_new = np.reshape(state_new, (1, self.observation_space_size))
                self.memory.append((state, action, reward, state_new, done))
                state = state_new
                
                if len(self.memory) >= self.memory_min:
                    i += 1
                    if i%self.frames_per_step == 0:
                        self.experience_replay()

                    if done:
                        j += 1
                        self.score_logger.log(f"\nEpisode: {episode}, exploration: {self.exploration_rate}, score: {score}")
                        if j%64 == 0:
                            self.qnet.save(os.path.join(self.dir_path, "model.HDF5"))
                            self.score_logger.log("Model Saved")
                        self.score_logger.add_score(score, episode, j)
                        if self.score_logger.save_best_model:
                            self.qnet.save(os.path.join(self.dir_path, "model_best.HDF5"))
                            self.score_logger.save_best_model = False
                            self.score_logger.log("Best model replaced")
        self.env.close()
    
    def act(self, state, off_policy=False):
        if off_policy:
            if np.random.rand() < self.exploration_rate:
                return self.env.action_space.sample()
        q_values = self.qnet.predict(state)
        return np.argmax(q_values[0])
    
    def experience_replay(self):
        self.tnet_counter += 1
        if self.tnet_counter > self.update_target_q_after_n_steps:
            self.tnet.set_weights(self.qnet.get_weights())
            self.score_logger.log("Weights of target Q-network updated")
            self.tnet_counter = 0

        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, state_new, done in batch:
            target = self.qnet.predict(state)
            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + \
                    self.gamma*self.tnet.predict(state_new)[0][np.argmax(self.qnet.predict(state_new)[0])]
            self.qnet.fit(state, target, verbose=0)
        
        self.exploration_rate = np.amax((self.exploration_rate*self.exploration_decay, self.exploration_min))

    def simulate(self, verbose=False):
        state = self.env.reset()
        state = np.reshape(state, (1, self.observation_space_size))
        score = 0
        while True:
            self.env.render()
            action = self.act(state, off_policy=False)
            if verbose:
                with np.printoptions(precision=5, sign=' ', floatmode='fixed', suppress=True):
                    self.score_logger.log(f"State: {state[0]}, Output model: {self.qnet.predict(state)[0]}, Action: {action}, score: {score}")
            state, reward, done, info = self.env.step(action)
            score += reward
            state = np.reshape(state, (1, self.observation_space_size))
            time.sleep(0.05)
            if done:
                self.score_logger.log(f"Episode finished, score: {score}")
                break
        self.env.close()

def train_model():
    dqn_agent = DQNAgent()
    dqn_agent.train()

def simulate_model():
    dqn_agent = DQNAgent(DIR_PATH)
    dqn_agent.simulate(verbose=True)

if __name__ == "__main__":
    if DO_TRAINING:
        train_model()
    
    simulate_model()


# ToDo:
#manual play
#estimated action-value function Q
#One-hot encoding
