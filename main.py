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

DIR_PATH = "./experiments/CartPole-v1_15"
DO_TRAINING = False

ENV_NAME = "CartPole-v1"
MEMORY_SIZE = 1000000
MEMORY_MIN = 20000
FRAMES_PER_STEP = 4
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_DECAY = 0.99988
MINIBATCH_SIZE = 32
BATCH_SIZE = 4
LEARNING_RATE = 0.00025
GAMMA = 0.98
WINDOW_SIZE = 100
SEED = 0
UPDATE_TARGET_Q_AFTER_N_STEPS = 1000
TAU = 1.0
NUM_EPISODES_EVAL = 100
STEPS_PER_EVAL = 125
EXPLORATION_RATE_EVAL = 0.2
SEED_EVAL = 0

class DQNAgent:
    def __init__(self, dir_path=None):

        def initialize():
            # create environment and initial parameters
            self.env = gym.make(self.env_name)
            self.env.seed(self.seed)
            self.env_eval = gym.make(self.env_name)
            self.observation_space_size = self.env.observation_space.shape[0]
            self.action_space_size = self.env.action_space.n
            self.reward_threshold = self.env.spec.reward_threshold
            self.score_max = self.env.spec.max_episode_steps
            self.exploration_rate = self.exloration_max
            self.memory = deque(maxlen=self.memory_size)
            self.tnet_counter = 0
            self.step_counter = 0

            # create ScoreLogger
            self.score_logger = ScoreLogger(self.dir_path, self.window_size, self.reward_threshold)

        if dir_path is None:
            # settings
            self.env_name = ENV_NAME
            self.exloration_max = EXPLORATION_MAX
            self.exploration_min = EXPLORATION_MIN
            self.exploration_decay = EXPLORATION_DECAY
            self.memory_size = MEMORY_SIZE
            self.memory_min = MEMORY_MIN
            self.minibatch_size = MINIBATCH_SIZE
            self.batch_size = BATCH_SIZE
            self.learning_rate = LEARNING_RATE
            self.gamma = GAMMA
            self.window_size = WINDOW_SIZE
            self.seed = SEED
            self.update_target_q_after_n_steps = UPDATE_TARGET_Q_AFTER_N_STEPS
            self.tau = TAU
            self.num_episodes_eval = NUM_EPISODES_EVAL
            self.steps_per_eval = STEPS_PER_EVAL
            self.exploration_rate_eval = EXPLORATION_RATE_EVAL
            self.seed_eval = SEED_EVAL
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
            self.qnet.compile(loss="huber_loss", optimizer=Adam(learning_rate=self.learning_rate))
            
            self.qnet.save(os.path.join(self.dir_path, "model.HDF5"))
            plot_model(self.qnet, to_file=os.path.join(self.dir_path, "model.png"), show_shapes=True)

            self.tnet = clone_model(self.qnet)
            self.tnet.set_weights(self.qnet.get_weights())
        else:
            with open(os.path.join(dir_path, "settings.json"), "r") as file:
                self.__dict__ = json.load(file)
            
            initialize()

            model_name = "model_best_5840.HDF5"
            self.qnet = load_model(os.path.join(self.dir_path, model_name))
            self.score_logger.log(f"{os.path.join(self.dir_path, model_name)} loaded")

    def train(self):        
        episode = 0
        episode_train = 0
        frame = 0
        temp = True
        while True:
            state = self.env.reset()
            state = np.reshape(state, (1, self.observation_space_size))
            episode += 1
            score = 0
            done = False
            while not done:
                action = self.act(state)
                state_new, reward, done, info = self.env.step(action)
                state_new = np.reshape(state_new, (1, self.observation_space_size))
                score += reward
                frame += 1
                if score >= self.score_max:
                    self.memory.append((state, action, reward, state_new, not done))
                else:
                    self.memory.append((state, action, reward, state_new, done))
                state = state_new
                
                if len(self.memory) >= self.memory_min:
                    if frame % self.frames_per_step == 0:
                        temp = True
                        self.experience_replay()

                    if self.step_counter % self.steps_per_eval == 0 and temp:
                        temp = False
                        self.evaluate()

            if len(self.memory) >= self.memory_min:
                episode_train += 1
                self.score_logger.log(f"\nEpisode: {episode_train} ({episode}), exploration: {self.exploration_rate}, score: {score}")
                self.score_logger.add_score(score, episode, episode_train)
                if episode_train % 64 == 0:
                    self.qnet.save(os.path.join(self.dir_path, "model.HDF5"))
                    self.score_logger.log("Model Saved")
                
                if self.score_logger.save_best_model:
                    self.qnet.save(os.path.join(self.dir_path, "model_best.HDF5"))
                    self.score_logger.save_best_model = False
                    self.score_logger.log("Best model replaced")
                    self.score_logger.solved()
    
    def act(self, state, exploration_rate=None):
        if exploration_rate == None:
            exploration_rate = self.exploration_rate
        if np.random.rand() < exploration_rate:
            return self.env.action_space.sample()
        q_values = self.qnet.predict(state)
        return np.argmax(q_values[0])
    
    def experience_replay(self):
        batch = random.sample(self.memory, self.minibatch_size)
        x = np.zeros((self.minibatch_size, self.observation_space_size))
        y = np.zeros((self.minibatch_size, self.action_space_size))
        for i, (state, action, reward, state_new, done) in enumerate(batch):
            target = self.qnet.predict(state)
            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + \
                    self.gamma*self.tnet.predict(state_new)[0][np.argmax(self.qnet.predict(state_new)[0])]
            x[i, :] = state[0]
            y[i, :] = target[0]
        self.qnet.fit(x, y, batch_size=self.batch_size, verbose=0)

        if self.tnet_counter >= self.update_target_q_after_n_steps:
            w_qnet = self.qnet.get_weights()
            w_tnet = self.tnet.get_weights()

            for i in range(len(w_tnet)):
                w_tnet[i] = w_qnet[i]*self.tau + w_tnet[i]*(1-self.tau)
            self.tnet.set_weights(w_tnet)
            self.tnet_counter = 0
        self.tnet_counter += 1

        self.exploration_rate = np.amax((self.exploration_rate*self.exploration_decay, self.exploration_min))
        self.step_counter += 1

    def evaluate(self):
        self.env_eval.seed(self.seed_eval)
        scores = []
        for i in range(self.num_episodes_eval):
            state = self.env_eval.reset()
            state = np.reshape(state, (1, self.observation_space_size))
            score = 0
            done = False
            while not done:
                action = self.act(state, self.exploration_rate_eval)
                state, reward, done, info = self.env_eval.step(action)
                state = np.reshape(state, (1, self.observation_space_size))
                score += reward
            scores.append(score)
        self.score_logger.add_evaluation(scores, self.step_counter)
    
    def simulate(self, exploration_rate=0.0, verbose=False):
        state = self.env.reset()
        state = np.reshape(state, (1, self.observation_space_size))
        score = 0
        while True:
            self.env.render()
            action = self.act(state, exploration_rate)
            if verbose:
                with np.printoptions(precision=5, sign=' ', floatmode='fixed', suppress=True):
                    self.score_logger.log(f"State: {state[0]}, Output model: {self.qnet.predict(state)[0]}, Action: {action}, score: {score}")
            state, reward, done, info = self.env.step(action)
            score += reward
            state = np.reshape(state, (1, self.observation_space_size))
            time.sleep(0.02)
            if done:
                self.score_logger.log(f"Episode finished, score: {score}")
                break
        self.env.close()

def train_model():
    dqn_agent = DQNAgent()
    dqn_agent.train()

def simulate_model():
    dqn_agent = DQNAgent(DIR_PATH)
    dqn_agent.simulate(exploration_rate=0.02, verbose=True)

if __name__ == "__main__":
    if DO_TRAINING:
        train_model()
    
    simulate_model()


# ToDo:
#manual play
#estimated action-value function Q
#One-hot encoding
