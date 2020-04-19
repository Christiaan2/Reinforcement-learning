import gym
import numpy as np

env_name = 'CartPole-v1'
env = gym.make(env_name)
env.seed(0)

total = 0
total2 = 0
#actions = np.array([1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1 1])

for j in range(10):
    done = False
    state = env.reset()
    while not done:
        action = env.action_space.sample()
        d1, d2, done, d4 = env.step(action)
        total2 += np.sum(d1)
    total += np.sum(state)

print(total)
print(total2)

env.seed(0)
total = 0
total2 = 0

for j in range(10):
    done = False
    state = env.reset()
    while not done:
        action = env.action_space.sample()
        d1, d2, done, d4 = env.step(action)
        total2 += np.sum(d1)
    total += np.sum(state)

print(total)
print(total2)