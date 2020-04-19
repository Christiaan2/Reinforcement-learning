import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv

sns.set()

DIR_PATH = "./experiments/CartPole-v1_15"
FILE_NAME = "scores.csv"
REWARD_THRESHOLD = 475

SHOW_GOAL = True
SHOW_TREND = True
SHOW_LEGEND = True

x = []
y = []
with open(os.path.join(DIR_PATH, FILE_NAME), "r") as file:
    reader = csv.reader(file)
    for i, scores in enumerate(reader):
        x.append(int(i))
        y.append([float(score) for score in scores])
x = np.array(x)
y = np.array(y)

plt.subplots()
plt.plot(x, y[:,0], label="score per episode")
plt.plot(x, y[:,2], label=f"last ... episodes avg")

if SHOW_GOAL:
    plt.plot(x, [REWARD_THRESHOLD] * len(x), linestyle=":", label=f"reward threshold: {REWARD_THRESHOLD}")

if SHOW_TREND and len(x) > 1:
    z = np.polyfit(x, y[:,0], 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), linestyle="-.",  label="trend")

plt.title(os.path.join(DIR_PATH, FILE_NAME))
plt.xlabel("episodes")
plt.ylabel("scores")

if SHOW_LEGEND:
    plt.legend(loc="upper left")