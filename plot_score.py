import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
import json

sns.set()
plt.close("all")

DIR_PATH = "./experiments/CartPole-v1_16"
SCORE_FILE = "scores.csv"
EVALUTION_FILE = "evaluation.csv"
SETTINGS_FILE = "settings.json"

REWARD_THRESHOLD = 475

SHOW_GOAL = True
SHOW_TREND = True
SHOW_LEGEND = True

with open(os.path.join(DIR_PATH, "settings.json"), "r") as settings_file:
    settings = json.load(settings_file)

WINDOW_SIZE = settings["window_size"]

def plot_score():    
    x = []
    y = []
    with open(os.path.join(DIR_PATH, SCORE_FILE), "r") as file:
        reader = csv.reader(file)
        for i, scores in enumerate(reader):
            x.append(int(i))
            y.append([float(score) for score in scores])
    x = np.array(x)
    y = np.array(y)
    
    plt.figure()
    plt.plot(x, y[:,0], label="score per episode")
    plt.plot(x, y[:,2], label=f"last {WINDOW_SIZE} episodes avg")
    
    if SHOW_GOAL:
        plt.plot(x, [REWARD_THRESHOLD] * len(x), linestyle=":", label=f"reward threshold: {REWARD_THRESHOLD}")
    
    if SHOW_TREND and len(x) > 1:
        z = np.polyfit(x, y[:,0], 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), linestyle="-.",  label="trend")
    
    plt.title(os.path.join(DIR_PATH, SCORE_FILE))
    plt.xlabel("episodes")
    plt.ylabel("scores")
    
    if SHOW_LEGEND:
        plt.legend(loc="upper left")
        
def plot_evaluation():    
    x = []
    y = []
    with open(os.path.join(DIR_PATH, EVALUTION_FILE), "r") as file:
        reader = csv.reader(file)
        for scores in reader:
            x.append(int(scores[0]))
            y.append([float(score) for score in scores[1:]])
    x = np.array(x)
    y = np.array(y)
    
    mean_eval = np.mean(y, axis=1)
    std_eval = np.std(y, axis=1)
    min_eval = np.min(y, axis=1)
    max_eval = np.max(y, axis=1)
    
    plt.figure()
    plt.plot(x, mean_eval, label="mean score", color=sns.color_palette()[0])
    plt.fill_between(x, mean_eval-std_eval, mean_eval+std_eval, alpha=0.5, color=sns.color_palette()[0])
    plt.fill_between(x, min_eval, max_eval, alpha=0.2, color=sns.color_palette()[0])
    
    if SHOW_GOAL:
        plt.plot(x, [REWARD_THRESHOLD] * len(x), linestyle=":", label=f"reward threshold: {REWARD_THRESHOLD}",
                 color=sns.color_palette()[2])
    
    if SHOW_TREND and len(x) > 1:
        z = np.polyfit(x, mean_eval, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), linestyle="-.",  label="trend", color=sns.color_palette()[3])
    
    plt.title(os.path.join(DIR_PATH, EVALUTION_FILE))
    plt.xlabel("steps")
    plt.ylabel("scores")
    
    if SHOW_LEGEND:
        plt.legend(loc="upper left")
        
plot_score()
plot_evaluation()