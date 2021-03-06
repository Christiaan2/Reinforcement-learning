import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import os
import csv

class ScoreLogger:
    def __init__(self, dir_path, window_size, reward_threshold):
        self.log_path = os.path.join(dir_path, "logs.txt")
        self.scores_csv_path = os.path.join(dir_path, "scores.csv")
        self.evaluation_csv_path = os.path.join(dir_path, "evaluation.csv")
        self.scores_png_path = os.path.join(dir_path, "scores.png")
        self.window_size = window_size
        self.reward_threshold = reward_threshold
        self.scores = deque(maxlen=self.window_size)
        self.avg_score_max = -np.inf
        self.save_best_model = False
        
    def add_score(self, score, episode, counter):
        self.scores.append(score)
        score_min = np.min(self.scores)
        score_avg = np.mean(self.scores)
        score_max = np.max(self.scores)
        self.log(f"Scores: (min: {score_min}, avg: {score_avg}, max: {score_max})")
        
        self._save_csv(score, score_min, score_avg, score_max)
        if counter % 10 == 0:
            self._save_png(input_path=self.scores_csv_path,
                        output_path=self.scores_png_path,
                        x_label="episode",
                        y_label="scores",
                        average_of_n_last=self.window_size,
                        show_goal=True,
                        show_trend=True,
                        show_legend=True)

        if score_avg >= self.avg_score_max and len(self.scores) >= self.window_size:
            self.avg_score_max = score_avg
            if score >= self.reward_threshold:
                self.save_best_model = True

    def add_evaluation(self, scores, step_counter):
        with open(self.evaluation_csv_path, "a") as evaluation_file:
            writer = csv.writer(evaluation_file)
            writer.writerow([step_counter] + scores)

    def solved(self):
        if np.mean(self.scores) >= self.reward_threshold and len(self.scores) >= self.window_size:
            self.log(f"Solved!")
            exit()

    def _save_png(self, input_path, output_path, x_label, y_label, average_of_n_last, show_goal, show_trend, show_legend):
        x = []
        y = []
        with open(input_path, "r") as scores_file:
            reader = csv.reader(scores_file)
            for i, scores in enumerate(reader):
                x.append(int(i))
                y.append([float(score) for score in scores])
        x = np.array(x)
        y = np.array(y)

        plt.subplots()
        plt.plot(x, y[:,0], label="score per episode")
        plt.plot(x, y[:,2], label=f"last {self.window_size} episodes avg")

        if show_goal:
            plt.plot(x, [self.reward_threshold] * len(x), linestyle=":", label=f"reward threshold: {self.reward_threshold}")

        if show_trend and len(x) > 1:
            z = np.polyfit(x, y[:,0], 1)
            p = np.poly1d(z)
            plt.plot(x, p(x), linestyle="-.",  label="trend")

        plt.title(self.scores_png_path)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if show_legend:
            plt.legend(loc="upper left")

        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    def _save_csv(self, score, score_min, score_avg, score_max):
        with open(self.scores_csv_path, "a") as scores_file:
            writer = csv.writer(scores_file)
            writer.writerow([score, score_min, score_avg, score_max])

    def log(self, message):
        print(message)
        with open(self.log_path, "a") as log_file:
            log_file.write(message + "\n")
