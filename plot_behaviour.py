from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from tensorflow.keras.models import load_model

sns.set()

DIR_PATH = "./experiments/CartPole-v1_15"
MODEL_NAME = "model_5824.HDF5"

LABELS = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Velocity']
ACTIONS = ['LEFT', 'RIGHT']

NUM_GRID_POINTS = 600

# Default values
CART_POSITION_DEFAULT = 0.0
CART_VELOCITY_DEFAULT = 0.0
POLE_ANGLE_DEFAULT = 0.0
POLE_VELOCITY_DEFAULT = 0.0

# Extreme values
CART_POSITION_EXTREMES = [-2.4, 2.4]
CART_VELOCITY_EXTREMES = [-2.0, 2.0]
POLE_ANGLE_EXTREMES = [-0.25, 0.25]
POLE_VELOCITY_EXTREMES = [-2.0, 2.0]

class PlotBehaviour:
    def __init__(self):
        self.model = load_model(os.path.join(DIR_PATH, MODEL_NAME))
        self.labels = LABELS
        self.observation_space_size = len(self.labels)
        self.num_grid_points = NUM_GRID_POINTS
        self.defaults = [CART_POSITION_DEFAULT, CART_VELOCITY_DEFAULT, POLE_ANGLE_DEFAULT, POLE_VELOCITY_DEFAULT]
        self.extremes = [CART_POSITION_EXTREMES, CART_VELOCITY_EXTREMES, POLE_ANGLE_EXTREMES, POLE_VELOCITY_EXTREMES]
        self.actions = ACTIONS
        
        self.ncols = self.observation_space_size // 2
        self.nrows = int(np.ceil(self.observation_space_size / self.ncols))
        self.hor_axis = [np.linspace(extreme[0], extreme[1], self.num_grid_points) for extreme in self.extremes]

    def update(self, val=None):
        sns_cmap = ListedColormap(sns.color_palette().as_hex())
        for i in range(len(self.labels)):
            model_input = np.ones((self.num_grid_points, self.observation_space_size))
            for j in range(len(self.labels)):
                model_input[:,j] *= self.sliders[j].val
            
            model_input[:,i] = self.hor_axis[i]
            q_values = self.model.predict(model_input)

            col = i % self.ncols
            row = i // self.nrows
            self.ax[row,col].clear()
            for j, action in enumerate(self.actions):
                self.ax[row,col].plot(self.hor_axis[i], q_values[:,j], label=action)
            
            action = np.argmax(q_values, axis=1)
            action = np.reshape(action, (1,self.num_grid_points))
            self.ax[row,col].pcolormesh(np.linspace(self.extremes[i][0], self.extremes[i][1], self.num_grid_points+1), \
                np.array(self.ax[row,col].get_ylim()), action, alpha=0.15, vmin=0, vmax=10, cmap=sns_cmap)
            
            self.ax[row,col].legend()
            self.ax[row,col].title.set_text(self.labels[i])
            self.ax[row,col].set_ylabel('Q-value')

    def setLayout(self):
        plt.subplots_adjust(top=0.920, bottom=0.096, left=0.25, right=0.972, hspace=0.404, wspace=0.297)

        self.sliders = []
        for i in range(len(self.labels)):
            col = i % self.ncols
            row = i // self.nrows
            
            ax = plt.axes([0.038+col*0.1, 0.58-row*0.484, 0.03, 0.34])
            self.sliders.append(Slider(ax, self.labels[i], valmin=self.extremes[i][0], \
                valmax=self.extremes[i][1], valinit=self.defaults[i], orientation='vertical'))

            self.sliders[i].on_changed(self.update)

    def plot(self):
        fig, self.ax = plt.subplots(ncols=self.ncols, nrows=self.nrows)
        self.setLayout()
        self.update()
        plt.show()

def main():
    plotbehaviour = PlotBehaviour()
    plotbehaviour.plot()

if __name__ == "__main__":
    main()