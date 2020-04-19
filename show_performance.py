from matplotlib.widgets import Slider, RadioButtons
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from tensorflow.keras.models import load_model

sns.set()

DIR_PATH = "./experiments/CartPole-v1_15"
MODEL_NAME = "model_5824.HDF5"

model = load_model(os.path.join(DIR_PATH, MODEL_NAME))

# Environment
OBSERVATION_SPACE_SIZE = 4

# Indices
CART_POSITION = 0
CART_VELOCITY = 1
POLE_ANGLE = 2
POLE_VELOCITY = 3

NUM_GRID_POINTS = 500

# Default values
CART_POSITION_DEFAULT = 0.0
CART_VELOCITY_DEFAULT = 0.0
POLE_ANGLE_DEFAULT = 0.0
POLE_VELOCITY_DEFAULT = 0.0
DEFAULTS = [CART_POSITION_DEFAULT, CART_VELOCITY_DEFAULT, POLE_ANGLE_DEFAULT, POLE_VELOCITY_DEFAULT]

# Extreme values
CART_POSITION_EXTREMES = [-2.4, 2.4]
CART_VELOCITY_EXTREMES = [-1.6, 1.6]
POLE_ANGLE_EXTREMES = [-0.22, 0.22]
POLE_VELOCITY_EXTREMES = [-2.0, 2.0]

values_array = np.ones((NUM_GRID_POINTS, OBSERVATION_SPACE_SIZE))
for i in range(OBSERVATION_SPACE_SIZE):
    values_array[:,i] *= default_values[i]

q_values = model.predict(values_array)

fig, ax = plt.subplots(2, 2)
plt.subplots_adjust(bottom=0.35)

axcolor = 'lightgoldenrodyellow'
cart_pos_ax = plt.axes([0.25, 0.25, 0.65, 0.03])
cart_vel_ax = plt.axes([0.25, 0.2, 0.65, 0.03])
pole_angle_ax = plt.axes([0.25, 0.15, 0.65, 0.03])
pole_vel_ax = plt.axes([0.25, 0.5, 0.65, 0.03])

cart_position_slider = Slider(cart_pos_ax, 'Cart Position', CART_POSITION_EXTREMES[0], \
    CART_POSITION_EXTREMES[1], valinit=DEFAULTS[CART_POSITION], valstep=delta_f)
samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)

plt.plot(CART_POSITION_SET, q_values[:,0], label="LEFT")
plt.plot(CART_POSITION_SET, q_values[:,1], label="RIGHT")
plt.legend()
plt.show()