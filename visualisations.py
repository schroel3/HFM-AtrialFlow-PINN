"""
@author: schroel3
"""

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""FOR JUST REFERENCE VELOCITY & PRESSURE"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

# Load in training data
data = scipy.io.loadmat('.../LAA_patient.mat')

# spatio-temporal coordinates
t_star = data['t_star']
x_star = data['x_star']
y_star = data['y_star']
# velocity & pressure fields
U_star = data['U_star']
V_star = data['V_star']
P_star = data['P_star']

# Set up dimensions for plotting
pixel_size = 0.001   # patient 4: 0.0008, test: 0.0012
x_dim = np.arange(np.min(x_star) - 0.002, np.max(x_star) + 0.002, pixel_size)
y_dim = np.arange(np.min(y_star) - 0.002, np.max(y_star) + 0.002, pixel_size)
print('x_dim:', x_dim.shape)
print('y_dim:',y_dim.shape)

# Initialize writer for creating video
movieflag = True
if movieflag:
    writer = FFMpegWriter(fps=3)

# Initialize figure
fig, axs = plt.subplots(2, 1, figsize=(6,10))

# Calculate overall minimum and maximum values of velocity magnitude and pressure
min_velocity = np.nanmin(np.sqrt(U_star**2 + V_star**2))
max_velocity = np.nanmax(np.sqrt(U_star**2 + V_star**2))
min_pressure = np.nanmin(P_star)
max_pressure = np.nanmax(P_star)

# Colorbars for velocity magnitude and pressure on the RHS of the plot
cbar1 = fig.colorbar(axs[0].imshow(np.zeros((10, 10)), vmin=min_velocity, vmax=max_velocity, cmap='jet', aspect='equal'))
cbar1.set_label('Velocity Magnitude')
cbar2 = fig.colorbar(axs[1].imshow(np.zeros((10, 10)), vmin=min_pressure, vmax=max_pressure, cmap='jet', aspect='equal'))
cbar2.set_label('Pressure')

# Start writing the movie
if movieflag:
    writer.setup(fig,'.../true.mp4'', dpi=100)

# Iterate over time steps:
for t in tqdm(range(len(t_star)), desc="Processing frames"):

    # Initialise arrays for velocity magnitude and pressure
    velocity_U = np.full((len(x_dim), len(y_dim)), np.nan)
    velocity_V = np.full((len(x_dim), len(y_dim)), np.nan)
    pressure = np.full((len(x_dim), len(y_dim)), np.nan)

    # Iterate over spatial coordinates:
    for iter_x in range(len(x_star)):
        x_val = x_star[iter_x]
        y_val = y_star[iter_x]

        # Find the index of the spatial coordinate in the grid (i.e. where is is located on the grid)
        x_idx = np.searchsorted(x_dim, x_val) - 1 #find corresponding indices of x_val within x_dim
        y_idx = np.searchsorted(y_dim, y_val) - 1
        #print('x_idx:', x_idx.shape)
        #print('y_idx:', y_idx.shape)

        # Check if the determined index (idx) is within the grid boundaries (dim)
        if np.all((0 <= x_idx) & (x_idx < len(x_dim))) and np.all((0 <= y_idx) & (y_idx < len(y_dim))):
            # Assign predicted velocity to corresponding grid points
            velocity_U[x_idx, y_idx] = U_star[iter_x, t]
            velocity_V[x_idx, y_idx] = V_star[iter_x, t]
            # Assign predicted pressure to corresponding grid points
            pressure[x_idx, y_idx] = P_star[iter_x, t]
            #print('Velocity_U:', velocity_U.shape)

    velocity = np.sqrt(velocity_U ** 2 + velocity_V ** 2) # velocity magnitude

    # Plot velocity magnitude
    im1 = axs[0].imshow(velocity.T, vmin=np.nanmin(velocity), vmax=np.nanmax(velocity), cmap='jet', aspect='equal')
    axs[0].set_title(f'Velocity Magnitude - {t}')
    axs[0].axis('off')
    #cbar1 = fig.colorbar(im1, ax=axs[0], pad=0.01)  # Adjust the pad value as needed
    #cbar1.set_label('Velocity Magnitude')

    # Plot pressure
    im2 = axs[1].imshow(pressure.T, vmin=np.nanmin(P_star), vmax=np.nanmax(P_star), cmap='jet', aspect='equal')
    axs[1].set_title(f'Pressure - {t}')
    axs[1].axis('off')
    #cbar2 = fig.colorbar(im2, ax=axs[1], pad=0.01)  # Adjust the pad value as needed
    #cbar2.set_label('Pressure')

    # Save frame for video
    if movieflag:
        writer.grab_frame()

# Close writer for creating video
if movieflag:
    writer.finish()

plt.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""FOR JUST REGRESSED VELOCITY & PRESSURE"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

# Load in training and predicted data
data1 = scipy.io.loadmat('.../LAA_patient.mat')
data2 = scipy.io.loadmat('.../Results.mat')

# Load data
# Training data loaded for only spatio-temporal coordinates
t_star = data1['t_star']
x_star = data1['x_star']
y_star = data1['y_star']
# Predicted data loaded for velocity and pressure fields
U_pred = data2['U_pred']
V_pred = data2['V_pred']
P_pred = data2['P_pred']

# Set up dimensions for plotting
pixel_size = 0.001  #0.001
x_dim = np.arange(np.min(x_star) - 0.002, np.max(x_star) + 0.002, pixel_size)
y_dim = np.arange(np.min(y_star) - 0.002, np.max(y_star) + 0.002, pixel_size)
print('x_dim:', x_dim.shape)
print('y_dim:',y_dim.shape)

# Initialize writer for creating video
movieflag = True
if movieflag:
    writer = FFMpegWriter(fps=3)

# Initialize figure
fig, axs = plt.subplots(2, 1, figsize=(6,10))

# Calculate overall min and max values of velocity magnitude and pressure
min_velocity = np.nanmin(np.sqrt(U_pred**2 + V_pred**2))
max_velocity = np.nanmax(np.sqrt(U_pred**2 + V_pred**2))
min_pressure = np.nanmin(P_pred)
max_pressure = np.nanmax(P_pred)

# Colorbars for velocity magnitude and pressure on the RHS of the plot
cbar1 = fig.colorbar(axs[0].imshow(np.zeros((10, 10)), vmin=min_velocity, vmax=max_velocity, cmap='jet', aspect='equal'))
cbar1.set_label('Velocity Magnitude')
cbar2 = fig.colorbar(axs[1].imshow(np.zeros((10, 10)), vmin=min_pressure, vmax=max_pressure, cmap='jet', aspect='equal'))
cbar2.set_label('Pressure')

# Start writing the movie
if movieflag:
    writer.setup(fig,'../Regressed.mp4', dpi=100)

# Iterate over time steps
for t in tqdm(range(len(t_star)), desc="Processing frames"):

    # Initialise arrays for velocity and pressure
    velocity_U = np.full((len(x_dim), len(y_dim)), np.nan)
    velocity_V = np.full((len(x_dim), len(y_dim)), np.nan)
    pressure = np.full((len(x_dim), len(y_dim)), np.nan)
    #print('Velocity_U:', velocity_U.shape)

    # Iterate over spatial coordinates
    for iter_x in range(len(x_star)):
        x_val = x_star[iter_x]
        y_val = y_star[iter_x]

        # Find the index of the spatial coordinate in the grid (i.e. where is is located on the grid)
        x_idx = np.searchsorted(x_dim, x_val) - 1 #find corresponding indices of x_val within x_dim
        y_idx = np.searchsorted(y_dim, y_val) - 1
        #print('x_idx:', x_idx.shape)
        #print('y_idx:', y_idx.shape)

        # Check if the determined index (idx) is within the grid boundaries (dim)
        if np.all((0 <= x_idx) & (x_idx < len(x_dim))) and np.all((0 <= y_idx) & (y_idx < len(y_dim))):
            # Assign predicted velocity to corresponding grid points
            velocity_U[x_idx, y_idx] = U_pred[iter_x, t]
            velocity_V[x_idx, y_idx] = V_pred[iter_x, t]
            #velocity = np.sqrt((velocity_U**2) + (velocity_V**2)) # velocity magnitude

            # Assign predicted pressure to corresponding grid points
            pressure[x_idx, y_idx] = P_pred[iter_x, t]

    velocity = np.sqrt(velocity_U ** 2 + velocity_V ** 2) # velocity magnitude

    # Plot velocity magnitude
    im1 = axs[0].imshow(velocity.T, vmin=np.nanmin(velocity), vmax=np.nanmax(velocity), cmap='jet', aspect='equal')
    axs[0].set_title(f'Velocity Magnitude - {t}')
    axs[0].axis('off')

    # Plot pressure
    im2 = axs[1].imshow(pressure.T, vmin=np.nanmin(P_pred), vmax=np.nanmax(P_pred), cmap='jet', aspect='equal')
    #im2 = axs[1].imshow(pressure.T, vmin=min_pressure, vmax=max_pressure, cmap='jet', aspect='equal')
    axs[1].set_title(f'Pressure - {t}')
    axs[1].axis('off')

    # Save frame for video
    if movieflag:
        writer.grab_frame()

# Close writer for creating video
if movieflag:
    writer.finish()

plt.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""SUBPLOT OF BOTH REFERENCE AND REGRESSED VELOCITY & PRESSURE""""""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

# Load in true data
true_data = scipy.io.loadmat('.../true_data.mat')

# Extract true data
t_star = true_data['t_star']
x_star = true_data['x_star']
y_star = true_data['y_star']
U_true = true_data['U_star']
V_true = true_data['V_star']
P_true = true_data['P_star']

#t_star = true_data['t_star_train'][0:39,0:] # set value to no. of timesteps in test dataset
#x_star = true_data['x_star_train'][0:,0:39]
#y_star = true_data['y_star_train'][0:,0:39]
#U_true = true_data['U_star_train'][0:,0:39]
#V_true = true_data['V_star_train'][0:,0:39]
#P_true = true_data['P_star_train'][0:,0:39]

# Load in predicted data
predicted_data = scipy.io.loadmat('.../Results.mat')

# Extract predicted data
U_pred = predicted_data['U_pred']
V_pred = predicted_data['V_pred']
P_pred = predicted_data['P_pred']

# Set up dimensions for plotting
pixel_size = 0.001   # Adjust as needed
x_dim = np.arange(np.min(x_star) - 0.002, np.max(x_star) + 0.002, pixel_size)
y_dim = np.arange(np.min(y_star) - 0.002, np.max(y_star) + 0.002, pixel_size)

# Initialize writer for creating video
movieflag = True
if movieflag:
    writer = FFMpegWriter(fps=3)

# Initialize figure
fig, axs = plt.subplots(2, 2, figsize=(12,10))

# Calculate overall min and max values of true and predicited velocity magnitude and pressure
min_velocity_pred = np.nanmin(np.sqrt(U_pred**2 + V_pred**2))
max_velocity_pred = np.nanmax(np.sqrt(U_pred**2 + V_pred**2))
min_pressure_pred = np.nanmin(P_pred)
max_pressure_pred = np.nanmax(P_pred)
min_velocity_true = np.nanmin(np.sqrt(U_true**2 + V_true**2))
max_velocity_true = np.nanmax(np.sqrt(U_true**2 + V_true**2))
min_pressure_true = np.nanmin(P_true)
max_pressure_true = np.nanmax(P_true)

# Colorbars for velocity magnitude and pressure on the RHS of the plot
cbar1 = fig.colorbar(axs[0,0].imshow(np.zeros((10, 10)), vmin=min_velocity_true, vmax=max_velocity_true, cmap='jet', aspect='equal'))
cbar1.set_label('Velocity Magnitude')
cbar2 = fig.colorbar(axs[1,0].imshow(np.zeros((10, 10)), vmin=min_pressure_true, vmax=max_pressure_true, cmap='jet', aspect='equal'))
cbar2.set_label('Pressure')
cbar3 = fig.colorbar(axs[0,1].imshow(np.zeros((10, 10)), vmin=min_velocity_pred, vmax=max_velocity_pred, cmap='jet', aspect='equal'))
cbar3.set_label('Velocity Magnitude')
cbar4 = fig.colorbar(axs[1,1].imshow(np.zeros((10, 10)), vmin=min_pressure_pred, vmax=max_pressure_pred, cmap='jet', aspect='equal'))
cbar4.set_label('Pressure')

# Start writing the movie
if movieflag:
    writer.setup(fig,'.../true_regressed_comparison.mp4', dpi=100)

# Process frames
for t in tqdm(range(len(t_star)), desc="Processing frames"):

    # Initialize arrays for true and predicted velocity and pressure
    velocity_U_true = np.full((len(x_dim), len(y_dim)), np.nan)
    velocity_V_true = np.full((len(x_dim), len(y_dim)), np.nan)
    pressure_true = np.full((len(x_dim), len(y_dim)), np.nan)
    velocity_U_pred = np.full((len(x_dim), len(y_dim)), np.nan)
    velocity_V_pred = np.full((len(x_dim), len(y_dim)), np.nan)
    pressure_pred = np.full((len(x_dim), len(y_dim)), np.nan)

    # Assign true data to corresponding grid points
    for iter_x in range(len(x_star)):
        x_val = x_star[iter_x]
        y_val = y_star[iter_x]
        x_idx = np.searchsorted(x_dim, x_val) - 1
        y_idx = np.searchsorted(y_dim, y_val) - 1

        if np.all((0 <= x_idx) & (x_idx < len(x_dim))) and np.all((0 <= y_idx) & (y_idx < len(y_dim))):
            velocity_U_true[x_idx, y_idx] = U_true[iter_x, t]
            velocity_V_true[x_idx, y_idx] = V_true[iter_x, t]
            pressure_true[x_idx, y_idx] = P_true[iter_x, t]

    velocity_true = np.sqrt(velocity_U_true ** 2 + velocity_V_true ** 2)

    # Assign predicted data to corresponding grid points
    for iter_x in range(len(x_star)):
        x_val = x_star[iter_x]
        y_val = y_star[iter_x]
        x_idx = np.searchsorted(x_dim, x_val) - 1
        y_idx = np.searchsorted(y_dim, y_val) - 1

        if np.all((0 <= x_idx) & (x_idx < len(x_dim))) and np.all((0 <= y_idx) & (y_idx < len(y_dim))):
            velocity_U_pred[x_idx, y_idx] = U_pred[iter_x, t]
            velocity_V_pred[x_idx, y_idx] = V_pred[iter_x, t]
            pressure_pred[x_idx, y_idx] = P_pred[iter_x, t]

    velocity_pred = np.sqrt(velocity_U_pred ** 2 + velocity_V_pred ** 2)

    # Plot true velocity magnitude
    im1_true = axs[0, 0].imshow(velocity_true.T, vmin=np.nanmin(velocity_true), vmax=np.nanmax(velocity_true), cmap='jet', aspect='equal')
    axs[0, 0].set_title(f'True Velocity Magnitude - {t}')
    axs[0, 0].axis('off')

    # Plot true pressure
    im2_true = axs[1,0].imshow(pressure_true.T, vmin=np.nanmin(P_true), vmax=np.nanmax(P_true), cmap='jet', aspect='equal')
    axs[1,0].set_title(f'True Pressure - {t}')
    axs[1,0].axis('off')

    # Plot predicted velocity magnitude
    im1_pred = axs[0,1].imshow(velocity_pred.T, vmin=np.nanmin(velocity_pred), vmax=np.nanmax(velocity_pred), cmap='jet', aspect='equal')
    axs[0,1].set_title(f'Predicted Velocity Magnitude - {t}')
    axs[0,1].axis('off')

    # Plot predicted pressure
    im2_pred = axs[1,1].imshow(pressure_pred.T, vmin=np.nanmin(P_pred), vmax=np.nanmax(P_pred), cmap='jet', aspect='equal')
    axs[1,1].set_title(f'Predicted Pressure - {t}')
    axs[1,1].axis('off')

    # Save frame for true and predicted video
    if movieflag:
        writer.grab_frame()

# Close writer for creating video
if movieflag:
    writer.finish()

plt.show()

    
