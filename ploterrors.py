
"""
authors: schroel3, Maziar Raissi
"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""FOR COMPARING TRAINING TIMES"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import numpy as np

def relative_L2_error(pred, exact):
    error = np.sqrt(np.mean((pred - exact)**2) / np.mean((exact - np.mean(exact))**2))
    return error

# Load _star variables from the main data file
data = scipy.io.loadmat('.../inputdata.mat')
t_star = data['t_star']
C_star = data['C_star']
U_star = data['U_star']
V_star = data['V_star']
W_star = data['W_star']
P_star = data['P_star']

# Load _pred variables from the first file (1 hour training)
data = scipy.io.loadmat('.../Results.mat')
C_pred_1hr = data['C_pred']
U_pred_1hr = data['U_pred']
V_pred_1hr = data['V_pred']
W_pred_1hr = data['W_pred']
P_pred_1hr = data['P_pred']

# Load _pred variables from the second file (2 hours training)
data = scipy.io.loadmat('.../Results.mat')
C_pred_2hr = data['C_pred']
U_pred_2hr = data['U_pred']
V_pred_2hr = data['V_pred']
W_pred_2hr = data['W_pred']
P_pred_2hr = data['P_pred']

# Load _pred variables from the third file (10 hours training)
data = scipy.io.loadmat('.../Results.mat')
C_pred_10hr = data['C_pred']
U_pred_10hr = data['U_pred']
V_pred_10hr = data['V_pred']
W_pred_10hr = data['W_pred']
P_pred_10hr = data['P_pred']

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

n_star = t_star.shape[0]

# Plot for concentration
axs[0, 2].plot(t_star, [relative_L2_error(C_pred_1hr[:, num], C_star[:, num]) for num in range(n_star)], label='1 hour training', linewidth=2.5)
axs[0, 2].plot(t_star, [relative_L2_error(C_pred_2hr[:, num], C_star[:, num]) for num in range(n_star)], label='2 hours training', linewidth=2.5)
axs[0, 2].plot(t_star, [relative_L2_error(C_pred_10hr[:, num], C_star[:, num]) for num in range(n_star)], label='10 hours training', linewidth=2.5)
axs[0, 2].set_xlabel('$t$', fontsize=15)
axs[0, 2].set_ylabel('Rel. $L_2$ Error', fontsize=15)
axs[0, 2].set_title('$c(t,x,y,z)$', fontsize=15)
axs[0, 2].tick_params(axis='both', which='major', labelsize=13)

# Plot for pressure
axs[0, 0].plot(t_star, [relative_L2_error(P_pred_1hr[:, num] - np.mean(P_pred_1hr[:, num]), P_star[:, num] - np.mean(P_star[:, num])) for num in range(n_star)], label='1 hour training', linewidth=2.5)
axs[0, 0].plot(t_star, [relative_L2_error(P_pred_2hr[:, num] - np.mean(P_pred_2hr[:, num]), P_star[:, num] - np.mean(P_star[:, num])) for num in range(n_star)], label='2 hours training', linewidth=2.5)
axs[0, 0].plot(t_star, [relative_L2_error(P_pred_10hr[:, num] - np.mean(P_pred_10hr[:, num]), P_star[:, num] - np.mean(P_star[:, num])) for num in range(n_star)], label='10 hours training', linewidth=2.5)
axs[0, 0].set_xlabel('$t$', fontsize=15)
axs[0, 0].set_ylabel('Rel. $L_2$ Error', fontsize=15)
axs[0, 0].set_title('$p(t,x,y,z)$', fontsize=15)
axs[0, 0].tick_params(axis='both', which='major', labelsize=13)

# Plot for velocity in x-direction
axs[1, 0].plot(t_star, [relative_L2_error(U_pred_1hr[:, num], U_star[:, num]) for num in range(n_star)], label='1 hour training', linewidth=2.5)
axs[1, 0].plot(t_star, [relative_L2_error(U_pred_2hr[:, num], U_star[:, num]) for num in range(n_star)], label='2 hours training', linewidth=2.5)
axs[1, 0].plot(t_star, [relative_L2_error(U_pred_10hr[:, num], U_star[:, num]) for num in range(n_star)], label='10 hours training', linewidth=2.5)
axs[1, 0].set_xlabel('$t$', fontsize=15)
axs[1, 0].set_ylabel('Rel. $L_2$ Error', fontsize=15)
axs[1, 0].set_title('$u(t,x,y,z)$', fontsize=15)
axs[1, 0].tick_params(axis='both', which='major', labelsize=13)

# Plot for velocity in y-direction
axs[1, 1].plot(t_star, [relative_L2_error(V_pred_1hr[:, num], V_star[:, num]) for num in range(n_star)], label='1 hour training', linewidth=2.5)
axs[1, 1].plot(t_star, [relative_L2_error(V_pred_2hr[:, num], V_star[:, num]) for num in range(n_star)], label='2 hours training', linewidth=2.5)
axs[1, 1].plot(t_star, [relative_L2_error(V_pred_10hr[:, num], V_star[:, num]) for num in range(n_star)], label='10 hours training', linewidth=2.5)
axs[1, 1].set_xlabel('$t$', fontsize=15)
axs[1, 1].set_ylabel('Rel. $L_2$ Error', fontsize=15)
axs[1, 1].set_title('$v(t,x,y,z)$', fontsize=15)
axs[1, 1].tick_params(axis='both', which='major', labelsize=13)  # Increase font size of axis tick labels

# Plot for velocity in z-direction
axs[1, 2].plot(t_star, [relative_L2_error(W_pred_1hr[:, num], W_star[:, num]) for num in range(n_star)], label='1 hour training', linewidth=2.5)
axs[1, 2].plot(t_star, [relative_L2_error(W_pred_2hr[:, num], W_star[:, num]) for num in range(n_star)], label='2 hours training', linewidth=2.5)
axs[1, 2].plot(t_star, [relative_L2_error(W_pred_10hr[:, num], W_star[:, num]) for num in range(n_star)], label='10 hours training', linewidth=2.5)
axs[1, 2].set_xlabel('$t$', fontsize=15)
axs[1, 2].set_ylabel('Rel. $L_2$ Error', fontsize=15)
axs[1, 2].set_title('$w(t,x,y,z)$', fontsize=15)
axs[1, 2].tick_params(axis='both', which='major', labelsize=13)

# Move legend to the empty subplot
handles, labels = axs[1, 2].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.52, 0.7), fontsize=15)

plt.suptitle('Aneurysm 3D', fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save figure to PNG file
plt.savefig('.../plot_comparison.png', dpi=300)

plt.show()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""## FOR ONE TRAINING TIME"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import numpy as np

def relative_L2_error(pred, exact):
    error = np.sqrt(np.mean((pred - exact)**2) / np.mean((exact - np.mean(exact))**2))
    return error


# Load _star variables from the main data file
data = scipy.io.loadmat('.../inputdata.mat')
t_star = data['t_star']
C_star = data['C_star']
U_star = data['U_star']
V_star = data['V_star']
P_star = data['P_star']

# Load _pred variables from the first file (1 hour training)
data = scipy.io.loadmat('.../Results.mat')
C_pred_1hr = data['C_pred']
U_pred_1hr = data['U_pred']
V_pred_1hr = data['V_pred']
P_pred_1hr = data['P_pred']


fig, axs = plt.subplots(2, 2, figsize=(15, 10))

n_star = t_star.shape[0]

# Plot for concentration
axs[0, 1].plot(t_star, [relative_L2_error(C_pred_1hr[:, num], C_star[:, num]) for num in range(n_star)], label='1 hour training', linewidth=2.5, color='g')
axs[0, 1].set_xlabel('$t$', fontsize=15)
axs[0, 1].set_ylabel('Rel. $L_2$ Error', fontsize=15)
axs[0, 1].set_title('$c(t,x,y,z)$', fontsize=15)
axs[0, 1].tick_params(axis='both', which='major', labelsize=13)

# Plot for pressure
axs[0, 0].plot(t_star, [relative_L2_error(P_pred_1hr[:, num] - np.mean(P_pred_1hr[:, num]), P_star[:, num] - np.mean(P_star[:, num])) for num in range(n_star)], label='1 hour training', linewidth=2.5, color='m')
axs[0, 0].set_xlabel('$t$', fontsize=15)
axs[0, 0].set_ylabel('Rel. $L_2$ Error', fontsize=15)
axs[0, 0].set_title('$p(t,x,y,z)$', fontsize=15)
axs[0, 0].tick_params(axis='both', which='major', labelsize=13)

# Plot for velocity in x-direction
axs[1, 0].plot(t_star, [relative_L2_error(U_pred_1hr[:, num], U_star[:, num]) for num in range(n_star)], label='1 hour training', linewidth=2.5, color='r')
axs[1, 0].set_xlabel('$t$', fontsize=15)
axs[1, 0].set_ylabel('Rel. $L_2$ Error', fontsize=15)
axs[1, 0].set_title('$u(t,x,y,z)$', fontsize=15)
axs[1, 0].tick_params(axis='both', which='major', labelsize=13)

# Plot for velocity in y-direction
axs[1, 1].plot(t_star, [relative_L2_error(V_pred_1hr[:, num], V_star[:, num]) for num in range(n_star)], label='1 hour training', linewidth=2.5, color='b')
axs[1, 1].set_xlabel('$t$', fontsize=15)
axs[1, 1].set_ylabel('Rel. $L_2$ Error', fontsize=15)
axs[1, 1].set_title('$v(t,x,y,z)$', fontsize=15)
axs[1, 1].tick_params(axis='both', which='major', labelsize=13)  # Increase font size of axis tick labels

# Move legend to the empty subplot
#handles, labels = axs[1, 2].get_legend_handles_labels()
#fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.52, 0.7), fontsize=15)

plt.suptitle('Aneurysm 2D', fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save figure to PNG file
plt.savefig('.../z.png', dpi=300)

plt.show()
