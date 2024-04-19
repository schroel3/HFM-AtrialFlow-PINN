"""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io
import time
import sys
import matplotlib.pyplot as plt
from matplotlib import pyplot
from utilities import numericalSort, relative_error, neural_net, mean_squared_error, Navier_Stokes_2D, tf_session, Gradient_Velocity_2D

tf.disable_eager_execution()
tf.config.list_physical_devices('GPU')

# Set up Tensorflow to use GPU memory growth to allocate GPU memory as needed
# ConfigProto allocates operations to CPU/GPU + customize Tensorflow session
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.InteractiveSession(config=config)
#tf.test.is_gpu_available()


class AtrialFlow(object):
    # notational conventions:
    # _tf: placeholders for input/output data and points used to regress NS equations
    # _pred: output of the neural network
    # _eqns: points used to regress NS equations
    # _data: input-output data
    # _star: predictions of PINN
    # _cyl: used to impose no-slip boundary on atrial boundary
    # _inlet: used to include four mitral valve inlets

    def __init__(self, t_data, x_data, y_data, c_data,
                       t_eqns, x_eqns, y_eqns,
	                   t_inlet, x_inlet, y_inlet, u_inlet, v_inlet,
                       t_cyl, x_cyl, y_cyl,
                       layers, batch_size,
                       Pec, Rey):

        # specificiations of the neural network
        self.layers = layers
        self.batch_size = batch_size
        
        # data
        [self.t_data, self.x_data, self.y_data, self.c_data] = [t_data, x_data, y_data, c_data]
        [self.t_eqns, self.x_eqns, self.y_eqns] = [t_eqns, x_eqns, y_eqns]
        [self.t_inlet, self.x_inlet, self.y_inlet, self.u_inlet, self.v_inlet] = [t_inlet, x_inlet, y_inlet, u_inlet, v_inlet]
        [self.t_cyl, self.x_cyl, self.y_cyl] = [t_cyl, x_cyl, y_cyl]

        # placeholders
        [self.t_data_tf, self.x_data_tf, self.y_data_tf, self.c_data_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(4)]
        [self.t_eqns_tf, self.x_eqns_tf, self.y_eqns_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
        [self.t_inlet_tf, self.x_inlet_tf, self.y_inlet_tf, self.u_inlet_tf, self.v_inlet_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(5)]
        [self.t_cyl_tf, self.x_cyl_tf, self.y_cyl_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]

        # physics "uninformed" neural networks (predict from data within atrial boundary)
        self.net_cuvp = neural_net(self.t_data, self.x_data, self.y_data, layers = self.layers)

        [self.c_data_pred,
         self.u_data_pred,
         self.v_data_pred,
         self.p_data_pred] = self.net_cuvp(self.t_data_tf,
                                           self.x_data_tf,
                                           self.y_data_tf)

        # physics "uninformed" neural networks (predict from data at the inlets)
        [_,
         self.u_inlet_pred,
         self.v_inlet_pred,
         _] = self.net_cuvp(self.t_inlet_tf,
                            self.x_inlet_tf,
                            self.y_inlet_tf)

        # physics "uninformed" neural networks (predict from data at the atrial boundary)
        [_,
         self.u_cyl_pred,
         self.v_cyl_pred,
         _] = self.net_cuvp(self.t_cyl_tf,
                            self.x_cyl_tf,
                            self.y_cyl_tf)

        # physics "informed" neural networks (predict with constraint from NS equations)
        [self.c_eqns_pred,
         self.u_eqns_pred,
         self.v_eqns_pred,
         self.p_eqns_pred] = self.net_cuvp(self.t_eqns_tf,
                                           self.x_eqns_tf,
                                           self.y_eqns_tf)
        [self.e1_eqns_pred,
         self.e2_eqns_pred,
         self.e3_eqns_pred,
         self.e4_eqns_pred] = Navier_Stokes_2D(self.c_eqns_pred,
                                               self.u_eqns_pred,
                                               self.v_eqns_pred,
                                               self.p_eqns_pred,
                                               self.t_eqns_tf,
                                               self.x_eqns_tf,
                                               self.y_eqns_tf,
                                               self.Pec,
                                               self.Rey)

        # loss function: 
        # introduces no-slip condition at atrial boundary and Dirichlet condition at inlets 
        # U,V,P data is 'hidden' from model and instead derivatives are subsituted into NS equations used in the loss
        # Only passive scalar concentration is 'seen' by the model
        self.loss = mean_squared_error(self.c_data_pred, self.c_data_tf) + \
                    mean_squared_error(self.u_inlet_pred, self.u_inlet_tf) + \
                    mean_squared_error(self.v_inlet_pred, self.v_inlet_tf) + \
                    mean_squared_error(self.u_cyl_pred, 0.0) + \
                    mean_squared_error(self.v_cyl_pred, 0.0) + \
                    mean_squared_error(self.e1_eqns_pred, 0.0) + \
                    mean_squared_error(self.e2_eqns_pred, 0.0) + \
                    mean_squared_error(self.e3_eqns_pred, 0.0) + \
                    mean_squared_error(self.e4_eqns_pred, 0.0)

        # optimizers
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

        # initialise TensorFlow session and save training with checkpoint every hour during training
        self.sess = tf_session()
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
        self.saver.save(self.sess,'.../save_model')

    def train(self, total_time, learning_rate):
        # intialises model training
        # input variables:
        #    self: model
        #    total_time: training duration (hrs)
        #    learning_rate: learning rate (usually 1e-3 for Adam optimizer) 

        N_data = self.t_data.shape[0]
        N_eqns = self.t_eqns.shape[0]
        
        start_time = time.time()
        running_time = 0
        it = 0
        while running_time < total_time:

            # select a random mini-batch
            idx_data = np.random.choice(N_data, self.batch_size)
            idx_eqns = np.random.choice(N_eqns, self.batch_size)

            (t_data_batch,
             x_data_batch,
             y_data_batch,
             c_data_batch) = (self.t_data[idx_data,:],
                              self.x_data[idx_data,:],
                              self.y_data[idx_data,:],
                              self.c_data[idx_data,:])

            (t_eqns_batch,
             x_eqns_batch,
             y_eqns_batch) = (self.t_eqns[idx_eqns,:],
                              self.x_eqns[idx_eqns,:],
                              self.y_eqns[idx_eqns,:])

            tf_dict = {self.t_data_tf: t_data_batch,
                       self.x_data_tf: x_data_batch,
                       self.y_data_tf: y_data_batch,
                       self.c_data_tf: c_data_batch,
                       self.t_eqns_tf: t_eqns_batch,
                       self.x_eqns_tf: x_eqns_batch,
                       self.y_eqns_tf: y_eqns_batch,
                       self.t_inlet_tf: self.t_inlet,
                       self.x_inlet_tf: self.x_inlet,
                       self.y_inlet_tf: self.y_inlet,
                       self.u_inlet_tf: self.u_inlet,
                       self.v_inlet_tf: self.v_inlet,
                       self.t_cyl_tf: self.t_cyl,
                       self.x_cyl_tf: self.x_cyl,
                       self.y_cyl_tf: self.y_cyl,
                       self.learning_rate: learning_rate}

            self.sess.run([self.train_op], tf_dict)
            self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
            self.saver.save(self.sess,'best_model')

            # Printed during training for each iteration
            if it % 10 == 0:
                elapsed = time.time() - start_time
                running_time += elapsed/3600.0
                [loss_value,
                 learning_rate_value] = self.sess.run([self.loss,
                                                       self.learning_rate], tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2fs, Running Time: %.2fh, Learning Rate: %.1e'
                       %(it, loss_value, elapsed, running_time, learning_rate_value))
                sys.stdout.flush()
                start_time = time.time()
            it += 1


    def predict(self, t_star, x_star, y_star):
        # model prediction on test dataset
        # input variables:
        #    self: model
        #    _star: spatiotemporal coordinates from test data

        tf_dict = {self.t_data_tf: t_star, self.x_data_tf: x_star, self.y_data_tf: y_star}

        c_star = self.sess.run(self.c_data_pred, tf_dict)
        u_star = self.sess.run(self.u_data_pred, tf_dict)
        v_star = self.sess.run(self.v_data_pred, tf_dict)
        p_star = self.sess.run(self.p_data_pred, tf_dict)

        return c_star, u_star, v_star, p_star

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

file_directory = '.../LeftAtrium2D/'

if __name__ == "__main__":

    batch_size = 10000
    
    layers = [3] + 10*[4*50] + [4]

    # Load Data
    data = scipy.io.loadmat('.../LAA_patient.mat')

    t_star = data['t_star'] # T x 1 = (150,1)
    x_star = data['x_star'] # N x T = (66540,150)
    y_star = data['y_star'] # N x T

    T = t_star.shape[0]  # no. of time steps = 150
    N = x_star.shape[0]  # no. of spatial points = 66540

    U_star = data['U_star'] # N x T = (66540,150)
    V_star = data['V_star'] # N x T 
    P_star = data['P_star'] # N x T
    C_star = data['C_star'] # N x T
    patch_ID = data['patch_ID'] # contains patch IDs for boundary,inlet/outlets

    # Rearrange Data
    T_star = np.tile(t_star, (1,N)).T # N x T = (66540,150)
    X_star = np.tile(x_star, (1,T)) # (66540,22500)
    Y_star = np.tile(y_star, (1,T)) # (66540,22500)

    # flatten arrays to one column (length NT)
    t = T_star.flatten()[:,None] # NT x 1 = (9981000,0)
    x = X_star.flatten()[:,None] # NT x 1
    y = Y_star.flatten()[:,None] # NT x 1
    u = U_star.flatten()[:,None] # NT x 1
    v = V_star.flatten()[:,None] # NT x 1
    p = P_star.flatten()[:,None] # NT x 1
    c = C_star.flatten()[:,None] # NT x 1
    
    ######################################################################

    # Prepare training data:
    T_data = T # 150
    N_data = N # 66540

    idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_data-2, replace=False)+1, np.array([T-1])] ) #(150,)
    idx_x = np.random.choice(N, N_data, replace=False) # (66540,)
    t_data = T_star[:, idx_t][idx_x,:].flatten()[:,None] #(9981000,1)
    x_data = X_star[:, idx_t][idx_x,:].flatten()[:,None] #(9981000,1)
    y_data = Y_star[:, idx_t][idx_x,:].flatten()[:,None] #(9981000,1)
    c_data = C_star[:, idx_t][idx_x,:].flatten()[:,None] #(9981000,1)

    T_eqns = T
    N_eqns = N
	idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_eqns-2, replace=False)+1, np.array([T-1])] )
    idx_x = np.random.choice(N, N_eqns, replace=False)
    t_eqns = T_star[:, idx_t][idx_x,:].flatten()[:,None] #(9981000,1)
    x_eqns = X_star[:, idx_t][idx_x,:].flatten()[:,None] #(9981000,1)
    y_eqns = Y_star[:, idx_t][idx_x,:].flatten()[:,None] #(9981000,1)

    # Training Data on velocity at inlets
    t_inlet = t[(np.where(patch_ID == 2)) or (np.where(patch_ID == 3)) or (np.where(patch_ID == 4)) or (np.where(patch_ID == 5))][:,None]
    x_inlet = x[(np.where(patch_ID == 2)) or (np.where(patch_ID == 3)) or (np.where(patch_ID == 4)) or (np.where(patch_ID == 5))][:,None]
    y_inlet = y[(np.where(patch_ID == 2)) or (np.where(patch_ID == 3)) or (np.where(patch_ID == 4)) or (np.where(patch_ID == 5))][:,None]
    u_inlet = u[(np.where(patch_ID == 2)) or (np.where(patch_ID == 3)) or (np.where(patch_ID == 4)) or (np.where(patch_ID == 5))][:,None]
    v_inlet = v[(np.where(patch_ID == 2)) or (np.where(patch_ID == 3)) or (np.where(patch_ID == 4)) or (np.where(patch_ID == 5))][:,None]

    # Training Data on velocity at atrial boundary
    t_cyl = t[(np.where(patch_ID==1) or (np.where(patch_ID==7)))][:, None]
    x_cyl = x[(np.where(patch_ID==1)) or (np.where(patch_ID==7))][:, None]
    y_cyl = y[(np.where(patch_ID==1)) or (np.where(patch_ID==7))][:, None]

    # Training
    model = AtrialFlow(t_data, x_data, y_data, c_data,
                       t_eqns, x_eqns, y_eqns,
                       t_inlet, x_inlet, y_inlet, u_inlet, v_inlet,
                       t_cyl, x_cyl, y_cyl,
                       layers, batch_size,
                       Pec = 7.5e6, Rey = 2000)

    model.train(total_time = 1, learning_rate=1e-3)

    # Test Data
    snap = np.array([50])

    t_test = T_star[:,snap] # N x snap = (66540,50)
    x_test = X_star[:,snap] # N x snap = (66540,50)
    y_test = Y_star[:,snap] # N x snap

    c_test = C_star[:,snap] # N x snap
    u_test = U_star[:,snap] # N x snap
    v_test = V_star[:,snap] # N x snap
    p_test = P_star[:,snap] # N x snap

    # Prediction
    c_pred, u_pred, v_pred, p_pred = model.predict(t_test, x_test, y_test)

    # Error
    error_c = relative_error(c_pred, c_test)
    error_u = relative_error(u_pred, u_test)
    error_v = relative_error(v_pred, v_test)
    error_p = relative_error(p_pred - np.mean(p_pred), p_test - np.mean(p_test))

    print('Error c: %e' % (error_c))
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error p: %e' % (error_p))


    ################# Save Data ###########################

    C_pred = 0*C_star # (66540,50)
    U_pred = 0*U_star
    V_pred = 0*V_star
    P_pred = 0*P_star

    C_error = 0*C_star # (66540,50)
    U_error = 0*U_star
    V_error = 0*V_star
    P_error = 0*P_star

    fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2)
    plt.figure(figsize=(25, 20))
    fig.suptitle('Errors for each Snapshot in Time')

       for snap in range(0,t_star.shape[0]): # takes for every snap between 0-50
        t_test = T_star[:,snap:snap+1] # e.g. first iteration: (66540,0:1)
        x_test = X_star[:,snap:snap+1]
        y_test = Y_star[:,snap:snap+1]

        c_test = C_star[:,snap:snap+1]
        u_test = U_star[:,snap:snap+1]
        v_test = V_star[:,snap:snap+1]
        p_test = P_star[:,snap:snap+1]

        # Prediction
        c_pred, u_pred, v_pred, p_pred = model.predict(t_test, x_test, y_test)

        C_pred[:,snap:snap+1] = c_pred
        U_pred[:,snap:snap+1] = u_pred
        V_pred[:,snap:snap+1] = v_pred
        P_pred[:,snap:snap+1] = p_pred

        # Error
        error_c = relative_error(c_pred, c_test)
        error_u = relative_error(u_pred, u_test)
        error_v = relative_error(v_pred, v_test)
        error_p = relative_error(p_pred - np.mean(p_pred), p_test - np.mean(p_test))

        # Save error
        C_error[snap] = error_c
        U_error[snap] = error_u
        V_error[snap] = error_v
        P_error[snap] = error_p

        ax1.scatter(snap, error_p, color='purple')
        ax2.scatter(snap, error_c, color='green')
        ax3.scatter(snap, error_u, color='red')
        ax4.scatter(snap, error_v, color='cyan')

        print('Error c: %e' % (error_c))
        print('Error u: %e' % (error_u))
        print('Error v: %e' % (error_v))
        print('Error p: %e' % (error_p))


    # Set titles and labels
    ax1.set_title('Pressure Field')
    ax1.set(xlabel='Snapshot in Time', ylabel='Error')

    ax2.set_title('Concentration Field')
    ax2.set(xlabel='Snapshot in Time', ylabel='Error')

    ax3.set_title('"U" Velocity Field')
    ax3.set(xlabel='Snapshot in Time', ylabel='Error')

    ax4.set_title('"V" Velocity Field')
    ax4.set(xlabel='Snapshot in Time', ylabel='Error')

    fig.show()
    fig.tight_layout()
    fig.savefig(file_directory + 'PINN_loss_%s' % (time.strftime('%d_%m_%Y')))
    pyplot.close()


    scipy.io.savemat('/content/drive/MyDrive/Colab/BEng_Project/HFM/Results/Aneurysm2D/Anuerysm2D_results_%s.mat' %(time.strftime('%d_%m_%Y')),
                     {'C_pred':C_pred, 'U_pred':U_pred, 'V_pred':V_pred,'P_pred':P_pred})

    scipy.io.savemat('/content/drive/MyDrive/Colab/BEng_Project/HFM/Results/Aneurysm2D/Anuerysm2D_errors_%s.mat' %(time.strftime('%d_%m_%Y')),
        {'C_error': C_error, 'U_error': U_error, 'V_error': V_error,'P_error': P_error})
