from matplotlib import pyplot as plt
import numpy as np
import os
import GPy

class GPModel:
    #xvals:表示观察位置
    #zvals：表示传感器观测的date
    def __init__(self, ranges, lengthscale, variance, noise = 0.0001, dimension = 2, kernel = 'rbf'):

        self.dim = dimension
        self.noise = noise
        self.lengthscale = lengthscale
        self.variance = variance
        self.ranges = ranges

        # The Gaussian dataset
        self.xvals = None
        self.zvals = None

        if kernel == 'rbf':
            self.kern = GPy.kern.RBF(input_dim=self.dim, lengthscale=lengthscale, variance=variance)
        else:
            raise ValueError('Kernel type must by \'rbf\'')

        self.model = None

    def predict_value(self, xvals):
        ''' Public method returns the mean and variance predictions at a set of input locations.
        Inputs:
        * xvals (float array): an nparray of floats representing observation locations, with dimension NUM_PTS x 2

        Returns:
        * mean (float array): an nparray of floats representing predictive mean, with dimension NUM_PTS x 1
        * var (float array): an nparray of floats representing predictive variance, with dimension NUM_PTS x 1 '''

        assert (xvals.shape[0] >= 1)
        assert (xvals.shape[1] == self.dim)

        n_points, input_dim = xvals.shape

        # With no observations, predict 0 mean everywhere and prior variance
        if self.model == None:
            return np.zeros((n_points, 1)), np.ones((n_points, 1)) * self.variance

        # Else, return
        mean, var = self.model.predict(xvals, full_cov=False, include_likelihood=True)
        return mean, var

    def set_data(self, xvals, zvals):
        ''' Public method that updates the data in the GP model.
        Inputs:
        * xvals (float array): an nparray of floats representing observation locations, with dimension NUM_PTS x 2
        * zvals (float array): an nparray of floats representing sensor observations, with dimension NUM_PTS x 1 '''
        # Save the data internally
        self.xvals = xvals
        self.zvals = zvals

        # If the model hasn't been created yet (can't be created until we have data), create GPy model
        if self.model == None:
            self.model = GPy.models.GPRegression(np.array(xvals), np.array(zvals), self.kern)
        # Else add to the exisiting model
        else:
            self.model.set_XY(X=np.array(xvals), Y=np.array(zvals))
        return

    def add_data(self, xvals, zvals):
        ''' Public method that adds data to an the GP model.
        Inputs:
        * xvals (float array): 观测位置，NUM_PTS x 2
        * zvals (float array): 观测值，NUM_PTS x 1 '''

        if self.xvals is None:
            self.xvals = xvals
        else:
            self.xvals = np.vstack([self.xvals, xvals])

        if self.zvals is None:
            self.zvals = zvals
        else:
            self.zvals = np.vstack([self.zvals, zvals])
        if self.model == None:
            self.model = GPy.models.GPRegression(np.array(xvals), np.array(zvals), self.kern)
        else:
            self.model.set_XY(X=np.array(self.xvals), Y=np.array(self.zvals))


    def load_kernel(self, kernel_file='kernel_model.npy'):
        ''' Public method that loads kernel parameters from file.
        Inputs:
        * kernel_file (string): a filename string with the location of the kernel parameters '''

        # Read pre-trained kernel parameters from file, if avaliable and no training data is provided
        if os.path.isfile(kernel_file):
            print("Loading kernel parameters from file")
            self.kern[:] = np.load(kernel_file)
        else:
            raise ValueError("Failed to load kernel. Kernel parameter file not found.")
        return

    def train_kernel(self, xvals=None, zvals=None, kernel_file='kernel_model.npy'):
        ''' Public method that optmizes kernel parameters based on input data and saves to files.
        Inputs:
        * xvals (float array): 观察位置，NUM_PTS x 2
        * zvals (float array): 传感器观察值，NUM_PTS x 1
        * kernel_file (string): a filename string with the location to save the kernel parameters '''
        # 如果没提供训练数据，读取预先训练好的内核数据
        if xvals is not None and zvals is not None:
            print("Optimizing kernel parameters given data")
            # Initilaize a GP model (used only for optmizing kernel hyperparamters)
            self.m = GPy.models.GPRegression(np.array(xvals), np.array(zvals), self.kern)
            self.m.initialize_parameter()

            # Constrain the hyperparameters during optmization
            self.m.constrain_positive('')
            # self.m['rbf.variance'].constrain_bounded(0.01, 10)
            # self.m['rbf.lengthscale'].constrain_bounded(0.01, 10)
            self.m['Gaussian_noise.variance'].constrain_fixed(self.noise)

            # Train the kernel hyperparameters
            self.m.optimize_restarts(num_restarts=2, messages=True)

            # Save the hyperparemters to file
            np.save(kernel_file, self.kern[:])
        else:
            raise ValueError("Failed to train kernel. No training data provided.")

        # Visualize the learned GP kernel
    def kernel_plot(self):
        _ = self.kern.plot()
        plt.ylim([-1, 1])
        plt.xlim([-1, 1])
        plt.show()