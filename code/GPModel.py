from matplotlib import pyplot as plt
import numpy as np
import os
import GPy

class GPModel:
    #xvals:表示观察位置
    #zvals：表示传感器观测的date
    def __init__(self, xvals, zvals, learn_kernel, lengthscale, variance,  noise = 0.05, dimension = 2, kernel = 'rbf'):
        # The dimension of the evironment
        self.dim = dimension
        # The noise parameter of the sensor
        self.nosie = noise

        if kernel == 'rbf':
            self.kern = GPy.kern.RBF(input_dim=self.dim, lengthscale=lengthscale, variance=variance)
        else:
            raise ValueError('Kernel type must by \'rbf\'')

        if learn_kernel:
            # Read pre-trained kernel parameters from file, if avaliable
            if os.path.isfile('kernel_model.npy'):
                print("Loading kernel parameters from file")
                # Initialize GP model from file
                #GP回归初始化
                self.m = GPy.models.GPRegression(np.array(xvals), np.array(zvals), self.kern, initialize=False)
                #不更新模型
                self.m.update_model(False)
                #初始化参数
                self.m.initialize_parameter()
                #下载数据后更新
                self.m[:] = np.load('kernel_model.npy')
                self.m.update_model(True)

            else:
                print("Optimizing kernel parameters")
                # Initilaize GP 回归 model
                self.m = GPy.models.GPRegression(np.array(xvals), np.array(zvals), self.kern)
                self.m.initialize_parameter()
                # Constrain the hyperparameters during optmization
                self.m.constrain_positive('')
                self.m['rbf.variance'].constrain_bounded(0.01, 10)
                self.m['rbf.lengthscale'].constrain_bounded(0.01, 10)
                self.m['Gaussian_noise.variance'].constrain_fixed(noise)
                # Train the kernel hyperparameters
                self.m.optimize_restarts(num_restarts=2, messages=True)
                # Save the hyperparemters to file
                np.save('kernel_model.npy', self.m.param_array)
        else:
            # Directly initilaize GP model
            self.m = GPy.models.GPRegression(np.array(xvals), np.array(zvals), self.kern)
            self.m.initialize_parameter()


    # Visualize the learned GP kernel
    def kernel_plot(self):
        _ = self.kern.plot()
        plt.ylim([-1, 1])
        plt.xlim([-1, 1])
        plt.show()