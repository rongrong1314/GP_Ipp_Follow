from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
from sklearn import mixture
from IPython.display import display
from scipy.stats import multivariate_normal
import numpy as np
import math
import os
import GPy

class GPModel:
    def __init__(self, xvals, zvals, lengthscale, variance, dimension = 2, noise = 0.05, kernel = 'rbf'):
        # The dimension of the evironment
        self.dim = dimension
        # The noise parameter of the sensor
        self.nosie = noise

        if kernel == 'rbf':
            self.kern = GPy.kern.RBF(input_dim=self.dim, lengthscale=lengthscale, variance=variance)
        else:
            raise ValueError('Kernel type must by \'rbf\'')

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

            # Constrain the hyperparameters during optmization
            self.m.constrain_positive('')
            self.m['rbf.variance'].constrain_bounded(0.01, 10)
            self.m['rbf.lengthscale'].constrain_bounded(0.01, 10)
            self.m['Gaussian_noise.variance'].constrain_fixed(noise)

            # Train the kernel hyperparameters
            self.m.optimize_restarts(num_restarts=1, messages=True)

            # Save the hyperparemters to file
            np.save('kernel_model.npy', self.m.param_array)

            # Visualize the learned GP kernel
            def kernel_plot(self):
                _ = self.kern.plot()
                plt.ylim([-1, 1])
                plt.xlim([-1, 1])
                plt.show()


#将环境表示为按比例混合的高斯分布
def rv(xvals,yvals):
    #创建混合的高斯模型
    C1 = [[10, 0], [0, 10]]
    C2 = [[24., 3], [0.8, 2.1]]
    C3 = [[3, 0], [0, 3]]
    m1 = [3, 8]
    m2 = [-5, -5]
    m3 = [5, -7]

    pos = np.empty(xvals.shape+(2,))
    pos[:, :, 0] = xvals
    pos[:, :, 1] = yvals

    val = 100. * ((1./3.) * 10.* multivariate_normal.pdf(pos, mean = m1, cov = C1) +
            5. * (1./3.) * multivariate_normal.pdf(pos, mean = m2, cov = C2) + 
            5. * (1./3.) * multivariate_normal.pdf(pos, mean = m3, cov = C3))
    return val

# Sensors have access to a noisy version of the true environmental distirbution
def rv_sample(xvals, yvals):
    data = rv(xvals, yvals)
    return rv(xvals, yvals) + np.random.randn(xvals.shape[0], xvals.shape[1]) * 0.35

class Evironment:
    def __init__(self,ranges,):
        #4个边界元组
        self.xmin = float(ranges[0])
        self.xmax = float(ranges[1])
        self.ymin = float(ranges[2])
        self.ymax = float(ranges[3])

        self.rv = rv
        self.rv_noisy = rv_sample
        print("Environment onitialized with "
              "bounds X: (", self.xmin, ",", self.xmax, ")  Y:(", self.ymin, ",", self.ymax, ")")

    #生成数据
    def initializeGP(self, ranges, training_points):
        #采样和输出2D数据
        np.random.seed(0)
        x = np.linspace(ranges[0], ranges[1], 100)
        y = np.linspace(ranges[2], ranges[3], 100)
        xvals, yvals = np.meshgrid(x, y, sparse=False, indexing='xy')
        #计算混合高斯分布的概率密度函数值
        zvals = rv(xvals, yvals)

        xtrain = np.linspace(ranges[0], ranges[1], training_points)
        ytrain = np.linspace(ranges[2], ranges[3], training_points)
        xtrain, ytrain= np.meshgrid(xtrain, ytrain, sparse = False, indexing = 'xy')
        data = np.vstack([xtrain.ravel(), ytrain.ravel()]).T
        ztrain = rv_sample(xtrain, ytrain)
        #创建和训练GP模型参数
        self.GP = GPModel(data, np.reshape(ztrain, (data.shape[0], 1)), lengthscale=10.0, variance=0.5)

        fig = plt.figure()
        ax = fig.add_subplot(211, projection = '3d')
        surf = ax.plot_surface(xvals, yvals, zvals, cmap = cm.coolwarm, linewidth = 0)

        ax1 = fig.add_subplot(312, projection='3d')
        surf1 = ax1.plot_surface(xtrain, ytrain, ztrain, cmap=cm.coolwarm,linewidth = 0)

        ax2 = fig.add_subplot(313, projection = '3d')
        scatter = ax2.scatter(xtrain, ytrain, ztrain, cmap=cm.coolwarm)
        plt.show()

class Robot:
    def __init__(self, start_loc):
        self.start_loc = start_loc

        #环境边界
ranges = (-10,10,-10,10)
#训练GP的点数
training_points = 25

#使用噪声观测来训练GP
world = Evironment(ranges)
world.initializeGP(ranges, training_points)

# Create the point robot
robot = Robot([0, 0])
