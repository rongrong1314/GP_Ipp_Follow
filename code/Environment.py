from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm
from GPModel import GPModel
from scipy.stats import multivariate_normal

class Environment:
    def __init__(self, ranges, NUM_PTS, variance = 0.5, lengthscale = 1.0, noise = 0.05, visualize = True, dim = 2):
        #GP参数
        self.variance = variance
        self.lengthscale = lengthscale
        self.dim = dim

        #4个边界元组
        self.x1min = float(ranges[0])
        self.x1max = float(ranges[1])
        self.x2min = float(ranges[2])
        self.x2max = float(ranges[3])

        #生成一组离散的网格点，均匀地分布在环境中
        # Generate a set of discrete grid points, uniformly spread across the environment
        x1 = np.linspace(self.x1min, self.x1max, NUM_PTS)
        x2 = np.linspace(self.x2min, self.x2max, NUM_PTS)
        x1vals, x2vals = np.meshgrid(x1, x2, sparse = False, indexing = 'xy') # dimension: NUM_PTS x NUM_PTS
        data = np.vstack([x1vals.ravel(), x2vals.ravel()]).T # dimension: NUM_PTS*NUM_PTS x 2

        # 初始点采样
        xsamples = np.reshape(np.array(data[0, :]), (1, dim)) # dimension: 1 x 2
        #生成一个均值loc，标准差variable的正态分布随机数
        zsamples = np.reshape(np.random.normal(loc = 0, scale = variance), (1,1)) # dimension: 1 x 1

        # Initialze a GP model with a first sampled point
        self.GP = GPModel(xsamples, zsamples, learn_kernel = False, lengthscale = lengthscale, variance = variance)

        #按顺序遍历网格的其余部分，并对z采样
        for index, point in enumerate(data[1:, :]):
            # Get a new sample point
            xs = np.reshape(np.array(point), (1, dim))

            # Compute the predicted mean and variance
            #predict是预测当前点的概率密度函数
            mean, var = self.GP.m.predict(xs, full_cov=False, include_likelihood=True)

            #对新的观察结果进行采样
            # Sample a new observation, given the mean and variance
            zs = np.random.normal(loc=mean, scale=var)

            # Add new sample point to the GP model
            zsamples = np.vstack([zsamples, np.reshape(zs, (1, 1))])
            xsamples = np.vstack([xsamples, np.reshape(xs, (1, dim))])
            #设置GP的输入输出
            self.GP.m.set_XY(X=xsamples, Y=zsamples)

            # Plot the surface mesh and scatter plot representation of the samples points
        if visualize == True:
            fig = plt.figure()
            ax = fig.add_subplot(211, projection='3d')

            surf = ax.plot_surface(x1vals, x2vals, zsamples.reshape(x1vals.shape), cmap=cm.coolwarm, linewidth=1)

            ax2 = fig.add_subplot(212, projection='3d')
            scatter = ax2.scatter(data[:, 0], data[:, 1], zsamples, cmap=cm.coolwarm)
            plt.show()

        print("Environment initialized with bounds X1: (", self.x1min, ",", self.x1max, ")  X2:(", self.x2min, ",", self.x2max, ")")

    def sample_value(self,point_set):
        assert(point_set.shape[0] >= 1)
        assert(point_set.shape[1] == self.dim)
        ##predict是预测当前点的概率密度函数
        mean, var = self.GP.m.predict(point_set, full_cov = False, include_likelihood = True)
        return mean