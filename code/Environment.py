from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm
from GPModel import GPModel


class Environment:
    def __init__(self, ranges, NUM_PTS, variance = 0.5, lengthscale = 1.0, noise = 0.05, visualize = True,seed = None,dim = 2):
        #GP参数
        self.variance = variance
        self.lengthscale = lengthscale
        self.dim = dim

        #4个边界元组
        self.x1min = float(ranges[0])
        self.x1max = float(ranges[1])
        self.x2min = float(ranges[2])
        self.x2max = float(ranges[3])

        # Intialize a GP model of the environment
        self.GP = GPModel(lengthscale = lengthscale, variance = variance)

        #生成一组离散的网格点，均匀地分布在环境中
        # Generate a set of discrete grid points, uniformly spread across the environment
        x1 = np.linspace(self.x1min, self.x1max, NUM_PTS)
        x2 = np.linspace(self.x2min, self.x2max, NUM_PTS)
        x1vals, x2vals = np.meshgrid(x1, x2, sparse = False, indexing = 'xy') # dimension: NUM_PTS x NUM_PTS
        data = np.vstack([x1vals.ravel(), x2vals.ravel()]).T # dimension: NUM_PTS*NUM_PTS x 2

        # 初始点采样
        xsamples = np.reshape(np.array(data[0, :]), (1, dim)) # dimension: 1 x 2
        #Gpy的predict外包
        mean, var = self.GP.predict_value(xsamples)
        # 生成一个均值loc，标准差variable的正态分布随机数
        zsamples = np.random.normal(loc = mean, scale = np.sqrt(var))
        zsamples = np.reshape(zsamples, (1,1)) # dimension: 1 x 1

        # Add new data point to the GP model
        self.GP.set_data(xsamples, zsamples)

        #按顺序遍历网格的其余部分，并对z采样
        for index, point in enumerate(data[1:, :]):
            # Get a new sample point
            xs = np.reshape(np.array(point), (1, dim))

            # Compute the predicted mean and variance
            #predict是预测当前点的概率密度函数
            mean, var = self.GP.predict_value(xs)

            #对新的观察结果进行采样
            # Sample a new observation, given the mean and variance
            zs = np.random.normal(loc=mean, scale=np.sqrt(var))

            # Add new sample point to the GP model
            zsamples = np.vstack([zsamples, np.reshape(zs, (1, 1))])
            xsamples = np.vstack([xsamples, np.reshape(xs, (1, dim))])
            #将location和value分别加入xvals和zvals
            self.GP.set_data(xsamples, zsamples)

            # Plot the surface mesh and scatter plot representation of the samples points
        if visualize == True:
            # 从机器人模型中生成一组观测值，用于绘制等高线图
            x1plt = np.linspace(ranges[0], ranges[1], 100)
            x2plt = np.linspace(ranges[2], ranges[3], 100)
            # dimension: NUM_PTS x NUM_PTS
            x1, x2 = np.meshgrid(x1plt, x2plt, sparse=False, indexing='xy')
            data = np.vstack([x1.ravel(), x2.ravel()]).T
            observations, var = self.GP.predict_value(data)
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            ax1.set_xlim(ranges[0:2])
            ax1.set_ylim(ranges[2:])
            ax1.set_title('Countour Plot of the real World Model')
            # 可视化机器人看到的世界模型
            plot = ax1.contourf(x1, x2, observations.reshape(x1.shape), cmap='viridis')
            plt.show()

        print("Environment initialized with bounds X1: (", self.x1min, ",", self.x1max, ")  X2:(", self.x2min, ",", self.x2max, ")")

    def sample_value(self,xvals):
        assert(xvals.shape[0] >= 1)
        assert(xvals.shape[1] == self.dim)
        ##predict是预测当前点的概率密度函数
        mean, var = self.GP.predict_value(xvals)
        return mean