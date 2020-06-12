from matplotlib import pyplot as plt
import numpy as np
from GPModel import GPModel
from matplotlib import cm

class Environment:
    def __init__(self, ranges, NUM_PTS, variance, lengthscale, noise = 0.0001, visualize = True, seed = None, dim = 2):
        '''生成real world'''
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
        self.GP = GPModel(ranges=ranges,lengthscale = lengthscale, variance = variance)

        #生成一组离散的网格点，均匀地分布在环境中
        x1 = np.linspace(self.x1min, self.x1max, NUM_PTS)
        x2 = np.linspace(self.x2min, self.x2max, NUM_PTS)
        # dimension: NUM_PTS x NUM_PTS
        x1vals, x2vals = np.meshgrid(x1, x2, sparse = False, indexing = 'xy')
        # dimension: NUM_PTS*NUM_PTS x 2
        data = np.vstack([x1vals.ravel(), x2vals.ravel()]).T

        # 初始点采样，dimension: 1 x 2
        xsamples = np.reshape(np.array(data[0, :]), (1, dim)) #
        #在初始（0,100）下利用采样数据，预测新的均值方差，predict是预测当前点的概率密度函数
        mean, var = self.GP.predict_value(xsamples)
        # 生成一个均值loc，标准差variable的正态分布随机数，这里将均值置0，loc = mean
        zsamples = np.random.normal(loc = 0, scale = np.sqrt(var))
        # dimension: 1 x 1
        zsamples = np.reshape(zsamples, (1,1))
        # 将数据加入高斯模型（先验），并更新
        self.GP.set_data(xsamples, zsamples)
        #按顺序遍历网格的其余部分，并对z采样
        for index, point in enumerate(data[1:, :]):
            # 从余下点中获取一个新点
            xs = np.reshape(np.array(point), (1, dim))
            mean, var = self.GP.predict_value(xs)
            #生成一个符合该概率密度的随机数，也就是观测值
            zs = np.random.normal(loc=mean, scale=np.sqrt(var))
            #向GP模型中添加新的样本点
            zsamples = np.vstack([zsamples, np.reshape(zs, (1, 1))])
            xsamples = np.vstack([xsamples, np.reshape(xs, (1, dim))])
            #将location和value分别加入xvals和zvals
            self.GP.set_data(xsamples, zsamples)
        #可视化real world
        if visualize == True:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection = '3d')
            ax.set_title('Surface of the Simulated Environment')
            #real world 表面
            surf = ax.plot_surface(x1vals, x2vals, zsamples.reshape(x1vals.shape), cmap = cm.coolwarm, linewidth = 1)
            fig2 = plt.figure(figsize=(8, 6))
            ax2 = fig2.add_subplot(111)
            ax2.set_title('Countour Plot of the Simulated Environment')
            #等高线
            plot = ax2.contourf(x1vals, x2vals, zsamples.reshape(x1vals.shape), cmap = 'viridis')
            #均匀采样散点
            scatter = ax2.scatter(data[:, 0], data[:, 1], c = zsamples.ravel(), s = 4.0, cmap = 'viridis')
            maxind = np.argmax(zsamples)
            #最大值（最大误差）点
            ax2.scatter(xsamples[maxind, 0], xsamples[maxind,1], color = 'k', marker = '*', s = 500)
            plt.show()
        print("Environment initialized with bounds X1: (", self.x1min, ",", self.x1max, ")  X2:(", self.x2min, ",", self.x2max, ")")

    def sample_value(self,xvals):
        assert(xvals.shape[0] >= 1)
        assert(xvals.shape[1] == self.dim)
        ##predict是预测当前点的概率密度函数
        mean, var = self.GP.predict_value(xvals)
        return mean