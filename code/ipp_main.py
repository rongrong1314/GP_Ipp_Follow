'''
Lengthscale:描述光滑函数。较小的纵向值表示函数值变化快，较大的纵向值表示函数值变化慢。纵向尺度也决定了我们能
从训练数据可靠地推断出多远。
Signal variance:是一个比例因子。它决定了函数值与平均值的偏差。小的值保持接近他们的平均值,更大的值允许更多的
变化。如果信号方差过大，所建立的函数可以自由地去追赶离群值。
噪声方差：是协方差函数本身的一部分。它被用于高斯过程模型，以考虑训练数据中存在的噪声。此参数指定数据中预期出现
的噪声量。
'''

from Environment import Environment
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

#环境边界
ranges = (-10,10,-10,10)
# Create a random enviroment sampled from a GP with an RBF kernel and specified hyperparameters
# mean function 0
# The enviorment will be constrained by a set of uniformly distributed  sample points of size NUM_PTS x NUM_PTS
world = Environment(ranges, NUM_PTS = 20, variance = 100.0, lengthscale = 3.0, visualize = True)


#训练GP的点数
training_points = 25

#使用噪声观测来训练GP
world = Environment(ranges, NUM_PTS = 20, variance = 100.0, lengthscale = 3.0, visualize = True)
# Generate observations at random locations in environment and plot resulting predicted model
x1observe = np.linspace(ranges[0], ranges[1], 5)
x2observe = np.linspace(ranges[2], ranges[3], 5)
x1observe, x2observe= np.meshgrid(x1observe, x2observe, sparse = False, indexing = 'xy')
data = np.vstack([x1observe.ravel(), x2observe.ravel()]).T
observations = world.sample_value(data)

fig = plt.figure()
ax1 = fig.add_subplot(111, projection = '3d')
surf = ax1.plot_surface(x1observe, x2observe, observations.reshape(x1observe.shape), cmap = cm.coolwarm, linewidth = 0)
plt.show()