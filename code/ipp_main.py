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
from Robot import *
from Evaluation import Evaluation

#环境边界
ranges = (-10,10,-10,10)
# Create a random enviroment sampled from a GP with an RBF kernel and specified hyperparameters
# mean function 0
# The enviorment will be constrained by a set of uniformly distributed  sample points of size NUM_PTS x NUM_PTS
world = Environment(ranges, NUM_PTS = 20, variance = 100.0, lengthscale = 3.0, visualize = True,seed=1)
evaluation = Evaluation(world)

# Gather some prior observations to train the kernel (optional)
x1observe = np.linspace(ranges[0], ranges[1], 5)
x2observe = np.linspace(ranges[2], ranges[3], 5)
x1observe, x2observe = np.meshgrid(x1observe, x2observe, sparse = False, indexing = 'xy')
data = np.vstack([x1observe.ravel(), x2observe.ravel()]).T
observations = world.sample_value(data)

# Create the point robot
robot = Robot(sample_world = world.sample_value,
              start_loc = (0.0, 0.0, 0.0),
              ranges= (-10., 10., -10., 10.),
              kernel_file = None,
              kernel_dataset = None,
              prior_dataset =  None,
              init_lengthscale = 3.0,
              init_variance = 100.0,
              noise = 0.05,
              path_generator = 'equal_dubins',
              frontier_size = 20,
              horizon_length = 5.0,
              turning_radius = 0.5,
              sample_step = 1.0,
              evaluation = evaluation)

robot.myopic_planner(T = 50)
#robot.plot_information()
robot.visualize_world_model()
#robot.visualize_trajectory()