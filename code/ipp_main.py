'''
Lengthscale:描述光滑函数。较小的纵向值表示函数值变化快，较大的纵向值表示函数值变化慢。纵向尺度也决定了我们能
从训练数据可靠地推断出多远。
Signal variance:是一个比例因子。它决定了函数值与平均值的偏差。小的值保持接近他们的平均值,更大的值允许更多的
变化。如果信号方差过大，所建立的函数可以自由地去追赶离群值。
噪声方差：是协方差函数本身的一部分。它被用于高斯过程模型，以考虑训练数据中存在的噪声。此参数指定数据中预期出现
的噪声量。
'''
from Robot import *
from Evaluation import Evaluation

#环境边界
ranges = (0.,10.,0.,10.)
#奖励函数
reward_function = 'mes'
#建立real world ， NUM_PTS x NUM_PTS
world = Environment(ranges, NUM_PTS = 20, variance = 100.0, lengthscale = 1.0, visualize = True,seed=3)
evaluation = Evaluation(world,reward_function=reward_function)
# Create the point robot
robot = Robot(sample_world = world.sample_value,
              start_loc = (5.0, 5.0, 0.0),
              ranges= ranges,
              kernel_file = None,
              kernel_dataset = None,
              prior_dataset =  None,
              init_lengthscale = 1.0,
              init_variance = 100.0,
              noise = 0.0001,
              path_generator = 'dubins',
              frontier_size = 20,
              horizon_length = 5.0,
              turning_radius = 0.1,
              sample_step = 1.5,
              evaluation = evaluation,
              f_rew = reward_function,
              create_animation = True)

#算法程序
robot.myopic_planner(T = 50)