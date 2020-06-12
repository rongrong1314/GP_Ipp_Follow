from Evaluation import *
from path_generator import *
from Environment import *
import os

class Robot:
    '''Robot class包括车辆当前模型,路径设置，信息路径规划算法
       sample_world (method)：以一组位置作为输入并返回一组观察值的函数句柄
       start_loc (tuple of floats)： 机器人起始位置
       init_variance (float)：平方指数核的方差参数
       init_lengthscale (float) ：平方指数核的长度参数
       noise (float)：传感器噪声参数的平方指数核
       f_rew：mes'''
    def __init__(self, sample_world, start_loc = (0.0, 0.0, 0.0), ranges = (-10., 10., -10., 10.), kernel_file = None,
            kernel_dataset = None, prior_dataset = None, init_lengthscale = 10.0, init_variance = 100.0, noise = 0.05,
            path_generator = 'default', frontier_size = 6, horizon_length = 5, turning_radius = 1, sample_step = 0.5,
            evaluation = None, f_rew = 'mean', create_animation = False):
        ''' Initialize the robot class with a GP model, initial location, path sets, and prior dataset'''
        self.ranges = ranges
        self.create_animation = create_animation
        self.eval = evaluation
        # 初始机器人位置
        self.loc = start_loc
        self.sample_world = sample_world
        self.f_rew = f_rew
        self.aquisition_function = mves
        #初始化机器人GP模型的核函数，model为none
        self.GP = GPModel(ranges = ranges, lengthscale = init_lengthscale, variance = init_variance)
        # 几种路径生成算法
        path_options = {
            'default': Path_Generator(frontier_size, horizon_length, turning_radius, sample_step, ranges),
            'dubins': Dubins_Path_Generator(frontier_size, horizon_length, turning_radius, sample_step, ranges),
            'equal_dubins': Dubins_EqualPath_Generator(frontier_size, horizon_length, turning_radius, sample_step,
                                                           ranges)}
        self.path_generator = path_options[path_generator]
    #主算法
    def myopic_planner(self, T):
        ''' Gather noisy samples of the environment and updates the robot's GP model
        Input:
        * T (int > 0): the length of the planning horization (number of planning iterations)'''
        self.trajectory = []

        for t in range(T):
            # Select the best trajectory according to the robot's aquisition function
            print("[", t, "] Current Location:  ", self.loc)
            #用UCB对路径进行评估，并选择最好的路径作为best_path
            #选择当前点下20个方向中最优的一个点方向路径
            best_path, best_val, all_paths, all_values, max_locs = self.choose_trajectory(t=t)
            # 更新评估指标，TODO
            self.eval.update_metrics(t, self.GP, all_paths, best_path)
            if best_path == None:
                break
            data = np.array(best_path)
            x1 = data[:, 0]
            x2 = data[:, 1]
            xlocs = np.vstack([x1, x2]).T
            #返回当前路径所有位置点real world观测均值，并更新
            if len(best_path)!=1:
                #添加先验
                self.collect_observations(xlocs)
            self.trajectory.append(best_path)
            #绘制环境和路径
            if self.create_animation:
                Robot.visualize_world_model(self,screen = True, filename = t, maxes = max_locs,all_paths = all_paths,
                    all_vals =all_values , best_path = best_path)
                #算法信息可视化
                #Robot.plot_information()
            #如果最佳路径为1个点
            if len(best_path) == 1:
                self.loc = (best_path[-1][0], best_path[-1][1], best_path[-1][2] - 0.45)
            else:
                self.loc = best_path[-1]

    def choose_trajectory(self,t):
        '''根据aquisition function，选取当前姿态最佳轨迹
        Input:
        * T (int > 0):规划迭代次数
        * t (int > 0):当前的迭代时间'''
        # 当前点对应的20个点相应的路径点
        paths = self.path_generator.get_path_set(self.loc)
        value = {}
        max_locs = None
        #
        if self.f_rew == 'mes':
            #全局最大熵和位置
            max_val, max_locs = sample_max_vals(self.GP)
        print("Max locs:", max_locs)
        for path, points in paths.items():
            if max_val is None:
                value[path]=np.array([1.0])
            else:
                #求αt(x)，xvals观测路径点，maxes当前地图中全局最大值
                value[path] = self.aquisition_function(time=t, xvals=points, robot_model=self.GP, maxes=max_val)
        try:
            return paths[max(value, key = value.get)], value[max(value, key = value.get)], paths, value, max_locs
        except:
            return None

    def collect_observations(self, xobs):
        #world.sample_value
        #返回当前点的世界坐标预测均值，用real world得到该点的值
        # 也就是通常意义上的无误差观测
        zobs = self.sample_world(xobs)
        #将新的最优路径坐标加入model
        self.GP.add_data(xobs, zobs)

    #可视化，TODO
    def visualize_world_model(self, screen = True, filename = 'SUMMARY', maxes = None, all_paths = None,
        all_vals = None, best_path = None):
        #通过在空间上均匀采样点并绘制预测值来实现机器人当前世界模型的可视化
        #从机器人模型中生成一组观测值，用于绘制等高线图
        x1vals = np.linspace(self.ranges[0], self.ranges[1], 100)
        x2vals = np.linspace(self.ranges[2], self.ranges[3], 100)
        # dimension: NUM_PTS x NUM_PTS
        x1, x2 = np.meshgrid(x1vals, x2vals, sparse = False, indexing = 'xy')
        data = np.vstack([x1.ravel(), x2.ravel()]).T
        observations, var = self.GP.predict_value(data)
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.set_xlim(self.ranges[0:2])
        ax1.set_ylim(self.ranges[2:])
        ax1.set_title('Countour Plot of the Robot\'s World Model')
        #机器人看到的世界
        plot = ax1.contourf(x1, x2, observations.reshape(x1.shape), cmap='viridis')
        #先验
        if self.GP.xvals is not None:
            scatter = ax1.scatter(self.GP.xvals[:, 0], self.GP.xvals[:, 1], c=self.GP.zvals.ravel(), s = 10.0, cmap = 'viridis')
        #熵最大位置
        if maxes is not None:
            print ("Plotting maxes" )
        if best_path is not None:
            f = np.array(best_path)
            print(f.size)
            # 当前最佳路径
            plt.plot(f[:, 0], f[:, 1], c='r', marker='*')
        if maxes is not None:
            # 最大熵点
            plt.scatter(maxes[:, 0], maxes[:, 1], color='r', marker='*', s=500.0)
        # Plot the samples taken by the robot,先验
        scatter = ax1.scatter(self.GP.xvals[:, 0], self.GP.xvals[:, 1], marker='*',s = 100.0)#, cmap = 'viridis'
        if screen:
            plt.show()

    def plot_information(self):
        ''' Visualizes the accumulation of reward and aquisition functions '''
        self.eval.plot_metrics()