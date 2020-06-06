import numpy as np
from GPModel import *
from IPython import display
from Evaluation import *
from path_generator import *
from matplotlib import cm

class Robot:
    '''Robot class包括车辆当前模型,路径设置，信息路径规划算法
       sample_world (method)：以一组位置作为输入并返回一组观察值的函数句柄
       start_loc (tuple of floats)： 机器人起始位置
       kernel_file (string)： a filename specifying the location of the stored kernel values
       kernel_dataset (tuple of nparrays) ：a tuple (xvals, zvals), where xvals is a Npoint x 2 nparray
                                          of type float and zvals is a Npoint x 1 nparray of type float
       prior_dataset (tuple of nparrays)： (xvals, zvals), xvals为Npoint x 2，zvals为Npoint x 1
       init_variance (float)：平方指数核的方差参数
       init_lengthscale (float) ：平方指数核的长度参数
       noise (float)：传感器噪声参数的平方指数核  '''

    def __init__(self, sample_world, start_loc,ranges, kernel_file = None,kernel_dataset = None,
            prior_dataset = None, init_lengthscale = 10.0, init_variance = 100.0, noise = 0.05,
            path_generator = 'default', frontier_size = 6, horizon_length = 5, turning_radius = 1,
            sample_step = 0.5,evaluation = None):
        ''' Initialize the robot class with a GP model, initial location, path sets, and prior dataset'''
        self.ranges = ranges
        self.eval = evaluation
        self.loc = start_loc  # Initial location of the robot
        self.sample_world = sample_world  # A function handel that allows the robot to sample from the environment
        self.aquisition_function = UCB
        self.total_value = {}

        # Initialize the robot's GP model with the initial kernel parameters
        self.GP = GPModel(lengthscale=init_lengthscale, variance=init_variance)

        # 如果同时提供了内核训练数据集和先验数据集，那么使用两者训练内核，并将先验数据集纳入模型
        if kernel_dataset is not None and prior_dataset is not None:
            data = np.vstack([prior_dataset[0], kernel_dataset[0]])
            observations = np.vstack([prior_dataset[1], kernel_dataset[1]])
            self.GP.train_kernel(data, observations)
            # Train the kernel using the provided kernel dataset
        elif kernel_dataset is not None:
            self.GP.train_kernel(kernel_dataset[0], kernel_dataset[1])
        # If a kernel file is provided, load the kernel parameters
        elif kernel_file is not None:
            self.GP.load_kernel()
        # No kernel information was provided, so the kernel will be initialized with provided values
        else:
            pass

        # Incorporate the prior dataset into the model
        if prior_dataset is not None:
            self.GP.set_data(prior_dataset[0], prior_dataset[1])

            # The path generation class for the robot
        path_options = {
            'default': Path_Generator(frontier_size, horizon_length, turning_radius, sample_step, ranges),
            'dubins': Dubins_Path_Generator(frontier_size, horizon_length, turning_radius, sample_step, ranges),
            'equal_dubins': Dubins_EqualPath_Generator(frontier_size, horizon_length, turning_radius, sample_step,
                                                           ranges)}
        self.path_generator = path_options[path_generator]

    def myopic_planner(self, T):
        ''' Gather noisy samples of the environment and updates the robot's GP model
        Input:
        * T (int > 0): the length of the planning horization (number of planning iterations)'''
        self.trajectory = []

        for t in range(T):
            # Select the best trajectory according to the robot's aquisition function
            #用UCB对路径进行评估，并选择最好的路径作为best_path
            #选择当前点下20个方向中最优的一个点方向路径
            best_path, best_val = self.choose_trajectory(T=T, t=t)

            if best_path == None:
                break
            data = np.array(best_path)
            x1 = data[:, 0]
            x2 = data[:, 1]
            xlocs = np.vstack([x1, x2]).T
            #一顿更新，返回世界坐标的均值,起点到终点的6*2路径点
            self.collect_observations(xlocs)
            #返回均方差，robot观察与world之间，存入m_value
            self.eval.m_value[t] = self.eval.MSE(self.GP)
            #将best_val存入a_value
            self.eval.a_value[t] = best_val
            #存储路径
            self.trajectory.append(best_path)

            if len(best_path) == 1:
                self.loc = (best_path[-1][0], best_path[-1][1], best_path[-1][2] - 0.45)
            else:
                self.loc = best_path[-1]

    def choose_trajectory(self, T, t):
        '''根据aquisition function，选取当前姿态最佳轨迹
        Input:
        * T (int > 0):规划迭代次数
        * t (int > 0):当前的迭代时间'''
        # 当前点对应的20个点相应的路径点
        paths = self.path_generator.get_path_set(self.loc)
        value = {}
        for path, points in paths.items():
            cmean = 0
            cvar = 0
            data = np.array(points)
            x = data[:, 0]
            y = data[:, 1]
            queries = np.vstack([x, y]).T
            #用UCB方法评估路径，在explore和exploit之间平衡
            value[path] = self.aquisition_function(timestep=t, query_points=queries, model=self.GP)
        try:
            # self.total_value[t] = value[max(value, key = value.get)]
            return paths[max(value, key=value.get)], value[max(value, key=value.get)]
        except:
            return None

    def collect_observations(self, xobs):
        ''' Gather noisy samples of the environment and updates the robot's GP model.
        Input:
        * xobs (float array):观测位置，NUM_PTS x 2'''
        #world.sample_value
        #返回当前点的世界坐标预测均值
        zobs = self.sample_world(xobs)
        #将新的最优路径坐标加入model
        self.GP.add_data(xobs, zobs)

    def plot_information(self):
        ''' Visualizes the accumulation of reward and aquisition functions '''
        last_value = 0
        t = []
        v = []
        for time, value in self.eval.a_value.items():
            t.append(time)
            v.append(value + last_value)
            last_value += value

        fig, ax = plt.subplots(figsize=(4, 3))
        plt.plot(t, v, 'b')

        m = []
        for time, value in self.eval.m_value.items():
            m.append(value)

        fig, ax = plt.subplots(figsize=(4, 3))
        plt.plot(t, m, 'r')
        plt.show()
    def visualize_world_model(self):
        #通过在空间上均匀采样点并绘制预测值来实现机器人当前世界模型的可视化
        #从机器人模型中生成一组观测值，用于绘制等高线图
        x1vals = np.linspace(self.ranges[0], self.ranges[1], 100)
        x2vals = np.linspace(self.ranges[2], self.ranges[3], 100)
        # dimension: NUM_PTS x NUM_PTS
        x1, x2 = np.meshgrid(x1vals, x2vals, sparse = False, indexing = 'xy')
        data = np.vstack([x1.ravel(), x2.ravel()]).T
        observations, var = self.GP.predict_value(data)

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.set_xlim(self.ranges[0:2])
        ax2.set_ylim(self.ranges[2:])
        ax2.set_title('Countour Plot of the Robot\'s World Model')
        #可视化机器人看到的世界模型
        plot = ax2.contourf(x1, x2, observations.reshape(x1.shape), cmap='viridis')

        # Plot the samples taken by the robot
        scatter = ax2.scatter(self.GP.xvals[:, 0], self.GP.xvals[:, 1], s = 10.0, cmap = 'viridis')
        plt.show()

    def visualize_trajectory(self):
        ''' Visualize the set of paths chosen by the robot '''
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_xlim(self.ranges[0:2])
        ax.set_ylim(self.ranges[2:])

        color = iter(plt.cm.cool(np.linspace(0, 1, len(self.trajectory))))

        for i, path in enumerate(self.trajectory):
            c = next(color)
            f = np.array(path)
            plt.plot(f[:, 0], f[:, 1], c=c, marker='*')
        plt.show()



    def gatherSamples(self):
        ''' Some example code to gather observations at a set of locations and plot those observations
            Can be deleted'''
        ranges = (-10, 10, -10, 10)
        x1observe = np.linspace(ranges[0], ranges[1], 10)
        x2observe = np.linspace(ranges[2], ranges[3], 10)
        x1observe, x2observe = np.meshgrid(x1observe, x2observe, sparse=False, indexing='xy')
        data = np.vstack([x1observe.ravel(), x2observe.ravel()]).T
        observations = self.sample_world(data)
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        surf = ax1.plot_surface(x1observe, x2observe, observations.reshape(x1observe.shape), cmap=cm.coolwarm,
                                linewidth=0)
        plt.show()