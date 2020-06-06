import dubins
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#class：
#Path_Generator：该class完成一个朴素的点到点的连接
#Dubins_Path_Generator：调用dubins生成一个点到点的路径
#Dubins_EqualPath_Generator：控制沿着dubins轨迹的样本点的数量，使所有轨迹具有相同数量的可用样本。

class Path_Generator:
    '''The Path_Generator class which creates naive point-to-point straightline paths'''

    def __init__(self, frontier_size, horizon_length, turning_radius, sample_step, extent):
        '''
        frontier_size (int)：frontier上我们需要考虑的点的数量，要生成的路径数
        horizon_length (float)：地平线与车辆之间的距离
        turning_radius (float)：转弯半径
        sample_step (float) ：采样频率
        '''
        self.fs = frontier_size
        self.hl = horizon_length
        self.tr = turning_radius
        self.ss = sample_step
        self.extent = extent

        # Global variables
        self.goals = []  # The frontier coordinates
        self.samples = {}  # The sample points which form the paths
        self.cp = (0, 0, 0)  # The current pose of the vehicle

    def generate_frontier_points(self):
        '''From the frontier_size and horizon_length, generate the frontier points to goal'''
        angle = np.linspace(-2.35,2.35,self.fs)#self.fs
        #圆形的积分，-2.35为yaw角度值，hl为圆心到圆边的距离
        self.goals = [(self.hl * np.cos(self.cp[2] + a) + self.cp[0], self.hl * np.sin(self.cp[2] + a) + self.cp[1],
                       self.cp[2] + a) for a in angle]
        '''
        goals = self.goals
        goals = np.array(goals).reshape(20, 3)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(0, 0, 0, 'r*')
        ax.scatter(goals[:, 0], goals[:,1], goals[:,2],'ro')
        plt.show()
        
        goals = self.goals
        goals = np.array(goals).reshape(20, 3)
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(111)
        ax.set_title('Countour Plot of the Robot\'s World Model')
        # Plot the samples taken by the robot
        scatter = ax.scatter(goals[:, 0], goals[:,1], s=10.0, cmap='viridis')
        plt.show()'''

        return self.goals

    def make_sample_paths(self):
        '''Connect the current_pose to the goal places'''
        cp = np.array(self.cp)
        coords = {}
        for i, goal in enumerate(self.goals):
            g = np.array(goal)
            distance = np.sqrt((cp[0] - g[0]) ** 2 + (cp[1] - g[1]) ** 2)
            samples = int(round(distance / self.ss))

            for j in range(0, samples):
                x = cp[0] + (i * self.ss) * np.cos(g[2])
                y = cp[1] + (i * self.ss) * np.sin(g[2])
                a = g[2]
                try:
                    coords[i].append((x, y))
                except:
                    coords[i] = []
                    coords[i].append((x, y,a))
        self.samples = coords
        return coords

    def get_path_set(self, current_pose):
        '''Primary interface for getting list of path sample points for evaluation'''
        self.cp = current_pose
        #生成20个观测边界点
        self.generate_frontier_points()
        #将路径离散化采样并判断边界
        paths = self.make_sample_paths()
        return paths

    def get_frontier_points(self):
        return self.goals

    def get_sample_points(self):
        return self.samples


class Dubins_Path_Generator(Path_Generator):
    '''
    The Dubins_Path_Generator class, which inherits from the Path_Generator class. Replaces the make_sample_paths
    method with paths generated using the dubins library
    '''

    def make_sample_paths(self):
        '''Connect the current_pose to the goal places'''
        coords = {}
        for i, goal in enumerate(self.goals):
            g = (goal[0], goal[1], self.cp[2])
            path = dubins.shortest_path(self.cp, goal, self.tr)
            configurations, _ = path.sample_many(self.ss)
            coords[i] = [config for config in configurations if
                         config[0] > self.extent[0] and config[0] < self.extent[1] and config[1] > self.extent[2] and
                         config[1] < self.extent[3]]

        self.samples = coords
        return coords


class Dubins_EqualPath_Generator(Path_Generator):
    '''
    The Dubins_EqualPath_Generator class which inherits from Path_Generator. Modifies Dubin Curve paths so that all
    options have an equal number of sampling points
    '''

    def make_sample_paths(self):
        '''Connect the current_pose to the goal places'''
        coords = {}
        for i, goal in enumerate(self.goals):
            g = (goal[0], goal[1], self.cp[2])
            path = dubins.shortest_path(self.cp, goal, self.tr)
            configurations, _ = path.sample_many(self.ss)
            coords[i] = [config for config in configurations if
                         config[0] > self.extent[0] and config[0] < self.extent[1] and config[1] > self.extent[2] and
                         config[1] < self.extent[3]]

        # find the "shortest" path in sample space
        current_min = 1000
        for key, path in coords.items():
            if len(path) < current_min and len(path) > 1:
                current_min = len(path)

        # limit all paths to the shortest path in sample space
        for key, path in coords.items():
            if len(path) > current_min:
                path = path[0:current_min]
                coords[key] = path

        self.samples = coords
        return coords