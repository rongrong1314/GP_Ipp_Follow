import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

class Evaluation:
    def __init__(self, world,reward_function='mean'):
        self.world = world

        self.metrics = {'aquisition_function': {},
                        'mean_reward': {},
                        'info_gain_reward': {},
                        'hotspot_info_reward': {},
                        'MSE': {},
                        'instant_regret': {},
                        'regret_bound': {}
                       }
        # 几种评估方法
        self.reward_function = reward_function
        if  reward_function == 'mes':
            self.f_rew = self.info_gain_reward
            self.f_aqu = info_gain
        else:
            raise ValueError('Only \'mean\' and \'hotspot_info\' reward functions currently supported.')

    '''Reward Functions 
            通常包含三个参数：
                * time (int): 当前的时间
                * xvals (list of float tuples): 一条路径的系列点
                * robot_model (GPModel)'''

    def update_metrics(self, t, robot_model, all_paths, selected_path):
        ''' Function to update avaliable metrics'''
        # Compute aquisition function
        #更新info_gain信息增益，求总熵
        self.metrics['aquisition_function'][t] = self.f_aqu(t, selected_path, robot_model)
        # Compute reward functions
        #路径观测值（real world）
        self.metrics['mean_reward'][t] = self.mean_reward(t, selected_path, robot_model)
        #和f_aqu一样
        self.metrics['info_gain_reward'][t] = self.info_gain_reward(t, selected_path, robot_model)
        #熵+均值和
        self.metrics['hotspot_info_reward'][t] = self.hotspot_info_reward(t, selected_path, robot_model)
        # 计算实际值和观测值的均方差
        self.metrics['MSE'][t] = self.MSE(robot_model, NTEST=25)
        #选择路径熵与所有路径最大熵的差
        self.metrics['instant_regret'][t] = self.inst_regret(t, all_paths, selected_path, robot_model)

    def mean_reward(self, time, xvals, robot_model):
        #input：选择的路径点
        #output：真实世界的观测值
        ''' 预测均值（真实）奖励函数'''
        data = np.array(xvals)
        x1 = data[:, 0]
        x2 = data[:, 1]
        queries = np.vstack([x1, x2]).T
        #real world 观测均值
        mu, var = self.world.GP.predict_value(queries)
        return np.sum(mu)

    def info_gain_reward(self, time, xvals, robot_model):
        ''' 收集到的信息奖励 '''
        return info_gain(time, xvals, robot_model)

    def hotspot_info_reward(self, time, xvals, robot_model):
        ''' 奖励信息与exploitation值的和'''
        LAMBDA = 1.0  # TOOD: should depend on time
        data = np.array(xvals)
        x1 = data[:, 0]
        x2 = data[:, 1]
        queries = np.vstack([x1, x2]).T

        mu, var = self.world.GP.predict_value(queries)
        #info_gain+lamda*mu
        return self.info_gain_reward(time, xvals, robot_model) + LAMBDA * np.sum(mu)

    def MSE(self, robot_model, NTEST=10):
        ''' 计算一系列测试点的MSE'''
        np.random.seed(0)
        #在地图上随机采样
        x1 = np.random.random_sample((NTEST, 1)) * (self.world.x1max - self.world.x1min) + self.world.x1min
        x2 = np.random.random_sample((NTEST, 1)) * (self.world.x2max - self.world.x2min) + self.world.x2min
        data = np.hstack((x1, x2))
        #采样点的观测值
        pred_world, var_world = self.world.GP.predict_value(data)
        #机器人现有观测环境的预测
        pred_robot, var_robot = robot_model.predict_value(data)
        return ((pred_world - pred_robot) ** 2).mean()

    def inst_regret(self, t, all_paths, selected_path, robot_model):
        ''' 根据指定的奖励函数，确定选择路径的instantaneous Kapoor regret
        Input:
        * all_paths: the set of all avalaible paths to the robot at time t
        * selected path: the path selected by the robot at time t '''

        value_omni = {}
        for path, points in all_paths.items():
            #计算每条路径的增益
            value_omni[path] = self.f_rew(time=t, xvals=points, robot_model=robot_model)
        #得到最大增益值
        value_max = value_omni[max(value_omni, key=value_omni.get)]
        #计算选择的路径的增益值
        value_selected = self.f_rew(time=t, xvals=selected_path, robot_model=robot_model)

        # assert(value_max - value_selected >= 0)
        return value_max - value_selected

    def plot_metrics(self):
        ''' Plots the performance metrics computed over the course of a info'''
        # Asumme that all metrics have the same time as MSE; not necessary
        time = np.array(self.metrics['MSE'].keys())

        ''' Metrics that require a ground truth global model to compute'''
        MSE = np.array(self.metrics['MSE'].values())
        regret = np.cumsum(np.array(self.metrics['instant_regret'].values()))
        mean = np.cumsum(np.array(self.metrics['mean_reward'].values()))
        hotspot_info = np.cumsum(np.array(self.metrics['hotspot_info_reward'].values()))

        ''' Metrics that the robot can compute online '''
        info_gain = np.cumsum(np.array(self.metrics['info_gain_reward'].values()))
        UCB = np.cumsum(np.array(self.metrics['aquisition_function'].values()))

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Accumulated UCB Aquisition Function')
        plt.plot(time, UCB, 'g')

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Accumulated Information Gain')
        plt.plot(time, info_gain, 'k')

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Accumulated Mean Reward')
        plt.plot(time, mean, 'b')

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Accumulated Hotspot Information Gain Reward')
        plt.plot(time, hotspot_info, 'r')

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Average Regret w.r.t. ' + self.reward_function + ' Reward')
        plt.plot(time, regret / time, 'b')

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Map MSE at 100 Random Test Points')
        plt.plot(time, MSE, 'r')

        plt.show()



'''Aquisition Functions'''
def info_gain(time, xvals, robot_model):
    #input：选择的路径点，机器人模型
    #计算一组潜在样本位置相对于底层功能条件或先前样本xobs的信息增益
    data = np.array(xvals)
    x1 = data[:, 0]
    x2 = data[:, 1]
    queries = np.vstack([x1, x2]).T
    xobs = robot_model.xvals
    #如果机器人还没有进行任何观察，只需返回势集的熵
    if xobs is None:
        Sigma_after = robot_model.kern.K(queries)
        #np.linalg.slogdet求行列式的符号和自然对数ln
        entropy_after, sign_after = np.linalg.slogdet(np.eye(Sigma_after.shape[0], Sigma_after.shape[1]) \
                                                      + robot_model.variance * Sigma_after)
        # print "Entropy with no obs: ", entropy_after
        return 0.5 * sign_after * entropy_after
    #将先验点与观测路径点合并
    all_data = np.vstack([xobs, queries])

    # The covariance matrices of the previous observations and combined observations respectively
    #先验核
    Sigma_before = robot_model.kern.K(xobs)
    #交叉核
    Sigma_total = robot_model.kern.K(all_data)

    # The term H(y_a, y_obs)
    entropy_before, sign_before = np.linalg.slogdet(np.eye(Sigma_before.shape[0], Sigma_before.shape[1]) \
                                                    + robot_model.variance * Sigma_before)

    # The term H(y_a, y_obs)
    entropy_after, sign_after = np.linalg.slogdet(np.eye(Sigma_total.shape[0], Sigma_total.shape[1]) \
                                                  + robot_model.variance * Sigma_total)

    # The term H(y_a | f)
    entropy_total = 2 * np.pi * np.e * sign_after * entropy_after - 2 * np.pi * np.e * sign_before * entropy_before
    ''' TODO: this term seems like it should still be in the equation, but it makes the IG negative'''
    # entropy_const = 0.5 * np.log(2 * np.pi * np.e * robot_model.variance)
    entropy_const = 0.0
    return entropy_total - entropy_const


def sample_max_vals(robot_model):
    ''' 潜在样本和局部极大值间的互信息（mutual information）'''
    # If the robot has not samples yet, return a constant value
    #初始状态返回none，先验
    if robot_model.xvals is None:
        return None, None

    nK = 1  # The number of samples maximum values
    nFeatures = 200  # number of random features samples to approximate GP
    d = robot_model.xvals.shape[1]  # The dimension of the points (should be 2D)

    ''' Sample Maximum values i.e. return sampled max values for the posterior GP, conditioned on 
    current observations. Construct random freatures and optimize functions drawn from posterior GP.'''
    samples = np.zeros((nK, 1))
    locs = np.zeros((nK, 2))

    for i in range(nK):
        # Draw the weights for the random features，生成200个0,1正态分布随机数
        W = np.random.normal(loc=0.0, scale=robot_model.lengthscale, size=(nFeatures, d))
        #生成一个均匀分布随机数
        b = 2 * np.pi * np.random.uniform(low=0.0, high=1.0, size=(nFeatures, 1))

        # Compute the features for xx
        Z = np.sqrt(2 * robot_model.variance / nFeatures) * np.cos(np.dot(W, robot_model.xvals.transpose()) + b)
        # 噪声
        noise = np.random.normal(loc=0.0, scale=1.0, size=(nFeatures, 1))
        #Figure this code out
        if robot_model.xvals.shape[0] < nFeatures and False:
            Sigma = np.dot(Z.transpose(), Z) + robot_model.noise * np.eye(robot_model.xvals.shape[0])
            mu = np.dot(np.dot(Z, np.linalg.inv(Sigma)), robot_model.zvals)
            [U, D] = np.linalg.eig(Sigma)
            D = np.diag(D)
            R = (np.sqrt(D))
        else:
            #σ^-2=1/noise
            Sigma = np.dot(Z, Z.transpose()) / robot_model.noise + np.eye(nFeatures)
            #Σt
            Sigma = np.linalg.inv(Sigma)
            # σ^−2*Σt*Z*y
            mu = np.dot(np.dot(Sigma, Z), robot_model.zvals) / robot_model.noise
            #
            theta = mu + np.dot(np.linalg.cholesky(Sigma), noise)

        # Obtain a function samples from posterior GP
        #输入x，输出冒号后面的值,nFeatures维特征映射，定义了一个function，输入x
        target = lambda x: np.dot(theta.T * np.sqrt(2.0 * robot_model.variance / nFeatures), \
                                  np.cos(np.dot(W, x.T) + b)).transpose()
        #target第一个数取反
        target_vector_n = lambda x: -float(target(x)[0, 0])

        # Can only take a 1D input，求导
        target_gradient = lambda x: np.dot(theta.T * -np.sqrt(2.0 * robot_model.variance / nFeatures), \
                                           np.sin(np.dot(W, x.reshape((2, 1))) + b) * W)
        target_vector_gradient_n = lambda x: -np.asarray(target_gradient(x).reshape(2, ))

        # Optimize the function，优化函数，获取当前map的全局最优解
        maxima, max_val = global_maximization(target, target_vector_n, target_gradient, target_vector_gradient_n,
                                              robot_model.ranges, robot_model.xvals)
        samples[i] = max_val
        locs[i, :] = maxima

        if max_val < np.max(robot_model.zvals) + 5.0 * np.sqrt(robot_model.noise):
            samples[i] = np.max(robot_model.zvals) + 5.0 * np.sqrt(robot_model.noise)
            # locs[i, :] = locs[i-1, :]

        return samples, locs


def global_maximization(target, target_vector_n, target_grad, target_vector_gradient_n, ranges, guesses=[]):
    ''' Perform efficient global maximization'''
    gridSize = 10

    # Uniformly sample gridSize number of points in interval xmin to xmax
    x1 = np.random.uniform(ranges[0], ranges[1], size=gridSize)
    x2 = np.random.uniform(ranges[2], ranges[3], size=gridSize)
    x1, x2 = np.meshgrid(x1, x2, sparse=False, indexing='xy')

    Xgrid = np.vstack([x1.ravel(), x2.ravel()]).T
    Xgrid = np.vstack([Xgrid, guesses])

    # Get the function value at Xgrid locations
    y = target(Xgrid)
    max_index = np.argmax(y)
    start = np.asarray(Xgrid[max_index, :])
    print ("Starting optimization at", start)
    #fun优化的函数，x0最初的猜测，用采样的最大值作为初始猜测，然后计算全局最大值
    res = sp.optimize.minimize(fun=target_vector_n, x0=start, method='SLSQP', jac=target_vector_gradient_n, \
                               bounds=((ranges[0], ranges[1]), (ranges[2], ranges[3])))
    #可视化
    '''
    # Generate a set of observations from robot model with which to make contour plots
    x1vals = np.linspace(ranges[0], ranges[1], 100)
    x2vals = np.linspace(ranges[2], ranges[3], 100)
    x1, x2 = np.meshgrid(x1vals, x2vals, sparse=False, indexing='xy')  # dimension: NUM_PTS x NUM_PTS
    data = np.vstack([x1.ravel(), x2.ravel()]).T
    observations = target(data)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    # ax2 = fig2.add_subplot(111)
    ax2.set_xlim(ranges[0:2])
    ax2.set_ylim(ranges[2:])
    ax2.set_title('Countour Plot of the Approximated World Model')
    plot = ax2.contourf(x1, x2, observations.reshape(x1.shape), cmap='viridis')
    scatter = ax2.scatter(res['x'][0], res['x'][1], color='r', s=100.0)
    plt.show()
    '''

    return res['x'], res['fun']



def mves(time, xvals, robot_model, maxes):
    '''定义MES获取函数和梯度'''
    #给定样本函数最大值和之前的函数最大值集，在查询点x处利用MES计算获取函数值f和梯度g
    #input：xvals路径点
    #初始处理
    data = np.array(xvals)
    x1 = data[:, 0]
    x2 = data[:, 1]
    queries = np.vstack([x1, x2]).T
    nK = 1  # The number of samples maximum values
    nFeatures = 200  # number of random features samples to approximate GP
    d = queries.shape[1]  # The dimension of the points (should be 2D)

    # Initialize f, g
    f = 0
    for i in range(nK):
        # 计算后验均值/方差预测和梯度
        mean, var = robot_model.predict_value(queries)
        std_dev = np.sqrt(var)
        # 计算MES的acquisition function
        #(y-µ)/σ
        gamma = (maxes[i] - mean) / var
        #正态分布概率密度函数
        pdfgamma = sp.stats.norm.pdf(gamma)
        #累计概率密度函数
        cdfgamma = sp.stats.norm.cdf(gamma)
        # (γ*ψ)/(2*Ψ)-log(Ψ)
        f += sum(gamma * pdfgamma / (2.0 * cdfgamma) - np.log(cdfgamma))
        # Average f
    f = f / nK
    return f

def mean_UCB(time, xvals, robot_model):
    ''' Computes the UCB for a set of points along a trajectory '''
    # The GPy interface can predict mean and variance at an array of points
    # this will be a strict overestimate of the value of the points
    #用的是robot model
    data = np.array(xvals)
    x1 = data[:,0]
    x2 = data[:,1]
    queries = np.vstack([x1, x2]).T
    mu, var = robot_model.predict_value(queries)
    delta = 0.9
    d = 20
    pit = np.pi ** 2 * (time + 1) ** 2 / 6.
    beta_t = 2 * np.log(d * pit / delta)

    return np.sum(mu) + np.sqrt(beta_t) * np.sum(np.fabs(var))