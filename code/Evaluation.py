import numpy as np


class Evaluation:
    def __init__(self, world):
        self.world = world

        self.a_value = {}
        self.r_value = {}
        self.regret_values = {}
        self.m_value = {}

    def simple_regret(self, path_sets):
        pass

    def MSE(self, robot_model, NTEST=10):
        ''' Compute the MSE on a set of test points, randomly distributed throughout the environment'''
        np.random.seed(0)
        x1 = np.random.random_sample((NTEST, 1)) * (self.world.x1max - self.world.x1min) + self.world.x1min
        x2 = np.random.random_sample((NTEST, 1)) * (self.world.x2max - self.world.x2min) + self.world.x2min
        data = np.hstack((x1, x2))

        pred_world, var_world = self.world.GP.predict_value(data)
        pred_robot, var_robot = robot_model.predict_value(data)

        return ((pred_world - pred_robot) ** 2).mean()


def UCB(timestep, query_points, model):
    ''' Computes the UCB for a set of points along a trajectory '''
    # The GPy interface can predict mean and variance at an array of points
    # this will be a strict overestimate of the value of the points
    mu, var = model.predict_value(query_points)

    delta = 0.9
    d = 20
    pit = np.pi ** 2 * (timestep + 1) ** 2 / 6.
    beta_t = 2 * np.log(d * pit / delta)

    value = np.sum(mu) + np.sqrt(beta_t) * np.sum(np.fabs(var))
    return value