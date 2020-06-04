import numpy as np

class Robot:
    def __init__(self, start_loc):
        self.start_loc = start_loc # Initial location of the robot
        self.delta = 0.30 # Sampling rate of the robot
        self.num_paths = 4 # Number of paths in the path set

    def path_set(self,start_pt,option):
        #判断异常，一般不考虑
        assert (option<self.num_paths), 'Option must be in range 0 - %d'%self.num_paths
        assert(option >= 0), 'Option must be in range 0 - %d'%self.num_paths
        x,y = start_pt
        if option == 0:
            locs = list([start_pt])
            for i in range(1,4):#3
                new_loc = [x+i*self.delta,y]
                locs.append(new_loc)
            return np.array(locs)
        elif option == 1:
            locs = list([start_pt])
            for i in range(1, 4):
                new_loc = [x - i * self.delta, y]
                locs.append(new_loc)
            return np.array(locs)
        elif option == 2:
            locs = list([start_pt])
            for i in range(1, 4):
                new_loc = [x, y + i * self.delta]
                locs.append(new_loc)
            return np.array(locs)
        elif option == 3:
            locs = list([start_pt])
            for i in range(1, 4):
                new_loc = [x, y - i * self.delta]
                locs.append(new_loc)
            return np.array(locs)

        # Generate data from a Gaussian mixture model
        def initializeGP(self, ranges, training_points, visualize=True):
            # Sample inputs and outputs 2D data
            if visualize:
                x = np.linspace(ranges[0], ranges[1], 100)
                y = np.linspace(ranges[2], ranges[3], 100)
                xvals, yvals = np.meshgrid(x, y, sparse=False, indexing='xy')
                zvals = rv(xvals, yvals)

            xtrain = np.linspace(ranges[0], ranges[1], training_points)
            ytrain = np.linspace(ranges[2], ranges[3], training_points)
            xtrain, ytrain = np.meshgrid(xtrain, ytrain, sparse=False, indexing='xy')
            data = np.vstack([xtrain.ravel(), ytrain.ravel()]).T
            ztrain = rv_sample(xtrain, ytrain)

            # Create and train parmeters of GP model
            self.GP = GPModel(data, np.reshape(ztrain, (1, data.shape[0])), lengthscale=10.0, variance=0.5)