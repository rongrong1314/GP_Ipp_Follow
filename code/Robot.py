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