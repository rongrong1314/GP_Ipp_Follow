import dubins
import numpy as np
import matplotlib.pyplot as plt
from path_generator import*
'''
# Generating the shortest path
#start (x,y,yaw)
q0 = (0, 0, 0)
#goal (x,y,yaw)
q1 = (10, 10, 3.14)

#radius of feasible turn
turning_radius = 1.0
#sampling "rate" or distance
step_size = 0.5

#find the shortest path, returns a dubins object
path = dubins.shortest_path(q0, q1, turning_radius)
#sample the path and return the points along the path
#methods to apply on the path include sample_many and sample
configurations, _ = path.sample_many(step_size)
#plot the path
x = []
y = []
for sample_points in configurations:
    x.append(sample_points[0])
    y.append(sample_points[1])
#plt.plot(x,y,'r*')

words = [0,1,2,3,4,5]
#start
q0 = (0, 0, 0)
#goal
q1 = (10, 10, -1.57)
#radius of feasible turn
turning_radius = 2.0
#sampling "rate" or distance
step_size = 1.5
fig,ax = plt.subplots(1,5)
fig.set_size_inches(10,3)
# generate a figure for each of the options
# 打包为元组的列表
for a, word in zip(ax, words):
    path = dubins.path(q0, q1, turning_radius, word)
    # Note! If a path is not feasible, path will be a Nonetype object, which will need to be handled
    if path != None:
        configurations, _ = path.sample_many(step_size)
        configurations = np.array(configurations)
        a.axis('equal')
        a.plot(configurations[:, 0], configurations[:, 1], 'b-')
        a.plot(configurations[:, 0], configurations[:, 1], 'r*')
        plt.show()'''
cp = (0, 0, 0)
h1 = 5.0
angle = np.linspace(-2.35, 2.35, 10000)  # self.fs
goals = [(h1 * np.cos(cp[2] + a) + cp[0], h1 * np.sin(cp[2] + a) + cp[1],
               cp[2] + a) for a in angle]
goals = np.array(goals).reshape(10000, 3)
fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(111)
ax.set_title('Countour Plot of the Robot\'s World Model')
scatter = ax.scatter(goals[:, 0], goals[:, 1], s=10.0, cmap='viridis')
plt.show()
