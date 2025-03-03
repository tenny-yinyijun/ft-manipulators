import numpy as np
import matplotlib.pyplot as plt
import math

# Calculation
def dist(x, y):
  return np.sqrt(np.sum(np.square(x-y)))

# Plot
def plot_cluster(points, color, alpha):
  plt.scatter(points[:, 0], points[:, 1], color=color, alpha=alpha)
  
def plot_config(manipulator, cfg, lim=None):
    p = manipulator.get_joint_poses(cfg)
    base = np.array([0,0])
    # color_link = (0.1, 0.1, 1, 1)
    for i in range(0, manipulator.dof):
      color_link = (0.1, 0.1, 1, 1)
      joint = p[i]
      if not lim is None:
        if lim["joint_id"]==i:
          color_link = (1, 0.1, 0.1, 1)
      plt.plot([base[0], joint[0]], [base[1], joint[1]], marker = 'o', color=color_link)
      base = joint
      
def line_circle_intersection(xstart, ystart, xend, yend, a, b, r):
    # Check if either of the endpoints of the line segment are inside the circle
    if (xstart - a)**2 + (ystart - b)**2 <= r**2 or \
       (xend - a)**2 + (yend - b)**2 <= r**2:
        return True

    # Calculate the distance between the center of the circle and the line segment
    dx = xend - xstart
    dy = yend - ystart
    xcenter = xstart + dx/2
    ycenter = ystart + dy/2
    distance = math.sqrt((a-xcenter)**2 + (b-ycenter)**2)

    # Check if the distance is less than or equal to the radius of the circle
    if distance <= r:
        return True

    return False