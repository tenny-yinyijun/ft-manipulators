import math
import json 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
# from utils.calc import *

def dh_transform(a, alpha, d, theta):
    """
    Compute the Denavit-Hartenberg transformation matrix for given DH parameters.
    """
    T = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                  [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                  [0, np.sin(alpha), np.cos(alpha), d],
                  [0, 0, 0, 1]])
    return T

class LinkManipulator3D:
    def __init__(
        self,
        dh_params,
        grid_size=0.4,
        grid_num=10,
        step_size=0.4,
        ftws_step_size=0.4
    ):
        self.dh_params = dh_params
        self.link_lengths = dh_params[:, 2]
        self.dof = len(self.link_lengths)

        # constants
        self.grid_size = grid_size
        self.step_size = step_size
        self.ftws_step_size = ftws_step_size
        self.grid_num = grid_num
        self.base = np.array([0, 0, 0]) # base mount position
        
        self.obstacles = []
        self.maps = {}
        self.map_failure_cfgs = {}
        for j in range(self.dof):
            new_map, new_cfg = self.new_map()
            self.maps[j] = new_map
            self.map_failure_cfgs[j] = new_cfg

    def new_map(self):
        new_map = np.zeros((self.grid_num, self.grid_num, self.grid_num), dtype=int)
        new_cfg = np.ones((self.grid_num, self.grid_num, self.grid_num), dtype=int) * np.inf
        return new_map, new_cfg
    
    def pos_to_grid_coord(self, point):
        x, y, z = point[0], point[1], point[2]
        i = min(np.floor(x / self.grid_size) + (self.grid_num // 2), int(self.grid_num - 1)) # change
        j = min(np.floor(y / self.grid_size) + (self.grid_num // 2), int(self.grid_num - 1))
        k = min(np.floor(z / self.grid_size) + (self.grid_num // 2), int(self.grid_num - 1))
        return int(i), int(j), int(k)
    
    def update_map(self, point, bangle, fixed_jid):
        # find which bin in the map does this point fit into
        # same failure configuration is only counted once
        i, j, k = self.pos_to_grid_coord(point)
        
        if self.map_failure_cfgs[fixed_jid][i][j][k] != bangle:
            self.maps[fixed_jid][i][j][k] += 1
            self.map_failure_cfgs[fixed_jid][i][j][k] = bangle
        
    def generate_joint_configurations(self, jid, jangle):
        joint_configurations = []

        for i in range(self.dof):
            if i == jid:
                joint_values = np.array([jangle])
            else:
                lower_limit = -np.pi
                upper_limit = np.pi
                joint_values = np.arange(
                    lower_limit, upper_limit + self.ftws_step_size, self.ftws_step_size
                )
            joint_configurations.append(joint_values)

        joint_configurations_meshgrid = np.meshgrid(*joint_configurations)
        all_joint_configurations = np.vstack(
            [arr.flatten() for arr in joint_configurations_meshgrid]
        ).T
        return all_joint_configurations
    
    def get_reachability_map(self):
        # first, iterate through all possible configurations
        maxval = 0
        for jid in range(self.dof): # for each joint
            print("joint id =", jid)
            for angle in np.arange(-np.pi, np.pi, self.step_size): # for each angle
                # print("angle =", angle)
                maxval += 1
                all_cfg = self.generate_joint_configurations(jid, angle)
                for cfg in all_cfg:
                    p, has_collision = self.get_eff_pose(cfg)
                    if not has_collision:
                        self.update_map(p, angle, jid)
        # then, compute the reachability map
        reachability_map = np.zeros((self.grid_num, self.grid_num, self.grid_num))
        for i in range(self.grid_num):
            for j in range(self.grid_num):
                for k in range(self.grid_num):
                    opacity = 0
                    for d in range(self.dof):
                        opacity += self.maps[d][i][j][k] #/ max_value
                    reachability_map[i][j][k] = opacity / maxval
        return reachability_map
    
    # check collision
    def check_collision(self, start, end):
        pass
    
    def get_joint_poses(self, cfg):
        # config cfg = [theta_1, theta_2, ...]
        assert len(cfg) == self.dof
        result = []
        T = np.eye(4)
        base = self.base

        link_ends = []
        # Plot links
        for i in range(self.dof):
            # a, alpha, d, _ = self.dh_params[i]
            _, d, a, alpha = self.dh_params[i]
            theta = cfg[i]
            T = np.dot(T, dh_transform(a, alpha, d, theta))
            link_ends.append(T[:3, 3])
        return link_ends, False

    def get_eff_pose(self, cfg):
        joint_poses, has_collision = self.get_joint_poses(cfg)
        if has_collision:
            return None, has_collision
        return joint_poses[-1], has_collision