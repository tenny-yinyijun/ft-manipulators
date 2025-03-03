import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils import *

class LinkManipulator2D:
    def __init__(
        self,
        link_lengths,
        joint_limits=None,
    ):
        self.dof = len(link_lengths)
        if joint_limits is None:
            joint_limits = np.array([[-np.pi, np.pi]] * self.dof)
        assert len(link_lengths) == len(joint_limits)
        self.link_lengths = link_lengths
        self.joint_limits = joint_limits

        # constants
        self.resolution = 1 / 10
        self.step_size = 0.2
        
        self.obstacles = []
        self.maps = {}
        self.map_failure_cfgs = {}
        for j in range(self.dof):
            new_map, new_cfg = self.new_map(self.resolution)
            self.maps[j] = new_map
            self.map_failure_cfgs[j] = new_cfg
    
    def add_obstacle(self, center, radius):
        self.obstacles.append([center, radius])
        
    def add_obstacle_list(self, obslist):
        for obs in obslist:
            self.obstacles.append(obs)
        
    def fix_joint(self, jid, jangle):
        self.fixed_jid = jid
        self.fixed_jangle = jangle

    def new_map(self, resolution):
        # returns the map and cfg (to keep track such that the same grid will
        # not be counted twice) of the specified resolution
        self.grid_size = resolution
        self.max_length = 4.0 # maximum ws side length
        self.grid_num = int(np.ceil(self.max_length / resolution))
        new_map = np.zeros((self.grid_num, self.grid_num), dtype=int)
        new_cfg = np.ones((self.grid_num, self.grid_num), dtype=int) * np.inf

        return new_map, new_cfg

    def pos_to_grid_coord(self, point):
        x, y = point[0], point[1]
        i = min(np.floor(x / self.grid_size) + (self.grid_num // 2), 39) # TODO hard code
        j = min(np.floor(y / self.grid_size) + (self.grid_num // 2), 39)
        return int(i), int(j)

    def grid_coord_to_pose(self, i, j):
        # returns lower-left corner
        i = i - (self.grid_num // 2) #+ 1
        j = j - (self.grid_num // 2) #+ 1
        x = i * self.grid_size  # - self.max_length/2
        y = j * self.grid_size  # - self.max_length/2
        return x, y

    def update_map(self, point, bangle):
        # find which bin in the map does this point fit into
        # same failure configuration is only counted once
        i, j = self.pos_to_grid_coord(point)
        
        if self.map_failure_cfgs[self.fixed_jid][i][j] != bangle:
            self.maps[self.fixed_jid][i][j] += 1
            self.map_failure_cfgs[self.fixed_jid][i][j] = bangle
            
    def set_plt(self, visualize_obstacles):
        plt.gca().clear()
        plt.gca().set_aspect("equal")
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        if visualize_obstacles:
            self.plot_obstacles()
    
    def plot_obstacles(self):
        for obs in self.obstacles:
            [x, y] = obs[0]
            r = obs[1]
            circle = patches.Circle((x, y), r, color=(1,0,0,0.2))
            plt.gca().add_patch(circle)
        
    def save_heatmap(self, file_name, max_value, visualize_obstacles=True):
        self.set_plt(visualize_obstacles)

        target_map = self.maps[self.fixed_jid]
        
        for i in range(self.grid_num):
            for j in range(self.grid_num):
                opacity = target_map[i][j] / max_value
                x, y = self.grid_coord_to_pose(i, j)
                rect = patches.Rectangle(
                    (x, y),
                    self.grid_size,
                    self.grid_size,
                    linewidth=1,
                    facecolor=(0, 0.5, 0, opacity),
                )
                plt.gca().add_patch(rect)
        plt.savefig(file_name)
        
    def save_binarymap(self, file_name, max_value, visualize_obstacles=True):
        self.set_plt(visualize_obstacles)

        target_map = self.maps[self.fixed_jid]
        for i in range(self.grid_num):
            for j in range(self.grid_num):
                if target_map[i][j] == max_value:
                  x, y = self.grid_coord_to_pose(i, j)
                  rect = patches.Rectangle(
                      (x, y),
                      self.grid_size,
                      self.grid_size,
                      linewidth=1,
                      facecolor=(0, 0.5, 0),
                  )
                  plt.gca().add_patch(rect)
        plt.savefig(file_name)
        
    def save_heatmap_ftws(self, file_name, max_value, visualize_obstacles=True):
        self.set_plt(visualize_obstacles)
        
        max_value = max_value * self.dof
        
        for i in range(self.grid_num):
            for j in range(self.grid_num):
                opacity = 0
                for d in range(self.dof):
                    opacity += self.maps[d][i][j] / max_value
                x, y = self.grid_coord_to_pose(i, j)
                rect = patches.Rectangle(
                    (x, y),
                    self.grid_size,
                    self.grid_size,
                    linewidth=1,
                    facecolor=(0, 0.5, 0, opacity),
                )
                plt.gca().add_patch(rect)
        plt.savefig(file_name)
        
    def get_reachability_map(self, max_value):
        
        max_value = max_value * self.dof

        empty_map = np.zeros((40, 40))

        assert self.grid_num == 40

        for i in range(self.grid_num):
            for j in range(self.grid_num):
                opacity = 0
                for d in range(self.dof):
                    opacity += self.maps[d][i][j] / max_value
                empty_map[i][j] = opacity
        return empty_map
        
    def save_binarymap_ftws(self, file_name, max_value, visualize_obstacles=True):
        self.set_plt(visualize_obstacles)
        
        max_value = max_value * self.dof
        
        for i in range(self.grid_num):
            for j in range(self.grid_num):
                opacity = 0
                for d in range(self.dof):
                    opacity += self.maps[d][i][j] / max_value
                if opacity >= 0.99:
                    x, y = self.grid_coord_to_pose(i, j)
                    rect = patches.Rectangle(
                        (x, y),
                        self.grid_size,
                        self.grid_size,
                        linewidth=1,
                        facecolor=(0, 0.5, 0),
                    )
                    plt.gca().add_patch(rect)
        plt.savefig(file_name)
        
    
    # ik
    def get_ik_solutions(self):
        # find the missing solutions that fk has missed
        pass
    
    # joint ops

    def generate_joint_configurations(self):
        joint_configurations = []

        for i in range(self.dof):
            if i == self.fixed_jid:
                joint_values = np.array([self.fixed_jangle])
            else:
                lower_limit = self.joint_limits[i][0]
                upper_limit = self.joint_limits[i][1]
                joint_values = np.arange(
                    lower_limit, upper_limit + self.step_size, self.step_size
                )
            joint_configurations.append(joint_values)

        joint_configurations_meshgrid = np.meshgrid(*joint_configurations)
        all_joint_configurations = np.vstack(
            [arr.flatten() for arr in joint_configurations_meshgrid]
        ).T
        return all_joint_configurations

    # check collision
    def check_collision(self, start, end):
        [xstart, ystart] = start
        [xend, yend] = end
        for obs in self.obstacles:
            [xc, yc] = obs[0]
            rc = obs[1]
            # check if the obs (circle) interesct with the link (line)
            if line_circle_intersection(xstart, ystart, xend, yend, xc, yc, rc):
                return True
        return False
    
    # pose

    def get_joint_poses(self, cfg):
        # config cfg = [theta_1, theta_2, ...]
        assert len(cfg) == self.dof
        result = []
        xs = ys = 0
        x = y = 0
        angle = 0
        for d in range(self.dof):
            angle = angle - cfg[d]
            x += np.cos(angle) * self.link_lengths[d]
            y += np.sin(angle) * self.link_lengths[d]
            if self.check_collision([xs, ys], [x, y]):
                return None, True
            result.append([x, y])
            xs = x
            ys = y
        return np.array(result), False

    def get_eff_pose(self, cfg):
        joint_poses, has_collision = self.get_joint_poses(cfg)
        if has_collision:
            return None, has_collision
        return joint_poses[-1], has_collision