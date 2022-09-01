import gym, pdb, torch
from gym import spaces
import numpy as np
from ColorGen import *
from collections import OrderedDict
from dataclasses import dataclass

from PIL import Image
import skimage


import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def fig2rgb_array(fig):
    fig.canvas.draw()
    buffer = fig.canvas.tostring_rgb()
    width, height = fig.canvas.get_width_height()
    return np.fromstring(buffer, dtype=np.uint8).reshape(height, width, 3)

def square(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + width, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)

@dataclass
class Coord:
    x: int
    y: int

    def __add__(self, other):
        return Coord(self.x + other.x,
                     self.y + other.y)


@dataclass
class Object:
    pos: Coord
    color: tuple


class LinGaussColorCubesRL(gym.Env):
    def __init__(self, width=5, height=5, render_type='shapes',
                 *, num_objects=5, num_colors=None, 
                 movement = 'Static', max_steps = 50, seed=None,
                 dataseed = 0):
        #np.random.seed(0)
        #torch.manual_seed(0)
        self.width = width
        self.height = height
        self.render_type = render_type
        self.num_objects = num_objects
        self.low = -8.
        self.high = 8.
        self.collisions = True

        action_dict = {
            'nodes': spaces.Discrete(self.num_objects+1),  # choose from 0,..d nodes to intervene on
            'values': spaces.Box(low=np.array([self.low] * self.num_objects),
                                high=np.array([self.high] * self.num_objects))
        }

        self.observation_space = spaces.Box
        self.action_space = spaces.Dict(action_dict)

        self.colorgen = LinearGaussianColor(1, self.num_objects,
                                            'erdos-renyi', 
                                            2.0, 
                                            'linear-gauss', 
                                            data_seed=dataseed)
        
        self.W = self.colorgen.W
        self.adjacency_matrix = self.colorgen.W.T
        self.objects = OrderedDict()
        self.reset()
    
    def set_graph(self, W):
        self.W = W
    
    def valid_pos(self, pos, obj_id):
        """Check if position is valid."""
        if pos.x not in range(0, self.width):
            return False
        if pos.y not in range(0, self.height):
            return False

        if self.collisions:
            for idx, obj in self.objects.items():
                if idx == obj_id:
                    continue

                if pos == obj.pos:
                    return False

        return True
    
    def render_shapes(self):
        im = np.zeros((self.width * 10, self.height * 10, 3), dtype=np.float32)
        for idx, obj in self.objects.items():
            if True:
                rr, cc = square(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = obj.color[:3]
        
        return im.transpose([2, 0, 1])

    def render(self):
        return np.concatenate(
                    (
                        dict(shapes=self.render_shapes)[self.render_type](), 
                        dict(shapes=self.render_shapes)[self.render_type]()
                    ), 
                axis = 0) 
        
    
    def reset(self):
        """
            Generate observational data
        """
        channel_colors = self.colorgen.capped_obs_sample(self.low, self.high).squeeze(0)
        channel_colors = 255. * ((channel_colors / (2 * self.high)) + 0.5)  
        rgb_colors = []

        for color in channel_colors:
            rgb_colors.append((color, 0., 0., 1))

        self.set_object_details(rgb_colors)
        
        interv_details = ([], np.array([0.] * self.num_objects)) 
        state_obs = self.render()

        return (interv_details, state_obs, self.W), (None, None)

    def set_object_details(self, rgb_colors):
        fixed_object_to_position_mapping = [(0, 0), (0, 4), (4, 0), (4, 4), (2, 2), (1,1), (1, 3), (3, 1), (3,3), (0, 2)]
        
        # Re-sample to ensure objects don't fall on same spot.
        for idx in range(self.num_objects):
            self.objects[idx] = Object(
                pos=Coord(
                    x=fixed_object_to_position_mapping[idx][0],
                    y=fixed_object_to_position_mapping[idx][1],
                ),
                color=rgb_colors[idx])


    def step(self, action, channel_colors=None):
        nodes_to_intervene = action['nodes']
        interv_values = action['values'].reshape(1, -1)
        done = False
        rgb_colors = []

        if channel_colors is None:
            weighted_adj_mat = self.W.copy()
            weighted_adj_mat[:, nodes_to_intervene] = 0

            if len(nodes_to_intervene) == 0:
                channel_colors = self.colorgen.capped_obs_sample(self.low, self.high).squeeze(0)
            else:
                channel_colors = self.colorgen.capped_interv_samples(nodes_to_intervene, interv_values, 
                                                                self.low, self.high).squeeze(0)

            
        normalized_channel_colors = 255. * ((channel_colors / (2 * self.high)) + 0.5)  
        rgb_colors = []
        for color in normalized_channel_colors:
            rgb_colors.append((color, 0., 0., 1))

        self.set_object_details(rgb_colors)
        state_obs = self.render()
        state_obs = state_obs[:3, :, :]
        interv_details = (nodes_to_intervene, interv_values) 
        state_obs = (interv_details, state_obs.T)
        return state_obs, None, done, None

