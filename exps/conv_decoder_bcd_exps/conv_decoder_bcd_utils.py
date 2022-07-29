from tqdm import tqdm
import gym
import numpy as np
from matplotlib import pyplot as plt
from typing import OrderedDict


def generate_chem_image_dataset(n, d, interv_values, interv_targets, z):
    images = None
    env = gym.make(f'LinGaussColorCubesRL-{d}-{d}-Static-10-v0')

    for i in tqdm(range(n)):
        action = OrderedDict()
        action['nodes'] = np.where(interv_targets[i])
        action['values'] = interv_values[i]
        ob, _, _, _ = env.step(action, z[i])
        
        if i == 0:
            images = ob[1]
        else:
            images = np.concatenate((images, ob[1]), axis=0)

        plt.figure()
        plt.imshow(ob[1] / 255.)
        plt.colorbar()
        plt.show()
        plt.savefig(f'data/image{i}.png')

    return images