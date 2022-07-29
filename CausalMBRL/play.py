import h5py
import gym
from PIL import Image
import envs
import matplotlib
from matplotlib import pyplot as plt

env = gym.make('ColorChangingRL-3-3-Static-10-v0')
_ = env.reset()

obj_idx = 2
direction = 4
obs, _, _, _ = env.step(obj_idx * 5 + direction)

plt.figure()
plt.imshow(obs[1].T)
plt.colorbar()
plt.show()