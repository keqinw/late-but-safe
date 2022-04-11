import gym
import torch
import simple_driving
import time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback
import os
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('SimpleDriving-v0')
ob = env.reset()


# try random policy 100 times
num = 100
for i in range(num):
    action = env.action_space.sample()
    # action = - 0.5
    # print(action)
    # print(env.previous_action)
    ob, rw, done, _ = env.step(action)
    # print(ob)
    # print(rw)
    train_in = np.concatenate([ob[0:2], env.previous_action, action], axis=-1)
    print(train_in.shape)
    time.sleep(1/30)
    # env.render()
    if done:
        ob = env.reset()
        time.sleep(1/30)
        break
            