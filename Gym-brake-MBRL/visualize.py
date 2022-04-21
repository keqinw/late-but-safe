import numpy as np
import gym
import simple_driving
import time
samples = np.load('./Gym-brake-MBRL/actions.npy',allow_pickle=True)
env = gym.make('SimpleDriving-v0')


for sample in samples:
    goal = sample['obs'][0,-1]
    print(goal)
    ob = env.reset()
    env.set_goal(goal)
    for action in sample['ac']:
        state,reward,done,info = env.step(action)
        time.sleep(1/10)
