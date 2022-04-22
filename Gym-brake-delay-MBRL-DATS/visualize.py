import numpy as np
import gym
import simple_driving
import time
samples = np.load('./Gym-brake-delay-MBRL-DATS/actions.npy',allow_pickle=True)
env = gym.make('SimpleDriving-v0')
time.sleep(5)
drifts = [0,0,0,0,0]
for sample, drift in zip(samples,drifts):
    
    goal = sample['obs'][0,-1]
    print(goal-drift)
    env.reset_visual(goal-drift)
    
    for action in sample['init_actions']:
        print(action)
        env.step_visual(action)
        time.sleep(1/10)
    for action in sample['ac']:
        env.step_visual(action)
        time.sleep(1/10)
