import gym
import numpy as np
import math
import pybullet as p
import pybullet_data
from simple_driving.resources.car import Car
from simple_driving.resources.plane import Plane
from simple_driving.resources.goal import Goal
import matplotlib.pyplot as plt
import time

class SimpleDrivingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # self.action_space = gym.spaces.box.Box(
        #     low=np.array([0, -.6], dtype=np.float32),
        #     high=np.array([1, .6], dtype=np.float32))
        self.action_space = gym.spaces.box.Box(
            low=np.array([-1], dtype=np.float32),
            high=np.array([0], dtype=np.float32))

        self.observation_space = gym.spaces.box.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([10, 10, 10], dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()

        self.client = p.connect(p.DIRECT)
        # self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/10, self.client)

        self.car = None
        self.goal = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()

    def step(self, action):
        # Feed action to the car and get observation of car's state
        acc = self.car.apply_action(action)
        p.stepSimulation()
        car_ob = self.car.get_observation()

        # Compute reward as L2 change in distance to goal
        # dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
        #                           (car_ob[1] - self.goal[1]) ** 2))
        # reward = max(self.prev_dist_to_goal - dist_to_goal, 0)
        # reward = self.prev_dist_to_goal - dist_to_goal
        # self.prev_dist_to_goal = dist_to_goal

        # Done by stopping
        if car_ob[1] <= 0.01: 
            # Done by reaching goal
            if abs(car_ob[0] - self.goal) < 0.3:
                reward = 50
            else:
                dis = abs(car_ob[0] - self.goal)
                reward = - dis
            self.done = True

        else:
            if car_ob[0] > self.goal:
                dis = abs(car_ob[0] - self.goal)
                reward = - dis
            else:
                dis = abs(car_ob[0] - self.goal)
                reward = - dis/self.goal

                # acc_best =  1/2 * car_ob[1]**2 / dis          
                # reward = - abs(acc[0] - acc_best) 
                # print(reward)

                # reward = (self.previous_speed - car_ob[1]

        self.previous_speed = car_ob[1]
            

        ob = np.array(car_ob + (self.goal,), dtype=np.float32)
        return ob, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)
        # Reload the plane
        # Plane(self.client)
        planeId = p.loadURDF("plane.urdf")
        
        # Set the goal to a random target
        # x = (self.np_random.uniform(5, 9) if self.np_random.randint(2) else
        #      self.np_random.uniform(-5, -9))
        # y = (self.np_random.uniform(5, 9) if self.np_random.randint(2) else
        #      self.np_random.uniform(-5, -9))
        x = self.np_random.uniform(3, 7)
        y = 0

        self.goal = x
        self.done = False

        # Reload the car
        self.car = Car(self.client)

        # Visual element of the goal
        Goal(self.client, (self.goal,y))

        # pre-accelerate
        # for i in range(20):
        #     self.car.pre_accelerate()
        #     p.stepSimulation()
        #     car_ob = self.car.get_observation()
        #     print(car_ob)
        #     time.sleep(0.5)

        # Get observation to return
        car_ob = self.car.get_observation()
        # print(car_ob)

        self.previous_speed = car_ob[1]

        return np.array(car_ob + (self.goal,), dtype=np.float32)

    def render(self, mode='human'):
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))

        # Base information
        car_id, client_id = self.car.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in
                    p.getBasePositionAndOrientation(car_id, client_id)]
        pos[2] = 0.2

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Display image
        frame = p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (100, 100, 4))
        self.rendered_img.set_data(frame)
        plt.draw()
        plt.pause(.00001)

    def close(self):
        p.disconnect(self.client)
