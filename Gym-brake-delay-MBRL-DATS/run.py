import matplotlib.pyplot as plt
import numpy as np
import torch
import gym
import simple_driving
import os
import logging
from utils import config
from agent import Agent, RandomPolicy
from mpc import MPC
from cem import CEMOptimizer
from model import PENN
import pdb

# Logging
now = config.Now()
log = logging.getLogger('root')
log.setLevel('INFO')
log.addHandler(config.MyHandler())

INFO = 2

# Training params
TASK_HORIZON = 100
PLAN_HORIZON = 10

# CEM params
POPSIZE = 200
NUM_ELITES = 20
MAX_ITERS = 50

# Model params
LR = 1e-3

# Dims
STATE_DIM = 2

# delay steps
DELAY_STEPS = 5


class ExperimentGTDynamics(object):
    def __init__(self, env_name="SimpleDriving-v0", mpc_params=None):
        self.env = gym.make(env_name)
        self.task_horizon = TASK_HORIZON
        self.agent = Agent(self.env)
        self.warmup = False

        # Use Ground Truth Dynamics
        mpc_params["use_gt_dynamics"] = True

        # Model Predictive Contoal (MPC)
        self.cem_policy = MPC(self.env, PLAN_HORIZON, None, POPSIZE, NUM_ELITES, MAX_ITERS, **mpc_params,
                              use_random_optimizer=False)
        self.random_policy = MPC(self.env, PLAN_HORIZON, None, POPSIZE, NUM_ELITES, MAX_ITERS, **mpc_params,
                                 use_random_optimizer=True)

    def test(self, num_episodes, optimizer="cem"):
        samples = []
        for j in range(num_episodes):
            # if j == 0 or (j + 1) % INFO == 0: log.info("Test episode {}".format(j))
            log.info("Test episode {}".format(j))
            samples.append(
                self.agent.sample(
                    self.task_horizon, self.cem_policy if optimizer == "cem" else self.random_policy
                )
            )
            print(samples[-1]["reward_sum"])
            print(samples[-1]["rewards"][-1]== 50)
        avg_return = np.mean([sample["reward_sum"] for sample in samples])
        avg_success = np.mean([sample["rewards"][-1] == 50 for sample in samples])
        return avg_return, avg_success


class ExperimentModelDynamics:
    def __init__(self, env_name="SimpleDriving-v0", num_nets=1, mpc_params=None, device=None):
        self.env = gym.make(env_name)
        self.task_horizon = TASK_HORIZON

        self.agent = Agent(self.env)
        mpc_params["use_gt_dynamics"] = False
        # dynamics model f(s, a, phi)
        self.model = PENN(num_nets, DELAY_STEPS,STATE_DIM, len(self.env.action_space.sample()), LR, device=device)
        self.cem_policy = MPC(self.env, PLAN_HORIZON, self.model, POPSIZE, NUM_ELITES, MAX_ITERS, **mpc_params,
                              use_random_optimizer=False)
        self.random_policy = MPC(self.env, PLAN_HORIZON, self.model, POPSIZE, NUM_ELITES, MAX_ITERS, **mpc_params,
                                 use_random_optimizer=True)

    def test(self, num_episodes, optimizer="cem"):
        samples = []
        for j in range(num_episodes):
            if j == 0 or (j + 1) % INFO == 0: log.info("Test episode {}".format(j))
            samples.append(
                self.agent.sample(
                    self.task_horizon, self.cem_policy if optimizer == "cem" else self.random_policy
                )
            )
            traj = samples[-1]["obs"]
            if j == 0 or (j + 1) % INFO == 0: log.info("If success: {}".format(samples[-1]['rewards'][-1]==50))
            
            # # plot the result
            # plt.clf()
            # x = [i for i in range(traj.shape[0])]
            # plt.plot(x,traj[:,0],"*")
            # plt.fill_between(x, traj[:,-1]-0.3, traj[:,-1]+0.3, color='b', alpha=.1)
            # plt.show()

        avg_return = np.mean([sample["reward_sum"] for sample in samples])
        avg_success = np.mean([sample["rewards"][-1] == 50 for sample in samples])
        return avg_return, avg_success

    def model_warmup(self, num_episodes, num_train_itrs):
        """ 
        Train a single probabilistic model using a random policy 
            :param num_episodes: randomly sampled episodes for training the a single probabilistic network
            :param num_train_itrs: number of iterations to pre-train the ensemble of networks for
        """
        traj_obs, traj_acs, traj_rews = [], [], []

        samples = []
        for i in range(num_episodes):
            if i == 0 or (i + 1) % 100 == 0: log.info("Warm up episode %d" % (i))
            samples.append(self.agent.random_sample(self.task_horizon, self.random_policy))

            traj_obs.append(samples[-1]["obs"])
            traj_acs.append(samples[-1]["ac"])
            traj_rews.append(samples[-1]["rewards"])

        losses = self.cem_policy.train(
            [sample["obs"] for sample in samples],
            [sample["ac"] for sample in samples],
            [sample["rewards"] for sample in samples],
            [sample["previous_actions"] for sample in samples],
            [sample["real_obs"] for sample in samples],
            num_train_itrs=num_train_itrs
        )
        self.random_policy.model = self.cem_policy.model
        self.random_policy.has_been_trained = True

        self.traj_obs, self.traj_acs, self.traj_rews = traj_obs, traj_acs, traj_rews
        self.warmup = True
        return losses

    def train(self, num_train_itrs, num_episodes_per_itr, evaluation_interval):
        """ 
        MBRL with PETS (Algorithm 1)
        """
        if self.warmup:
            traj_obs, traj_acs, traj_rews = self.traj_obs, self.traj_acs, self.traj_rews
        else:
            traj_obs, traj_acs, traj_rews = [], [], []

        cme_test, rnd_test, loss = [], [], []
        for i in range(num_train_itrs):
            log.info("####################################################################")
            log.info("Starting training iteration %d." % (i + 1))

            samples = []
            for j in range(num_episodes_per_itr):
                samples.append(
                    self.agent.sample(
                        self.task_horizon, self.cem_policy
                    )
                )
            log.info("Rewards obtained: {}, Rollout Length: {}".format(
                [sample["reward_sum"] for sample in samples], len(samples[-1]["obs"])))
            traj_obs.extend([sample["obs"] for sample in samples])
            traj_acs.extend([sample["ac"] for sample in samples])
            traj_rews.extend([sample["rewards"] for sample in samples])

            losses = self.cem_policy.train(
                [sample["obs"] for sample in samples],
                [sample["ac"] for sample in samples],
                [sample["rewards"] for sample in samples],
                num_train_itrs=5
            )
            loss.append(losses[-1])
            self.random_policy.model = self.cem_policy.model  # model is PENN
            self.random_policy.has_been_trained = True

            # Q1.2.7: Test with both CEM + MPC & Random Policy + MPC
            if i == 0 or (i + 1) % evaluation_interval == 0:
                avg_return, avg_success = self.test(20, optimizer="cem")
                log.info("Test success CEM + MPC: {}".format(avg_success))
                cme_test.append([avg_return, avg_success])
            #     avg_return, avg_success = self.test(20, optimizer="random")
            #     log.info("Test success Random + MPC: {}".format(avg_success))
            #     rnd_test.append([avg_return, avg_success])

        return cme_test, rnd_test, loss

def plot_loss(losses, title, fpath):
    plt.figure(dpi=100)
    plt.plot(range(len(losses)),losses)
    plt.xlabel('Iterations')
    plt.ylabel('Negative Log Likelihood')
    plt.title(title)
    plt.tight_layout(pad=0)
    plt.savefig(fpath)


def train_single_dynamics(num_test_episode=50, device=None):
    log.info("### Q2.2.1: Train a single dynamics model f(s, a, phi) using a random policy")
    num_nets = 2
    num_episodes = 1000
    num_train_itrs = 300
    mpc_params = {"use_mpc": True, "num_particles": 3}
    exp = ExperimentModelDynamics(env_name="SimpleDriving-v0", num_nets=num_nets, mpc_params=mpc_params, device=device)

    log.info("### Q2.2.1: Train from 1000 randomly sampled episodes with 100 iterations")
    losses = exp.model_warmup(num_episodes=num_episodes, num_train_itrs=num_train_itrs)
    plot_loss(losses, '2.2.1: Single Network Training', 'out/loss.png')

    # log.info("### Q2.2.2: Test with Random Policy for %d episodes" % num_test_episode)
    # avg_reward, avg_success = exp.test(num_test_episode, optimizer="random")
    # log.info("Single + Random: avg_reward: {}, avg_success: {}".format(avg_reward, avg_success))

    log.info("### Q2.2.3: Test with CEM for %d episodes" % num_test_episode)
    avg_reward, avg_success = exp.test(num_test_episode, optimizer="cem")
    log.info("Single + CEM: avg_reward: {}, avg_success: {}".format(avg_reward, avg_success))

if __name__ == "__main__":
    gpu_number = 0
    device = torch.device('cuda:%d' % gpu_number if torch.cuda.is_available() else 'cpu')
    train_single_dynamics(10, device=device)   # Q2.2