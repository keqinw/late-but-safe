import os
import numpy as np
import torch
from cem import CEMOptimizer
from randopt import RandomOptimizer
import logging
import math

log = logging.getLogger('root')

class MPC:
    def __init__(self, env, plan_horizon, model, popsize, num_elites, max_iters,
                 num_particles=3,
                 use_gt_dynamics=True,
                 use_mpc=True,
                 use_random_optimizer=False):
        """ 
        Model Predictive Control (MPC) Class
            :param env:
            :param plan_horizon:
            :param model: The learned dynamics model to use, which can be None if use_gt_dynamics is True
            :param popsize: Population size
            :param num_elites: CEM parameter
            :param max_iters: CEM parameter
            :param num_particles: Number of trajectories for TS1
            :param use_gt_dynamics: Whether to use the ground truth dynamics from the environment
            :param use_mpc: Whether to use only the first action of a planned trajectory
            :param use_random_optimizer: Whether to use CEM or take random actions
            :param num_particles: Number of particles 
        """
        self.env = env
        self.plan_horizon = plan_horizon
        self.use_gt_dynamics = use_gt_dynamics
        self.use_mpc = use_mpc
        self.use_random_optimizer = use_random_optimizer
        self.num_particles = num_particles
        self.num_nets = None if model is None else model.num_nets

        self.state_dim, self.action_dim = env.observation_space.shape[0]-1, env.action_space.shape[0]
        self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low

        # Set up optimizer
        self.model = model

        if use_gt_dynamics:
            self.predict_next_state = self.predict_next_state_gt
            assert num_particles == 1
        else:
            self.predict_next_state = self.predict_next_state_model

        # Initialize your planner with the relevant arguments.
        # Use different optimizers for cem and random actions respectively
        optimizer = RandomOptimizer if self.use_random_optimizer else CEMOptimizer
        self.optimizer = optimizer(
            self.get_action_cost,
            out_dim=self.plan_horizon * self.action_dim,
            popsize=popsize,
            num_elites=num_elites,
            max_iters=max_iters,
            lower_bound=np.tile(self.ac_lb, [self.plan_horizon]),
            upper_bound=np.tile(self.ac_ub, [self.plan_horizon])
        )

        # Controller initialization: action planner
        self.curr_obs = np.zeros(self.state_dim)
        self.acs_buff = np.array([]).reshape(0, self.action_dim)  # planned action buffer
        self.prev_sol = np.tile(np.zeros(self.action_dim),
                                self.plan_horizon)  # size: (action_dim * plan_horizon,)
        self.init_var = np.tile(np.ones(self.action_dim) * (0.5 ** 2),  # std = 0.5
                                self.plan_horizon)  # size: (action_dim * plan_horizon,)

        # Controller initialization: dynamics model: predict transition s' = s + f(s, a; phi)
        self.train_in = np.array([]).reshape([0, self.state_dim + self.action_dim])
        self.train_targs = np.array([]).reshape([0, self.state_dim])
        self.has_been_trained = False

    def obs_cost_fn(self, states):
        """ Cost function of the current state """
        # here need to be re-writed
        # my cost function for simple driving:
        # Done by stopping

        # here states do not include the goal
        # cost function is a tricky problem
        # Done by stopping
        rewards = []
        for car_ob in states:
            if car_ob[1] <= 0.001: 
                # Done by reaching goal
                if abs(car_ob[0] - self.goal) < 0.3:
                    reward = 50
                else:
                    dis = abs(car_ob[0] - self.goal)
                    reward = - dis/self.goal
                self.done = True

            else:
                if car_ob[0] > self.goal:
                    dis = abs(car_ob[0] - self.goal)
                    reward = - dis
                else:
                    dis = abs(car_ob[0] - self.goal)
                    reward = - dis/self.goal

            rewards.append(-reward)
        return np.array(rewards)

    def get_action_cost(self, ac_seqs):
        """
        Evaluate the policy (for each member of the CEM population):
            calculate the cost of a state and action sequence pair as the sum of
            cost(s_t, a_t) w.r.t. to the policy rollout for each particle,
            and then aggregate over all particles.

        Arguments:
            ac_seqs: shape = (popsize, plan_horizon * action_dim)
        """

        # keqin: here, the cost of action equals to the cost of the state after deploy this action  
        popsize = np.shape(ac_seqs)[0]
        # One cost per member of population per particle
        total_costs = np.zeros([popsize, self.num_particles])
        cur_obs = np.tile(self.curr_obs[None], [popsize * self.num_particles, 1])

        # ac_seqs.shape: (popsize, T x |A|),
        # reshape to: (popsize, T, |A|)
        ac_seqs = np.reshape(ac_seqs, [-1, self.plan_horizon, self.action_dim])
        # transpose to: (T, popsize, |A|), then ac_seqs[t] gives you action_t of all popsize
        ac_seqs = np.transpose(ac_seqs, [1, 0, 2])

        for t in range(self.plan_horizon):
            # action of current time step, of all popsize
            cur_acs = ac_seqs[t]
            # action_costs = 0.1 * np.tile(np.sum(np.square(cur_acs), axis=1)[:, None], [1, self.num_particles])
            next_obs = self.predict_next_state(cur_obs, cur_acs)
            if isinstance(next_obs, list):
                next_obs = np.stack(next_obs)
            delta_costs = self.obs_cost_fn(next_obs).reshape([popsize, -1]) # + action_costs  # think: do we need action noise ???

            total_costs += delta_costs
            cur_obs = next_obs
        return np.mean(np.where(np.isnan(total_costs), 1e6 * np.ones_like(total_costs), total_costs), axis=1)

    def predict_next_state_model(self, states, actions):
        """ 
        Trajectory Sampling with TS1 (Algorithm 3) using an ensemble of learned dynamics model to predict the next state.
            :param states  : [self.popsize * self.num_particles, self.state_dim]
            :param actions : [self.popsize, self.action_dim]
        """

        # TODO: write your code here
        # REMEMBER: model prediction is delta   
        # Next state = delta sampled from model prediction + CURRENT state!
        
        #format input to model- need to append action to end of each state
        states = torch.tensor(states)
        actions = torch.tensor(actions).repeat_interleave(self.num_particles, dim=0)
        inputs = torch.cat((states, actions), axis=1).float()

        #prediction- (mean, logvar)
        ensemble_predictions = self.model(inputs)

        #separate mean and logvar
        ensemble_means = np.array([p[0].detach().numpy() for p in ensemble_predictions])
        ensemble_logvars = np.array([p[1].detach().numpy() for p in ensemble_predictions])

        #select predictions from random model
        random_model_idx = np.random.choice(len(ensemble_predictions), len(states))
        ensemble_means = ensemble_means[random_model_idx, np.arange(len(states)), :]
        ensemble_logvars = ensemble_logvars[random_model_idx, np.arange(len(states)), :]    

        #generate sampling distribution using the model outputs
        states = states.numpy()
        sd = np.sqrt(np.exp(ensemble_logvars))
        next_states = np.random.normal(ensemble_means, sd) + states
        return next_states
        

    def predict_next_state_gt(self, states, actions):
        """ Given a list of state action pairs, use the ground truth dynamics to predict the next state"""
        # TODO: write your code here
        new_states = []
        for state, action in zip(states, actions):
            next_state = self.env.get_nxt_state(state, action)
            new_states.append(next_state)
            
        return np.array(new_states)

    def train(self, obs_trajs, acs_trajs, rews_trajs, num_train_itrs=5):
        """ 
        Take the input obs, acs, rews and append to existing transitions the train model.   
        Arguments:  
          obs_trajs: states 
          acs_trajs: actions    
          rews_trajs: rewards (NOTE: this may not be used)  
          num_train_itrs: number of iterations to train for
        """
        log.info("Train dynamics model with CEM for %d iterations" % num_train_itrs)
        new_train_in, new_train_targs = [], []
        for obs, acs in zip(obs_trajs, acs_trajs):
            # input (state, action) 
            new_train_in.append(np.concatenate([obs[:-1, 0:2], acs], axis=-1))
            # predict dynamics: f(s, a, phi)    
            new_train_targs.append(obs[1:, 0:2] - obs[:-1, 0:2])
        self.train_in = np.concatenate([self.train_in] + new_train_in, axis=0)
        self.train_targs = np.concatenate([self.train_targs] + new_train_targs, axis=0)
        loss, rmse = self.model.train_model(self.train_in, self.train_targs, num_train_itrs=num_train_itrs)
        self.has_been_trained = True
        return loss

    def reset(self):
        # size: (action_dim * plan_horizon, )
        self.prev_sol = np.tile(np.zeros(self.action_dim), self.plan_horizon)

    def act(self, state, t):
        """
        Choose the action given current state using planning (CEM / Random) with or without MPC.
        
        Tip: You need to fill acs_buff to return the action 

        Arguments:
          state: current state
          t: current timestep
        """

        if not self.has_been_trained and not self.use_gt_dynamics:
            # Use random policy in warmup stage (for Q1.2)
            return np.random.uniform(self.ac_lb, self.ac_ub, self.ac_lb.shape)

        if self.acs_buff.shape[0] > 0:
            action, self.acs_buff = self.acs_buff[0], self.acs_buff[1:]
            print(t,action)
            return action

        self.curr_obs = state[0:-1]
        self.goal = state[-1]

        if self.use_mpc:
            # Use MPC with CEM / Random Policy for planning (generate action sequences)
            # Review the MPC part in the writeup carefully.
            # Save only the first action (at timestep t) in self.acs_buff
            # Think carefully about what to keep in self.prev_sol for the next timestep (t+1):
            # keep the future actions (\mu) planned by CEM / Random policy
            # for t+1, ..., t+T-1 and initialize the action at t+T to 0
            soln = self.optimizer.solve(self.prev_sol, self.init_var)  # size: (action_dim * plan_horizon,)
            # plan_horizon是规划的长度
            # 然后这里，更新previous_sol的方式就是pop掉第一个action
            self.prev_sol = np.concatenate([np.copy(soln)[self.action_dim:], np.zeros(self.action_dim)])
            # MPC: only use the first/next action planned
            self.acs_buff = soln[:self.action_dim].reshape(-1, self.action_dim)

        else:
            # Otherwise, directly use CEM / Random Policy without MPC
            # i.e. use planned future actions up to self.plan_horizon
            self.acs_buff = self.optimizer.solve(self.prev_sol, self.init_var).reshape(-1, self.action_dim)


        return self.act(state, t)
