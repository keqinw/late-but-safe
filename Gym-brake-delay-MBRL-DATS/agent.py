import numpy as np
import copy

class Agent:
    def __init__(self, env):
        self.env = env

    def random_sample(self, horizon, policy):
        """
        Sample a rollout from the agent.

        Arguments:
          horizon: (int) the length of the rollout
          policy: the policy that the agent will use for actions
        """
        rewards = []
        previous_actions = []
        
        states, actions, reward_sum, done = [self.env.reset()], [], 0, False

        real_obs = [states[0][0:2]]

        policy.reset()

        for t in range(horizon):
            actions.append(policy.act(states[t],[], t))
            # here we record the previous action, which is needed in dynamics model training
            previous_actions.append(copy.copy(self.env.previous_action))
            # apply action
            state, reward, done, info = self.env.step(actions[t])
            states.append(state)
            reward_sum += reward
            rewards.append(reward)
            real_obs.append(np.array(self.env.ob_buffer[-1],dtype=np.float32))

            if done:
                break

        # print("Rollout length: %d,\tTotal reward: %d,\t Last reward: %d" % (len(actions), reward_sum), reward)

        return {
            "obs": np.array(states),
            "ac": np.array(actions),
            "reward_sum": reward_sum,
            "rewards": np.array(rewards),
            "previous_actions": np.array(previous_actions),
            "real_obs": np.array(real_obs)
        }

    def sample(self, horizon, policy):
        """
        Sample a rollout from the agent.

        Arguments:
          horizon: (int) the length of the rollout
          policy: the policy that the agent will use for actions
        """
        rewards = []
        
        states, actions, reward_sum, done = [self.env.reset()], [], 0, False
        previous_actions = self.env.previous_action

        policy.reset()
        for t in range(horizon):
            actions.append(policy.act(states[t], previous_actions, t))
            state, reward, done, real_state = self.env.step(actions[t])
            print(real_state)
            states.append(state)
            reward_sum += reward
            rewards.append(reward)
            print('current position:',real_state[0:2])
            print('goal:',real_state[-1])
            print("distance:",real_state[-1]-real_state[0])
            if done:
                break
           
        # print("Rollout length: %d,\tTotal reward: %d,\t Last reward: %d" % (len(actions), reward_sum), reward)

        return {
            "obs": np.array(states),
            "ac": np.array(actions),
            "reward_sum": reward_sum,
            "rewards": np.array(rewards),
        }
             

class RandomPolicy:
    def __init__(self, action_dim):
        self.action_dim = 1

    def reset(self):
        pass

    def act(self, arg1, arg2):
        print(np.random.uniform(-0.6,0.6))
        # return (np.random.uniform(1)[0],np.random.uniform(1)[0] * 1.2 - 0.6)
        # return np.random.choice([0,1,2,3,4],1)[0]


if __name__ == "__main__":
    import gym

    env = gym.make("Pushing2D-v1")
    policy = RandomPolicy(2)
    agent = Agent(env)
    for _ in range(5):
        agent.sample(20, policy)
        
