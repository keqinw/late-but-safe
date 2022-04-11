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

# save the best model (copy from the https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#using-callback-monitoring-training)
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model_3')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True


def main():
    # Create log dir
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make('SimpleDriving-v0')
    env = Monitor(env, log_dir)

    ob = env.reset()

    model = PPO('MlpPolicy',env,verbose=0)

    # train the model
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    timesteps = 80000
    model.learn(total_timesteps=timesteps,callback=callback)

    plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "PPO")
    plt.show()

    # evaluate the model 
    model = PPO.load('./tmp/best_model_3.zip')
    a = 0
    num = 100
    for i in range(num):
      while True:
          action,_state= model.predict(ob,deterministic=True)
          ob, _, done, _ = env.step(action)
          # time.sleep(0.1)
          # env.render()
          if done:
              if ob[1] <= 0.01 and abs(ob[0] - ob[2]) <= 0.3:
                a += 1
              ob = env.reset()
              print(i)
              break

    print('Success rate is %.2f'%(a/num))
                


if __name__ == '__main__':
    main()

