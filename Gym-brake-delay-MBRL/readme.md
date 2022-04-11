# Gym-brake-delay-MBRL
Basic OpenAI gym environment. 

This package is the delay version with MBRL, which means,1o action-delay, with the baseline PETS

How to use this package?
* Run the run.py to train the dynamics model and get actions and visualize the result.
* no model will be saved, the loss of dynamics model will be ploted in ./out/loss.png
* Use CEM (cross-entropy-method) to get the action 
* The result is saved in ./out/log.txt
* The actions are stored in actions.npy, which is used to visualize the result in Pybullet.

Note: The success rate is around 0.24.

![alt text](https://github.com/keqinw/late-but-safe/blob/master/Gym-brake-delay-MBRL/brake_delay_MBRL.gif?)

