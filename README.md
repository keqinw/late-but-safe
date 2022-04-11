# late-but-safe

Describe:

the number of total trial: 100

Gym-brake: no action delay with PPO (60,000 timesteps). Success rate: 1.0

Gym-brake-delay: 5-steps action delay with PPO (50,000 timesteps, if increase to 100,000 timesteps. could also reach 1.0 rate). Success rate: 0.41

Gym-brake-MBRL: no action delay with PETS. Success rate: 1.0

Gym-brake-delay-MBRL: 10-steps action delay with PETS. Success rate: 0.24

Gym-brake-delay-MBRL-DATS: 10-steps action delay with DATS. Success rate: 0.74
