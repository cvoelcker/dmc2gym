import distracting_dmc2gym
import matplotlib.pyplot as plt

env = distracting_dmc2gym.make(domain_name='cheetah', task_name='run', difficulty="hard", seed=1, channels_first=False)

done = False
obs = env.reset()
while not done:
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
  plt.imshow(obs)
  plt.show()
  print(obs)
