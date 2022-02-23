import distracting_dmc2gym
import matplotlib.pyplot as plt

DAVIS_PATH = "/home/claas/.distracting_dmc/DAVIS/JPEGImages/480p/"

env = distracting_dmc2gym.make(
    domain_name="cheetah",
    task_name="run",
    difficulty="hard",
    seed=1,
    channels_first=False,
    background_dataset_path=DAVIS_PATH,
    background_dataset_videos="train",
    pixels_only=True,
    pixels_observation_key="pixels")

done = False
obs = env.reset()
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    plt.imshow(obs)
    plt.show()
    print(obs)
