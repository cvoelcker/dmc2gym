import gym
from gym.envs.registration import register


def make(
        domain_name,
        task_name,
        seed=1,
        difficulty=None,
        dynamic=False,
        background_dataset_path=None,
        background_dataset_videos="train",
        background_kwargs=None,
        camera_kwargs=None,
        render_kwargs=None,
        pixels_only=True,
        pixels_observation_key="pixels",
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        environment_kwargs=None,
        channels_first=True,
        episode_length=1000,
        time_limit=None,
):
    env_id = 'dmc_%s_%s_%s-v1' % (domain_name, task_name, seed)

    # shorten episode length
    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

    if not env_id in gym.envs.registry.env_specs:
        task_kwargs = {}
        if seed is not None:
            task_kwargs['random'] = seed
        if time_limit is not None:
            task_kwargs['time_limit'] = time_limit
        register(
            id=env_id,
            entry_point='distracting_dmc2gym.wrappers:DMCWrapper',
            kwargs=dict(
                domain_name=domain_name,
                task_name=task_name,
                difficulty=difficulty,
                dynamic=dynamic,
                background_dataset_path=background_dataset_path,
                background_dataset_videos=background_dataset_videos,
                background_kwargs=background_kwargs,
                camera_kwargs=camera_kwargs,
                task_kwargs=task_kwargs,
                visualize_reward=False,
                render_kwargs=render_kwargs,
                pixels_only=pixels_only,
                pixels_observation_key=pixels_observation_key,
                height=height,
                width=width,
                camera_id=camera_id,
                frame_skip=frame_skip,
                environment_kwargs=environment_kwargs,
                channels_first=channels_first
            ),
            max_episode_steps=max_episode_steps,
        )
    return gym.make(env_id)
