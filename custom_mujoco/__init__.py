import os
from custom_mujoco.custom_inverted_pendulum import CustomInvertedPendulum
from custom_mujoco.utils import EvalProgressGifCallback
from gymnasium.envs.registration import register


register(
    id="CustomInvertedPendulum",
    entry_point="custom_mujoco:CustomInvertedPendulum",
    kwargs={
        "render_mode": None,
        "length": 0.6,
        "pole_density": 1000.0,
        "cart_density": 1000.0,
        "xml_file": os.path.join(os.getcwd(), "./custom_mujoco/assets/inverted_pendulum.xml"),
        "initial_states": None,
        "init_dist": "uniform",
        "n_states": 100,
        "init_ranges": None,
        "init_mode": "random",
        "dense_reward": False,
        "seed": None,
    },
    max_episode_steps=500,
)
