import os
import platform

from custom_mujoco import get_asset_path
from custom_mujoco.get_asset_path import get_asset_path

if "microsoft" in platform.uname().release.lower():
    # print("Using osmesa for rendering on WSL...")
    os.environ["MUJOCO_GL"] = "osmesa"
else:
    # print("Using egl for rendering...")
    os.environ["MUJOCO_GL"] = "egl"
from custom_mujoco.custom_inverted_pendulum import CustomInvertedPendulum
from custom_mujoco.custom_inverted_double_pendulum import CustomInvertedDoublePendulum
from custom_mujoco.custom_humanoidstandup import CustomHumanoidStandup
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
        "xml_file": get_asset_path("inverted_pendulum.xml"),
        "initial_states": None,
        "init_dist": "uniform",
        "n_rand_initial_states": 100,
        "init_ranges": None,
        "init_mode": "random",
        "dense_reward": False,
        "seed": None,
    },
    max_episode_steps=500,
)

register(
    id="CustomInvertedDoublePendulum",
    entry_point="custom_mujoco:CustomInvertedDoublePendulum",
    kwargs={
        "render_mode": None,
        "xml_file": get_asset_path("inverted_double_pendulum.xml"),
        "pole1_length": 0.6,
        "pole2_length": 0.6,
        "pole1_density": 1000.0,
        "pole2_density": 1000.0,
        "cart_density": 1000.0,
        "hinge1_friction": 0.00,
        "hinge2_friction": 0.00,
        "hinge1_stiffness": 0.0,
        "hinge2_stiffness": 0.0,
        "initial_states": None,
        "init_dist": "uniform",
        "n_rand_initial_states": 100,
        "init_ranges": None,
        "init_mode": "random",
        "dense_reward": False,
        "seed": None,
    },
    max_episode_steps=500,
)

register(
    id="CustomHumanoidStandup",
    entry_point="custom_mujoco:CustomHumanoidStandup",
    kwargs={
        "xml_file": get_asset_path("humanoidstandup.xml"),
        "top_heaviness": 1.0,  # Upper-body mass scaling
        "floor_friction_scale": 1.0,  # Floor friction scaling
        "dense_reward": False,  # Whether to use dense head-ratio reward
        "initial_states": None,  # Predefined initial state set (optional)
        "init_dist": "uniform",  # Distribution for state sampling
        "n_rand_initial_states": 100,  # Number of samples if using random init
        "init_ranges": None,  # Sampling range for state variables
        "init_mode": "random",  # Mode: "random", "sequential", "seeded"
        "seed": None,  # RNG seed
    },
    max_episode_steps=1000,
)
