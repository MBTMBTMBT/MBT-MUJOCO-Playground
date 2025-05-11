import os
import platform
if "microsoft" in platform.uname().release.lower():
    # print("Using osmesa for rendering on WSL...")
    os.environ["MUJOCO_GL"] = "osmesa"
else:
    # print("Using egl for rendering...")
    os.environ["MUJOCO_GL"] = "egl"
import importlib.resources as pkg_resources
from custom_mujoco.custom_inverted_pendulum import CustomInvertedPendulum
from custom_mujoco.custom_inverted_double_pendulum import CustomInvertedDoublePendulum
from custom_mujoco.utils import EvalProgressGifCallback
from gymnasium.envs.registration import register


def get_asset_path(filename):
    """Get absolute path to an asset file."""
    try:
        # Python 3.9+
        with pkg_resources.as_file(
            pkg_resources.files("custom_mujoco.assets") / filename
        ) as path:
            return str(path)
    except (ImportError, AttributeError):
        package_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(package_dir, "assets", filename)


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
        "joint_friction": 0.0,
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
