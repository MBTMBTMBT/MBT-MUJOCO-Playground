import os

os.environ["MUJOCO_GL"] = "egl"
import mujoco
from gymnasium.envs.mujoco.inverted_double_pendulum_v5 import InvertedDoublePendulumEnv
import tempfile
import xml.etree.ElementTree as ET
import numpy as np


def modify_double_pendulum_xml(
    xml_path: str,
    pole1_length: float,
    pole2_length: float,
    pole1_density: float,
    pole2_density: float,
    cart_density: float,
) -> str:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    cpole = root.find(".//geom[@name='cpole']")
    if cpole is not None:
        cpole.set("fromto", f"0 0 0 0 0 {pole1_length}")
        cpole.set("density", str(pole1_density))

    cpole2 = root.find(".//geom[@name='cpole2']")
    if cpole2 is not None:
        cpole2.set("fromto", f"0 0 0 0 0 {pole2_length}")
        cpole2.set("density", str(pole2_density))

    cart = root.find(".//geom[@name='cart']")
    if cart is not None:
        cart.set("density", str(cart_density))

    pole2_body = root.find(".//body[@name='pole2']")
    if pole2_body is not None:
        pole2_body.set("pos", f"0 0 {pole1_length}")

    tip_site = root.find(".//site[@name='tip']")
    if tip_site is not None:
        tip_site.set("pos", f"0 0 {pole2_length}")

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xml", mode="w")
    tree.write(tmp_file.name)
    tmp_file.close()
    return tmp_file.name


class CustomInvertedDoublePendulum(InvertedDoublePendulumEnv):
    """
    A configurable extension of the InvertedDoublePendulum environment, supporting:

    - Customizable physical properties, including pole lengths, densities, and cart density.
    - Fully controllable and reproducible initial state distributions.
    - Flexible reset sampling modes: random, sequential, or deterministic (via seed).
    - Optional dense reward mode that encourages proximity of the pole tip to its maximum height.

    This environment is well-suited for curriculum learning, domain randomization,
    and robustness evaluation under variable dynamics and controlled initialization.

    Args:
        xml_file (str): Path to the base MuJoCo XML model.
        pole1_length (float): Length (in meters) of the first pendulum segment.
        pole2_length (float): Length (in meters) of the second pendulum segment.
        pole1_density (float): Density (kg/m³) of the first pendulum segment.
        pole2_density (float): Density (kg/m³) of the second pendulum segment.
        cart_density (float): Density (kg/m³) of the cart body.
        initial_states (np.ndarray or None): Optional array of predefined initial states (shape: [n, 6]).
                                             If provided, overrides automatic sampling.
        init_dist (str): Sampling distribution to use if `initial_states` is not given. Options: "uniform", "gaussian".
        n_states (int): Number of initial states to sample when using `init_dist`.
        init_ranges (list of tuple): Ranges [(low, high), ...] for each of the 6 initial state dimensions.
        init_mode (str): State sampling mode for environment resets: "random", "sequential", or "seeded".
        seed (int or None): Random seed for reproducible initial state generation.
        dense_reward (bool): If True, uses a smooth exponential reward based on the cart's distance from the center (x=0).
                             If False, uses the original reward structure with position and velocity penalties.
        **kwargs: Additional arguments forwarded to the base `InvertedDoublePendulumEnv`.
    """

    def __init__(
        self,
        xml_file: str = "./custom_mujoco/assets/inverted_double_pendulum.xml",
        pole1_length=0.6,
        pole2_length=0.6,
        pole1_density=1000.0,
        pole2_density=1000.0,
        cart_density=1000.0,
        dense_reward=False,
        initial_states=None,
        init_dist="uniform",
        n_states=100,
        init_ranges=None,
        init_mode="random",
        seed=None,
        **kwargs,
    ):
        # Initialize random number generator for reproducibility
        self._rng = np.random.default_rng(seed)
        self.sample_mode = init_mode
        self._init_index = 0  # Index for sequential or seeded reset order

        self.dense_reward = dense_reward

        # Used for dynamic termination based on the current maximum tip height
        self.max_tip_y = pole1_length + pole2_length
        self.fail_threshold = self.max_tip_y * (1 / 1.2)

        # Modify the original XML file with updated pole lengths and densities
        modified_xml = modify_double_pendulum_xml(
            xml_file,
            pole1_length,
            pole2_length,
            pole1_density,
            pole2_density,
            cart_density,
        )

        # Initialize the parent environment with the modified XML
        super().__init__(xml_file=modified_xml, **kwargs)
        self._temp_xml_path = modified_xml  # Save for later cleanup

        # Default state space ranges (position and velocity)
        self.init_ranges = init_ranges or [(-0.05, 0.05)] * 6

        # Step 1: Use explicitly provided initial_states if available
        if initial_states is not None:
            self.initial_states = np.array(initial_states)
        else:
            # Step 2: Otherwise sample based on distribution and range
            lows = np.array([r[0] for r in self.init_ranges])
            highs = np.array([r[1] for r in self.init_ranges])
            if init_dist == "uniform":
                self.initial_states = self._rng.uniform(
                    low=lows, high=highs, size=(n_states, 6)
                )
            elif init_dist == "gaussian":
                self.initial_states = np.clip(
                    self._rng.normal(loc=0, scale=0.02, size=(n_states, 6)), lows, highs
                )
            else:
                raise ValueError(
                    "Unsupported init_dist: choose 'uniform' or 'gaussian'"
                )

        # Step 3: Precompute reset index order for sequential or seeded modes
        if init_mode == "seeded":
            self._index_order = self._rng.permutation(len(self.initial_states))
        elif init_mode == "sequential":
            self._index_order = np.arange(len(self.initial_states))
        else:
            self._index_order = None  # Random mode: no order required

    def reset_model(self):
        """
        Resets the environment by explicitly setting a sampled initial state.
        The state is selected according to the configured sampling strategy.

        Returns:
            observation (np.ndarray): The initial observation after reset.
        """
        # Select initial state based on reset mode
        if self.sample_mode in ("sequential", "seeded"):
            idx = self._index_order[self._init_index]
            self._init_index = (self._init_index + 1) % len(self.initial_states)
            state = self.initial_states[idx]
        else:
            state = self.initial_states[self._rng.integers(len(self.initial_states))]

        # Set qpos (positions) and qvel (velocities)
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        qpos[:3] = state[:3]  # [cart x, hinge1, hinge2]
        qvel[:3] = state[3:]  # [cart vx, hinge1 v, hinge2 v]

        self.set_state(qpos, qvel)
        return self._get_obs()

    def close(self):
        """
        Cleans up the temporary XML file generated during environment construction.
        Should be called when the environment is no longer needed.
        """
        super().close()
        if hasattr(self, "_temp_xml_path") and os.path.exists(self._temp_xml_path):
            os.remove(self._temp_xml_path)

    def _get_rew(self, x, y, terminated):
        """
        Compute the reward. Supports two reward modes:

        - Default: original reward with dynamic max_tip_y target (uses distance and velocity penalty).
        - Dense: exponential reward based on tip.y proximity to max_tip_y.
        """

        if getattr(self, "dense_reward", False):
            # Dense reward based on cart position relative to center (x=0)
            cart_x = self.data.qpos[0]
            alpha = 2.0  # Controls sharpness of reward drop-off
            reward = np.exp(-alpha * abs(cart_x))

            reward_info = {
                "dense_cart_center_reward": reward,
                "cart_x": cart_x,
                "alpha": alpha,
            }

        else:
            # Original reward (modified to use dynamic max_tip_y instead of hardcoded '2')
            v1, v2 = self.data.qvel[1:3]
            dist_penalty = 0.01 * x ** 2 + (y - self.max_tip_y) ** 2
            vel_penalty = 1e-3 * v1 ** 2 + 5e-3 * v2 ** 2
            alive_bonus = self._healthy_reward * int(not terminated)

            reward = alive_bonus - dist_penalty - vel_penalty

            reward_info = {
                "reward_survive": alive_bonus,
                "distance_penalty": -dist_penalty,
                "velocity_penalty": -vel_penalty,
            }

        return reward, reward_info

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        # Get tip position from the site named "tip"
        x, _, y = self.data.site_xpos[0]
        observation = self._get_obs()

        # Use adaptive termination threshold based on pole lengths
        terminated = bool(y <= self.fail_threshold)
        reward, reward_info = self._get_rew(x, y, terminated)

        info = reward_info
        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info
