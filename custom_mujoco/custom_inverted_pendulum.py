import os
import mujoco
from gymnasium.envs.mujoco.inverted_pendulum_v5 import InvertedPendulumEnv
import tempfile
import xml.etree.ElementTree as ET
import numpy as np

from custom_mujoco import get_asset_path


def modify_inverted_pendulum_xml(
    xml_file: str,
    length: float,
    pole_density: float,
    cart_density: float,
) -> str:
    """
    Modify an inverted pendulum XML by updating pendulum length and density values for pole and cart.

    Args:
        xml_file (str): Path to the original XML file.
        length (float): Pendulum length in meters.
        pole_density (float): Density of the pendulum body (kg/m³).
        cart_density (float): Density of the cart body (kg/m³).

    Returns:
        str: Path to the modified temporary XML file.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Update pole length
    cpole = root.find(".//geom[@name='cpole']")
    if cpole is not None:
        cpole.set("fromto", f"0 0 0 0.001 0 {length}")
        cpole.set("density", str(pole_density))
    else:
        raise ValueError("Cannot find geom with name 'cpole' in the XML.")

    # Update cart density
    cart_geom = root.find(".//geom[@name='cart']")
    if cart_geom is not None:
        cart_geom.set("density", str(cart_density))
    else:
        raise ValueError("Cannot find geom with name 'cart' in the XML.")

    # Remove <inertial> if any (optional, MuJoCo will recompute from density and size)
    pole_body = root.find(".//body[@name='pole']")
    if pole_body is not None:
        existing_inertial = pole_body.find("inertial")
        if existing_inertial is not None:
            pole_body.remove(existing_inertial)
    else:
        raise ValueError("Cannot find body with name 'pole' in the XML.")

    # Save updated XML to a temporary file
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xml", mode="w")
    tree.write(tmp_file.name)
    tmp_file.close()
    return tmp_file.name


class CustomInvertedPendulum(InvertedPendulumEnv):
    """
    Extended version of the MuJoCo Inverted Pendulum environment,
    which allows precise control over the initial state distribution,
    system parameters, and the way that episodes are reset.

    Key features:
    - Ability to modify the pendulum length, pole density, and cart density via XML patching at construction time.
    - Flexible initial state sampling: support for uniform or Gaussian distributions, or providing a user-defined finite set of initial states.
    - Option to set a fixed number of sampled initial states (`n_rand_initial_states`).
    - Ability to explicitly specify the value ranges for each initial state variable (`init_ranges`).
    - Multiple sampling policies for the reset order: purely random, sequential cycling, or pseudo-random deterministic cycling using a given seed.

    Args:
        length (float): Length in metres of the pendulum pole.
        pole_density (float): Density (kg/m³) for the pendulum body.
        cart_density (float): Density (kg/m³) for the cart body.
        xml_file (str): Path to the MuJoCo XML describing the inverted pendulum.
        initial_states (np.ndarray or None): Optional. If provided, use this finite set of 4D state vectors for environment reset. Each state is in the form [cart position, cart velocity, pole angle, pole angular velocity].
        initial_state_idxs (list[int] or None): Optional. If provided, only use the specified subset of `initial_states` by index. Useful for curriculum learning or selective evaluation.
        init_dist (str): Sampling distribution to generate random initial states if `initial_states` is not provided. Must be one of 'uniform' or 'gaussian' (default: 'uniform').
        n_rand_initial_states (int): Number of initial states to generate when using automatic sampling (default: 100).
        init_ranges (list of tuple): Ranges [(low, high), ...] for each of the 4 state variables. Defaults to [(-0.01, 0.01)] * 4 if not specified.
        init_mode (str): Sampling policy to choose from the initial state pool. Options:
            - 'random': Sample randomly at each reset.
            - 'sequential': Cycle through initial states in order.
            - 'seeded': Deterministic pseudo-random order using the given seed.
        dense_reward (bool): Whether to use a dense reward function based on distance from the vertical upright position (default: False).
        seed (int or None): Random seed for deterministic sampling and shuffling (used in 'seeded' mode).
        **kwargs: Additional keyword arguments are passed directly to the base class `InvertedPendulumEnv`.

    reset Args:
        state (np.ndarray, optional): If provided, overrides all sampling and directly sets the environment to the given 4D state.
        state_idx (int, optional): If provided, overrides sampling and uses the state at the specified index in `initial_states`.

    Notes:
        - The code ensures that the underlying MuJoCo model XML is modified on-the-fly each time an instance is created, and the temporary file
          is properly cleaned up when the environment is closed.
        - For init_dist='uniform', sampling is done independently per state variable within the specified ranges.
        - For init_dist='gaussian', sampling is also independent per dimension, but clipped to the bounds set by init_ranges.
        - Explicitly providing `initial_states` has highest priority and overrides all automatic sampling.
        - State is always specified in the order: [cart position, cart velocity, pole angle, pole angular velocity].

    Typical usage:
        env = CustomInvertedPendulum(length=0.7, pole_density=1200, n_rand_initial_states=500,
                                    init_ranges=[(-0.1, 0.1)]*4, init_mode="sequential")
    """

    def __init__(
        self,
        length: float = 0.6,
        pole_density: float = 1000.0,
        cart_density: float = 1000.0,
        xml_file: str = get_asset_path("inverted_pendulum.xml"),
        initial_states=None,
        initial_state_idxs=None,
        init_dist: str = "uniform",
        n_rand_initial_states: int = 100,
        init_ranges: list = None,
        init_mode: str = "random",
        dense_reward: bool = False,
        seed: int = None,
        **kwargs,
    ):
        # Patch the XML with the specified cart/pole parameters.
        modified_xml_file = modify_inverted_pendulum_xml(
            xml_file, length, pole_density, cart_density
        )
        super().__init__(xml_file=modified_xml_file, **kwargs)
        self._temp_xml_path = modified_xml_file

        self.dense_reward = dense_reward
        self.cart_range = 1.0
        self.length = length

        # Initialise RNG for determinism if seed is provided.
        self._rng = np.random.default_rng(seed)
        self.sample_mode = init_mode
        self._init_index = 0

        # Default state space ranges (position and velocity)
        self.init_ranges = init_ranges or [(-0.01, 0.01)] * 4

        # Step 1: Explicit initial_states input
        if initial_states is not None:
            all_states = np.array(initial_states)
        else:
            lows = np.array([r[0] for r in self.init_ranges])
            highs = np.array([r[1] for r in self.init_ranges])
            if init_dist == "uniform":
                all_states = self._rng.uniform(
                    low=lows, high=highs, size=(n_rand_initial_states, 4)
                )
            elif init_dist == "gaussian":
                all_states = np.clip(
                    self._rng.normal(
                        loc=0, scale=0.02, size=(n_rand_initial_states, 4)
                    ),
                    lows, highs,
                )
            else:
                raise ValueError("Unsupported init_dist: choose 'uniform' or 'gaussian'")

        # Step 2: Subset selection using initial_state_idxs
        if initial_state_idxs is not None:
            self.initial_states = all_states[np.array(initial_state_idxs)]
        else:
            self.initial_states = all_states

        if self.sample_mode == "seeded":
            self._index_order = np.arange(len(self.initial_states))
            self._rng.shuffle(self._index_order)
        elif self.sample_mode == "sequential":
            self._index_order = np.arange(len(self.initial_states))
        else:
            self._index_order = None

    def reset(self, state: np.ndarray = None, state_idx: int = None, **kwargs):
        """
        Reset the environment and set the initial simulator state.
        Supports override with an explicit state or state index.

        Args:
            state (np.ndarray, optional): A specific 4-dim initial state.
            state_idx (int, optional): An index into the initial_states array.

        Returns:
            obs (np.ndarray): The observation after environment reset.
            info (dict): Additional environment information.
        """
        if state is not None:
            # Use directly provided initial state
            assert state.shape == (4,), "Provided state must be a 4-dimensional array."
        elif state_idx is not None:
            # Use specific index from initial_states
            assert 0 <= state_idx < len(self.initial_states), "Invalid state_idx."
            state = self.initial_states[state_idx]
        else:
            # Default sampling
            if self.sample_mode in ("sequential", "seeded"):
                idx = self._index_order[self._init_index]
                self._init_index = (self._init_index + 1) % len(self.initial_states)
                state = self.initial_states[idx]
            else:
                idx = self._rng.integers(len(self.initial_states))
                state = self.initial_states[idx]

        # Convert to MuJoCo state format: qpos and qvel
        qpos = np.array([state[0], state[2]], dtype=np.float64)
        qvel = np.array([state[1], state[3]], dtype=np.float64)
        self.set_state(qpos, qvel)

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.dense_reward:
            # Get current cart position and pole angle (assuming obs[0] is cart position, obs[2] is pole angle)
            cart_pos = obs[0]
            pole_angle = obs[2]  # pole angle in radians

            # Calculate horizontal position of the pole tip
            pole_tip_pos = cart_pos + self.length * np.sin(pole_angle)

            # Normalize distance: absolute distance of pole tip from the center line,
            # divided by the maximum possible (cart max range + pole length)
            max_dist = self.cart_range + self.length
            distance = np.abs(pole_tip_pos)
            norm_dist = distance / max_dist

            # Exponential reward: closer to the center line, higher the reward at a much faster rate.
            # alpha > 0 adjusts the steepness, typical value in [3, 10]
            alpha = 3.0
            reward = np.exp(
                -alpha * norm_dist
            )  # Best at the center (reward approaches 1), drops off quickly

        return obs, reward, terminated, truncated, info

    def close(self):
        """
        Clean up the temporary XML file created for this environment instance.
        Always call this method when the environment is no longer needed to avoid
        leaving unnecessary files in the filesystem.
        """
        super().close()
        if hasattr(self, "_temp_xml_path") and os.path.exists(self._temp_xml_path):
            os.remove(self._temp_xml_path)
