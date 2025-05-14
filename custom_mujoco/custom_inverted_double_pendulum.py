import os

import mujoco
from gymnasium.envs.mujoco.inverted_double_pendulum_v5 import InvertedDoublePendulumEnv
import tempfile
import xml.etree.ElementTree as ET
import numpy as np

from custom_mujoco import get_asset_path


def modify_double_pendulum_xml(
    xml_path: str,
    pole1_length: float,
    pole2_length: float,
    pole1_density: float,
    pole2_density: float,
    cart_density: float,
    hinge1_friction: float,
    hinge2_friction: float,
    # hinge1_stiffness: float,
    # hinge2_stiffness: float,
) -> str:

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Modify geom lengths and densities
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

    # Adjust second pole position (to match new first pole length)
    pole2_body = root.find(".//body[@name='pole2']")
    if pole2_body is not None:
        pole2_body.set("pos", f"0 0 {pole1_length}")

    # Adjust site tip to match pole2 length
    tip_site = root.find(".//site[@name='tip']")
    if tip_site is not None:
        tip_site.set("pos", f"0 0 {pole2_length}")

    # Modify damping and stiffness for both joints
    hinge = root.find(".//joint[@name='hinge']")
    if hinge is not None:
        hinge.set("damping", str(hinge1_friction))
        # hinge.set("springstiff", str(hinge1_stiffness))
        # hinge.set("springref", "0")

    hinge2 = root.find(".//joint[@name='hinge2']")
    if hinge2 is not None:
        hinge2.set("damping", str(hinge2_friction))
        # hinge2.set("springstiff", str(hinge2_stiffness))
        # hinge2.set("springref", "0")

    # Save the modified XML to a temporary file
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xml", mode="w")
    tree.write(tmp_file.name)
    tmp_file.close()
    return tmp_file.name


class CustomInvertedDoublePendulum(InvertedDoublePendulumEnv):
    """
    A configurable extension of the InvertedDoublePendulum environment, supporting:

    - Customizable physical properties:
        - Independent control of pole lengths and densities.
        - Adjustable cart body density.
        - Tunable joint friction (viscous damping).
        - Optional spring-like restoring forces at each joint via joint stiffness.

    - Flexible and reproducible initial state setup:
        - Supports user-provided initial states or automatic sampling.
        - Uniform or Gaussian distributions across configurable ranges.
        - Reset modes include random, sequential, or deterministic via seed.
        - Optionally restrict to a subset of initial states via `initial_state_idxs`.

    - Termination condition automatically adapts to the combined pole length,
      ensuring consistency even under structural changes.

    - Reward structure:
        - Default (False): original Gym-style reward with survival bonus, distance penalty, and velocity penalty.
        - Dense (True): exponential reward encouraging the cart to remain near the center (x ≈ 0).

    This environment is ideal for curriculum learning, domain randomization,
    robustness evaluation, and ablation studies involving physical structure variation.

    Args:
        xml_file (str): Path to the base MuJoCo XML model.
        pole1_length (float): Length (in meters) of the first pendulum segment.
        pole2_length (float): Length (in meters) of the second pendulum segment.
        pole1_density (float): Density (kg/m³) of the first pendulum.
        pole2_density (float): Density (kg/m³) of the second pendulum.
        cart_density (float): Density (kg/m³) of the cart.
        hinge1_friction (float): Viscous damping at the first joint (cart ↔ pole1).
        hinge2_friction (float): Viscous damping at the second joint (pole1 ↔ pole2).
        hinge1_stiffness (float): Restoring spring stiffness (N·m/rad) for the first joint.
        hinge2_stiffness (float): Restoring spring stiffness (N·m/rad) for the second joint.
        dense_reward (bool): Whether to use dense reward based on cart position (exp(-|x|)). If False, use classic reward.
        initial_states (np.ndarray or None): Optional fixed list of initial states (shape: [n, 6]).
        initial_state_idxs (list[int] or None): Optional indices into `initial_states`, specifying a subset to use for reset sampling.
        init_dist (str): Distribution to sample initial states ("uniform" or "gaussian").
        n_rand_initial_states (int): Number of samples to generate if no explicit initial_states are provided.
        init_ranges (list of tuple): Ranges [(low, high), ...] for each state dimension, in order [cart position, pole1 angle, pole2 angle, cart velocity, pole1 angular velocity, pole2 angular velocity]. Defaults to [(-0.01, 0.01)] * 6 if None.
        init_mode (str): Mode for selecting initial states: "random", "sequential", or "seeded".
        seed (int or None): Random seed to ensure reproducibility.
        **kwargs: Additional keyword arguments passed to the base `InvertedDoublePendulumEnv`.
    """

    def __init__(
        self,
        xml_file: str = get_asset_path("inverted_double_pendulum.xml"),
        pole1_length: float = 0.6,
        pole2_length: float = 0.6,
        pole1_density: float = 1000.0,
        pole2_density: float = 1000.0,
        cart_density: float = 1000.0,
        hinge1_friction: float = 0.0,
        hinge2_friction: float = 0.0,
        hinge1_stiffness: float = 0.0,
        hinge2_stiffness: float = 0.0,
        dense_reward: bool = False,
        initial_states=None,
        initial_state_idxs=None,
        init_dist="uniform",
        n_rand_initial_states=100,
        init_ranges=None,
        init_mode="random",
        seed=None,
        **kwargs,
    ):
        self._rng = np.random.default_rng(seed)
        self.sample_mode = init_mode
        self._init_index = 0

        self.dense_reward = dense_reward
        self.hinge1_friction = hinge1_friction
        self.hinge2_friction = hinge2_friction
        self.hinge1_stiffness = hinge1_stiffness
        self.hinge2_stiffness = hinge2_stiffness

        self.max_tip_y = pole1_length + pole2_length
        self.fail_threshold = self.max_tip_y * 0.5

        modified_xml = modify_double_pendulum_xml(
            xml_path=xml_file,
            pole1_length=pole1_length,
            pole2_length=pole2_length,
            pole1_density=pole1_density,
            pole2_density=pole2_density,
            cart_density=cart_density,
            hinge1_friction=hinge1_friction,
            hinge2_friction=hinge2_friction,
            # hinge1_stiffness=hinge1_stiffness,
            # hinge2_stiffness=hinge2_stiffness,
        )

        super().__init__(xml_file=modified_xml, **kwargs)
        self._temp_xml_path = modified_xml

        # Default state space ranges (position and velocity)
        self.init_ranges = init_ranges or [(-0.01, 0.01)] * 6

        # Step 1: Determine full initial state pool
        if initial_states is not None:
            all_states = np.array(initial_states)
        else:
            lows = np.array([r[0] for r in self.init_ranges])
            highs = np.array([r[1] for r in self.init_ranges])
            if init_dist == "uniform":
                all_states = self._rng.uniform(low=lows, high=highs, size=(n_rand_initial_states, 6))
            elif init_dist == "gaussian":
                all_states = np.clip(
                    self._rng.normal(loc=0, scale=0.02, size=(n_rand_initial_states, 6)),
                    lows, highs,
                )
            else:
                raise ValueError("Unsupported init_dist: choose 'uniform' or 'gaussian'")

        # Step 2: Use only the selected subset if indices are provided
        if initial_state_idxs is not None:
            self.initial_states = all_states[np.array(initial_state_idxs)]
        else:
            self.initial_states = all_states

        # Step 3: Compute index order for deterministic modes
        if init_mode == "seeded":
            self._index_order = self._rng.permutation(len(self.initial_states))
        elif init_mode == "sequential":
            self._index_order = np.arange(len(self.initial_states))
        else:
            self._index_order = None

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        """
        Reset the environment. Overrides the default reset method to support custom
        state or state_idx from options dictionary.

        Args:
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Can contain the following keys:
                - 'state': np.ndarray of shape (6,), used to manually specify the initial state.
                - 'state_idx': int, used to pick a state from `self.initial_states`.

        Returns:
            observation (np.ndarray): Initial observation.
            info (dict): Reset-related info.
        """
        super().reset(seed=seed)

        # Reset MuJoCo data to match XML model
        mujoco.mj_resetData(self.model, self.data)

        # Parse options
        state = options.get("state") if options is not None else None
        state_idx = options.get("state_idx") if options is not None else None

        # Call reset_model with control
        observation = self.reset_model(state=state, state_idx=state_idx)
        info = self._get_reset_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def reset_model(self, state: np.ndarray = None, state_idx: int = None):
        """
        Resets the environment by selecting or specifying an initial state.

        Args:
            state (np.ndarray, optional): If provided, use this exact 6D state directly.
            state_idx (int, optional): If provided, use the state at the given index in `initial_states`.

        Returns:
            observation (np.ndarray): The initial observation after reset.
        """
        if state is not None:
            # Use explicitly provided state
            assert state.shape == (6,), "Provided state must be a 6-dimensional array."
        elif state_idx is not None:
            # Use specified index from state pool
            assert 0 <= state_idx < len(self.initial_states), "state_idx out of bounds."
            state = self.initial_states[state_idx]
        else:
            # Sample from available pool
            if self.sample_mode in ("sequential", "seeded"):
                idx = self._index_order[self._init_index]
                self._init_index = (self._init_index + 1) % len(self.initial_states)
                state = self.initial_states[idx]
            else:
                state = self.initial_states[self._rng.integers(len(self.initial_states))]

        # Set initial qpos (positions) and qvel (velocities)
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        qpos[:3] = state[:3]  # [x, theta1, theta2]
        qvel[:3] = state[3:]  # [vx, dtheta1, dtheta2]

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
            # Dense reward combining two objectives:
            # 1. Tip proximity to maximum height (50%)
            # 2. Cart proximity to center point (50%)

            # Get tip position and cart position
            _, _, y = self.data.site_xpos[0]  # Tip vertical position
            cart_x = self.data.qpos[0]  # Cart horizontal position

            # Calculate height difference component
            tip_height_diff = abs(y - self.max_tip_y)
            alpha_height = 5.0  # Controls sharpness of height reward drop-off
            height_reward = np.exp(-alpha_height * tip_height_diff)

            # Calculate cart center component
            alpha_cart = 1.0  # Controls sharpness of cart position reward drop-off
            cart_reward = np.exp(-alpha_cart * abs(cart_x))

            # Combine rewards with equal weights (50% each)
            reward = 0.9 * height_reward + 0.1 * cart_reward

            reward_info = {
                "combined_dense_reward": reward,
                "height_reward": height_reward,
                "cart_reward": cart_reward,
                "tip_height": y,
                "max_height": self.max_tip_y,
                "height_diff": tip_height_diff,
                "cart_x": cart_x,
                "alpha_height": alpha_height,
                "alpha_cart": alpha_cart,
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
        """Apply user action + linear spring torques, then step the simulator."""

        # ------------------------------------------------------------------
        # 1) copy agent action
        self.data.ctrl[:] = action

        # ------------------------------------------------------------------
        # 2) spring torques  τ = -kθ
        theta1 = float(self.data.qpos[1])
        theta2 = float(self.data.qpos[2])
        tau1 = -self.hinge1_stiffness * theta1
        tau2 = -self.hinge2_stiffness * theta2

        # ------------------------------------------------------------------
        # 3) clear previous applied forces
        self.data.qfrc_applied.fill(0.0)

        # get DOF indices via mj_name2id   (works for MuJoCo >=2.1, 3.x)
        JOINT_OBJ = getattr(mujoco, "mjtObj_JOINT", 2)
        jnt1 = mujoco.mj_name2id(self.model, JOINT_OBJ, "hinge")
        jnt2 = mujoco.mj_name2id(self.model, JOINT_OBJ, "hinge2")
        dof1 = int(self.model.jnt_dofadr[jnt1])
        dof2 = int(self.model.jnt_dofadr[jnt2])

        # inject spring torques
        self.data.qfrc_applied[dof1] = tau1
        self.data.qfrc_applied[dof2] = tau2

        # ------------------------------------------------------------------
        # 4) simulate
        self.do_simulation(self.data.ctrl, self.frame_skip)

        # ------------------------------------------------------------------
        # 5) obs / reward / done
        x, _, y = self.data.site_xpos[0]
        obs = self._get_obs()
        done = bool(y <= self.fail_threshold)
        reward, info = self._get_rew(x, y, done)

        if self.render_mode == "human":
            self.render()

        return obs, reward, done, False, info
