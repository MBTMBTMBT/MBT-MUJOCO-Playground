import os
from typing import Optional, List, Tuple

import mujoco
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv
import tempfile
import xml.etree.ElementTree as ET
import numpy as np

from custom_mujoco import get_asset_path


def modify_humanoid_xml(
    xml_path: str,
    floor_friction_scale: float,
    top_heaviness: float,
) -> str:
    """
    Modify the humanoid XML to scale floor friction and top-body mass.

    - Floor friction is scaled by multiplying the `friction` attribute of the geom named 'floor'.
    - Top-body parts ("head", "uwaist", "torso1") have their density or mass scaled.

    Args:
        xml_path (str): Path to original XML file.
        floor_friction_scale (float): Multiplier for floor friction.
        top_heaviness (float): Multiplier for head/uwaist/torso1 mass or density.

    Returns:
        str: Path to modified temporary XML file.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Step 1: Scale floor friction
    floor = root.find(".//geom[@name='floor']")
    if floor is not None and floor.get("friction"):
        friction_values = [float(x) for x in floor.get("friction").split()]
        scaled = [x * floor_friction_scale for x in friction_values]
        floor.set("friction", " ".join(f"{x:.6f}" for x in scaled))

    # Step 2: Scale top-heavy parts (mass/density)
    top_parts = ["head", "uwaist", "torso1"]
    for geom in root.findall(".//geom"):
        name = geom.get("name", "")
        if any(part in name for part in top_parts):
            if geom.get("density") is not None:
                old_density = float(geom.get("density"))
                geom.set("density", str(old_density * top_heaviness))
            elif geom.get("mass") is not None:
                old_mass = float(geom.get("mass"))
                geom.set("mass", str(old_mass * top_heaviness))

    # Save to temporary file
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xml", mode="w")
    tree.write(tmp_file.name)
    tmp_file.close()
    return tmp_file.name


class CustomHumanoid(HumanoidEnv):
    """
    A configurable extension of the standard Humanoid (locomotion) environment.

    Features:
    - Modifies MuJoCo XML on the fly to change top-body mass and floor friction.
    - Supports dense or sparse reward modes:
        - Sparse: Original reward with velocity + alive bonus.
        - Dense: Reward is based on forward displacement (distance traveled).
    - Flexible initial state configuration:
        - Can accept fixed list of initial states.
        - Can generate random initial states (uniform or Gaussian).
        - Can restrict sampling to a subset of state indices.
        - Supports 'random', 'sequential', or 'seeded' sampling modes.
        - Supports external reset with `state_idx`.

    Args:
        xml_file (str): Path to humanoid MuJoCo XML file.
        top_heaviness (float): Scaling factor for top-body parts (head, torso1, uwaist).
        floor_friction_scale (float): Multiplier to apply to floor friction.
        dense_reward (bool): Whether to use dense reward based on forward displacement.
        initial_states (np.ndarray or None): If given, use this pool of initial states.
        initial_state_idxs (list[int] or None): Optional subset of indices into `initial_states`.
        init_dist (str): 'uniform' or 'gaussian', used if `initial_states` not provided.
        n_rand_initial_states (int): Number of random states to generate.
        init_ranges (list[tuple[float, float]]): Per-dimension ranges for state sampling.
        init_mode (str): One of 'random', 'sequential', 'seeded'.
        seed (int or None): RNG seed for reproducibility.
        **kwargs: Additional arguments forwarded to base HumanoidEnv.
    """

    def __init__(
        self,
        xml_file: str = get_asset_path("humanoid.xml"),
        top_heaviness: float = 1.0,
        floor_friction_scale: float = 1.0,
        dense_reward: bool = False,
        initial_states: Optional[np.ndarray] = None,
        initial_state_idxs: Optional[List[int]] = None,
        init_dist: str = "uniform",
        n_rand_initial_states: int = 100,
        init_ranges: Optional[List[Tuple[float, float]]] = None,
        init_mode: str = "random",
        seed: Optional[int] = None,
        **kwargs,
    ):
        self._rng = np.random.default_rng(seed)
        self.sample_mode = init_mode
        self._init_index = 0
        self.dense_reward = dense_reward

        # Modify MuJoCo XML to inject friction/mass changes
        modified_xml = modify_humanoid_xml(xml_file, floor_friction_scale, top_heaviness)
        self._temp_xml_path = modified_xml

        # Load the MuJoCo environment
        super().__init__(xml_file=modified_xml, **kwargs)

        # Extract state dimensions
        self.nq = self.model.nq
        self.nv = self.model.nv
        self._state_dim = self.nq + self.nv

        # Initialize sampling range if not provided
        self.init_ranges = init_ranges or [(-0.02, 0.02)] * self._state_dim

        # Step 1: Create full initial state pool
        if initial_states is not None:
            all_states = np.array(initial_states)
        else:
            lows = np.array([r[0] for r in self.init_ranges])
            highs = np.array([r[1] for r in self.init_ranges])
            if init_dist == "uniform":
                all_states = self._rng.uniform(low=lows, high=highs, size=(n_rand_initial_states, self._state_dim))
            elif init_dist == "gaussian":
                all_states = np.clip(
                    self._rng.normal(loc=0.0, scale=0.5 * (highs - lows), size=(n_rand_initial_states, self._state_dim)),
                    lows, highs,
                )
            else:
                raise ValueError("Unsupported init_dist: must be 'uniform' or 'gaussian'")

        # Step 2: Apply state index filtering
        if initial_state_idxs is not None:
            self.initial_states = all_states[np.array(initial_state_idxs)]
        else:
            self.initial_states = all_states

        # Step 3: Set up sampling order
        if init_mode == "seeded":
            self._index_order = self._rng.permutation(len(self.initial_states))
        elif init_mode == "sequential":
            self._index_order = np.arange(len(self.initial_states))
        else:
            self._index_order = None  # random

    def reset_model(self, state: Optional[np.ndarray] = None, state_idx: Optional[int] = None):
        """
        Reset model to a specific or sampled initial state.

        Args:
            state: Optional full state vector (qpos + qvel).
            state_idx: Optional index into initial_states.

        Returns:
            observation (np.ndarray): Initial observation.
        """
        if state is not None:
            assert state.shape == (self._state_dim,)
        elif state_idx is not None:
            state = self.initial_states[state_idx]
        else:
            if self.sample_mode in ("sequential", "seeded"):
                idx = self._index_order[self._init_index]
                self._init_index = (self._init_index + 1) % len(self.initial_states)
                state = self.initial_states[idx]
            else:
                state = self.initial_states[self._rng.integers(len(self.initial_states))]

        qpos = self.init_qpos + state[:self.nq]
        qvel = self.init_qvel + state[self.nq:]
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """
        Override reset to support passing 'state' or 'state_idx' via options dict.
        """
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        state = options.get("state") if options else None
        state_idx = options.get("state_idx") if options else None
        obs = self.reset_model(state=state, state_idx=state_idx)
        info = self._get_reset_info()

        if self.render_mode == "human":
            self.render()
        return obs, info

    def _get_rew(self, x_pos_before: float, x_pos_after: float, action: np.ndarray) -> Tuple[float, dict]:
        """
        Compute reward using dense or sparse formulation.
        - Dense: Use forward velocity.
        - Sparse: Use original alive_bonus + linear velocity.
        """
        dt = self.model.opt.timestep

        if self.dense_reward:
            forward_reward = (x_pos_after - x_pos_before) / dt
            main_reward = forward_reward
        else:
            forward_reward = (x_pos_after - x_pos_before) / dt
            main_reward = forward_reward + self._alive_bonus

        ctrl_cost = self._ctrl_cost_weight * np.square(action).sum()
        impact_cost = np.clip(
            self._impact_cost_weight * np.square(self.data.cfrc_ext).sum(),
            *self._impact_cost_range,
        )
        total_reward = main_reward - ctrl_cost - impact_cost

        return total_reward, {
            "reward_forward": forward_reward,
            "reward_ctrl_cost": -ctrl_cost,
            "reward_impact_cost": -impact_cost,
            "reward_total": total_reward,
        }

    def step(self, action: np.ndarray):
        """
        Execute one step of simulation and compute reward.
        """
        x_before = self.get_body_com("torso")[0]
        self.do_simulation(action, self.frame_skip)
        x_after = self.get_body_com("torso")[0]

        obs = self._get_obs()
        reward, reward_info = self._get_rew(x_before, x_after, action)
        terminated = self._is_done()
        info = {"reward_info": reward_info}

        return obs, reward, terminated, False, info

    def close(self):
        super().close()
        if hasattr(self, "_temp_xml_path") and os.path.exists(self._temp_xml_path):
            os.remove(self._temp_xml_path)
