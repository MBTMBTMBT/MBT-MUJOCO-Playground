import os
from typing import Optional, List, Tuple

import mujoco
from gymnasium.envs.mujoco.humanoidstandup_v5 import HumanoidStandupEnv
import tempfile
import xml.etree.ElementTree as ET
import numpy as np

from custom_mujoco import get_asset_path


def modify_humanoid_xml(
    xml_path: str,
    floor_friction_scale: float,
    top_heaviness: float,
) -> str:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Modify floor friction by scaling existing values
    floor = root.find(".//geom[@name='floor']")
    if floor is not None and floor.get("friction"):
        fric = [float(x) for x in floor.get("friction").split()]
        fric_scaled = [x * floor_friction_scale for x in fric]
        floor.set("friction", " ".join(f"{x:.6f}" for x in fric_scaled))

    # Scale only the top components
    top_parts = ["head", "uwaist", "torso1"]
    for geom in root.findall(".//geom"):
        name = geom.get("name", "")
        if any(part in name for part in top_parts):
            if geom.get("density") is not None:
                old = float(geom.get("density"))
                geom.set("density", str(old * top_heaviness))
            elif geom.get("mass") is not None:
                old = float(geom.get("mass"))
                geom.set("mass", str(old * top_heaviness))

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xml", mode="w")
    tree.write(tmp_file.name)
    tmp_file.close()
    return tmp_file.name


class CustomHumanoidStandup(HumanoidStandupEnv):
    def __init__(
        self,
        xml_file: str = get_asset_path("humanoidstandup.xml"),
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
        """
        Customizable HumanoidStandup environment.

        Args:
            xml_file (str): Path to the base MuJoCo XML model.
            top_heaviness (float): Scale factor for upper-body components ("head", "uwaist", "torso1").
                                   Values >1 make standing harder due to higher center of mass.
            floor_friction_scale (float): Multiplier applied to the floor friction values.
                                          Values <1 make the ground more slippery.
            dense_reward (bool): If True, uses head-height ratio reward. If False, uses original uph_cost reward.
            initial_states (np.ndarray or None): Predefined initial states of shape [n, nq + nv].
            initial_state_idxs (list of int or None): Optional subset of indices into `initial_states` to restrict sampling.
            init_dist (str): Distribution type for sampling initial states ("uniform" or "gaussian").
            n_rand_initial_states (int): Number of random initial states to sample.
            init_ranges (list of tuple): Ranges [(low, high), ...] for each state dimension, covering all position and velocity components of the humanoid. For this environment, the state has nq=78 position dimensions and nv=67 velocity dimensions, for a total of 145 dimensions. Each tuple specifies the allowed range for random sampling of the corresponding state variable. Defaults to [(-0.02, 0.02)] * 145 if None.
            init_mode (str): State sampling mode: "random", "sequential", or "seeded".
            seed (int or None): Random seed for reproducibility.
            **kwargs: Additional keyword arguments forwarded to HumanoidStandupEnv.
        """
        self._rng = np.random.default_rng(seed)
        self.sample_mode = init_mode
        self._init_index = 0
        self.dense_reward = dense_reward

        modified_xml = modify_humanoid_xml(
            xml_file,
            floor_friction_scale=floor_friction_scale,
            top_heaviness=top_heaviness,
        )
        self._temp_xml_path = modified_xml

        super().__init__(xml_file=modified_xml, **kwargs)
        self._compute_reference_heights()

        self.nq = self.model.nq
        self.nv = self.model.nv
        self._state_dim = self.nq + self.nv

        self.init_ranges = init_ranges or [(-0.02, 0.02)] * self._state_dim

        # Step 1: Prepare full pool of initial states
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
                raise ValueError("Unsupported init_dist: choose 'uniform' or 'gaussian'")

        # Step 2: Restrict to indexed subset if provided
        if initial_state_idxs is not None:
            self.initial_states = all_states[np.array(initial_state_idxs)]
        else:
            self.initial_states = all_states

        # Step 3: Setup sampling order
        if init_mode == "seeded":
            self._index_order = self._rng.permutation(len(self.initial_states))
        elif init_mode == "sequential":
            self._index_order = np.arange(len(self.initial_states))
        else:
            self._index_order = None

    def reset_model(self, state: Optional[np.ndarray] = None, state_idx: Optional[int] = None):
        """
        Reset the model to a sampled or explicitly given initial state.

        Args:
            state (np.ndarray, optional): Full (nq + nv)-dim state vector to override automatic sampling.
            state_idx (int, optional): Index into `initial_states` to override sampling.

        Returns:
            observation (np.ndarray): The observation after reset.
        """
        if state is not None:
            assert state.shape == (self._state_dim,), "State must be shape (nq + nv,)"
        elif state_idx is not None:
            assert 0 <= state_idx < len(self.initial_states), "Invalid state_idx."
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
        Overrides the base reset method to support `state` or `state_idx` via options.

        Args:
            seed (int, optional): RNG seed for this reset.
            options (dict, optional): May include:
                - 'state' (np.ndarray): Full state vector to use.
                - 'state_idx' (int): Index into `initial_states`.

        Returns:
            observation (np.ndarray): Initial observation.
            info (dict): Environment reset info.
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

    def _compute_reference_heights(self):
        """
        Calculate reference heights for proper reward normalization:
        - Standing height (maximum) using the "stand" keyframe
        - Initial height (minimum) using current lying pose

        This allows proper scaling of rewards between 0 (lying) and 1 (standing)
        """
        # Get head geom ID for position tracking
        try:
            self._head_gid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, "head"
            )

            # 1. Calculate standing height using the "stand" keyframe
            stand_key_id = -1
            for i in range(self.model.nkey):
                key_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY, i)
                if key_name == "stand":
                    stand_key_id = i
                    break

            if stand_key_id >= 0:
                # Get the "stand" keyframe position
                stand_qpos = self.model.key_qpos[stand_key_id].copy()

                # Create temporary copy of data for simulation
                tmp_data = mujoco.MjData(self.model)
                tmp_data.qpos[:] = stand_qpos
                mujoco.mj_forward(self.model, tmp_data)

                # Store max head height when standing
                self._head_z_max = float(tmp_data.geom_xpos[self._head_gid][2])
            else:
                # Fallback: estimate max height as 1.7m if keyframe not found
                self._head_z_max = 1.7

            # 2. Calculate minimum height from initial lying position
            self._head_z_min = float(self.data.geom_xpos[self._head_gid][2])

            # Print diagnostic info for debugging
            # print(f"Head height reference: min={self._head_z_min:.3f}m, max={self._head_z_max:.3f}m")

        except Exception as e:
            # Fallback if anything goes wrong
            print(f"Warning: Could not compute reference heights: {e}")
            self._head_gid = -1
            self._head_z_min = 0.1  # Approximate height when lying
            self._head_z_max = 1.7  # Approximate height when standing

    def close(self):
        super().close()
        if hasattr(self, "_temp_xml_path") and os.path.exists(self._temp_xml_path):
            os.remove(self._temp_xml_path)

    def _get_rew(self, pos_after: float, action):
        if getattr(self, "dense_reward", False):
            # Get the head's z position - handle API differences between mujoco versions
            try:
                # Modern MuJoCo Python API (≥2.3)
                if self._head_gid >= 0:
                    z_head = self.data.geom_xpos[self._head_gid][2]
                else:
                    # Fallback - try to find head from named bodies
                    torso_id = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
                    )
                    z_head = self.data.xpos[torso_id][2]
            except (ImportError, AttributeError):
                # Legacy mujoco-py API
                head_id = self.model.body_name2id("head")
                z_head = self.data.body_xpos[head_id][2]

            # Calculate normalized reward based on head height (0.0 = lying, 1.0 = standing)
            # Allow rewards > 1.0 if head goes higher than reference standing height
            head_ratio = (z_head - self._head_z_min) / (
                self._head_z_max - self._head_z_min
            )

            # Scale factor to make the main reward comparable to cost terms
            # Typical control/impact costs are ~0.01-0.1, so we scale main reward down
            reward_scale = 0.1
            main_reward = head_ratio * reward_scale

            reward_info = {
                "reward_head_ratio": head_ratio,
                "reward_scaled": main_reward,
                "head_height": z_head,
                "min_height": self._head_z_min,
                "max_height": self._head_z_max,
            }
        else:
            # Original reward: torso lifting rate
            uph_cost = (pos_after - 0) / self.model.opt.timestep
            main_reward = uph_cost + 1  # include the constant bonus
            reward_info = {
                "reward_linup": uph_cost,
            }

        # Control cost (same in both modes)
        # Adjust these weights to balance with main reward
        ctrl_cost_weight = (
            0.01 * self._ctrl_cost_weight
            if self.dense_reward
            else self._ctrl_cost_weight
        )
        impact_cost_weight = (
            0.01 * self._impact_cost_weight
            if self.dense_reward
            else self._impact_cost_weight
        )

        quad_ctrl_cost = ctrl_cost_weight * np.square(self.data.ctrl).sum()

        # Impact cost (same in both modes)
        quad_impact_cost = impact_cost_weight * np.square(self.data.cfrc_ext).sum()
        min_impact_cost, max_impact_cost = self._impact_cost_range
        if self.dense_reward:
            min_impact_cost, max_impact_cost = (
                0.01 * min_impact_cost,
                0.01 * max_impact_cost,
            )
        quad_impact_cost = np.clip(quad_impact_cost, min_impact_cost, max_impact_cost)

        # Total reward
        reward = main_reward - quad_ctrl_cost - quad_impact_cost
        if self.dense_reward:
            # reward *= 100
            pass

        # Add cost info to reward_info
        reward_info.update(
            {
                "reward_quadctrl": -quad_ctrl_cost,
                "reward_impact": -quad_impact_cost,
            }
        )

        return reward, reward_info
