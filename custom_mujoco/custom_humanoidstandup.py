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
            init_dist (str): Distribution type for sampling initial states ("uniform" or "gaussian").
            n_rand_initial_states (int): Number of random initial states to sample.
            init_ranges (list of tuple): Ranges [(low, high), ...] for each initial state dimension.
            init_mode (str): State sampling mode: "random", "sequential", or "seeded".
            seed (int or None): Random seed for reproducibility.
            **kwargs: Additional keyword arguments forwarded to HumanoidStandupEnv.
        """
        # Initialize RNG and reset strategy
        self._rng = np.random.default_rng(seed)
        self.sample_mode = init_mode
        self._init_index = 0

        # Store reward mode
        self.dense_reward = dense_reward

        # Create modified XML with floor friction and head mass scaling
        modified_xml = modify_humanoid_xml(
            xml_file,
            floor_friction_scale=floor_friction_scale,
            top_heaviness=top_heaviness,
        )
        self._temp_xml_path = modified_xml

        # Initialize parent environment with custom XML
        super().__init__(xml_file=modified_xml, **kwargs)

        # Record model state dimensions
        self.nq = self.model.nq
        self.nv = self.model.nv
        self._state_dim = self.nq + self.nv

        # Define initial state sampling ranges
        self.init_ranges = init_ranges or [(-0.02, 0.02)] * self._state_dim

        if initial_states is not None:
            self.initial_states = np.array(initial_states)
        else:
            lows = np.array([r[0] for r in self.init_ranges])
            highs = np.array([r[1] for r in self.init_ranges])
            if init_dist == "uniform":
                self.initial_states = self._rng.uniform(
                    low=lows, high=highs, size=(n_rand_initial_states, self._state_dim)
                )
            elif init_dist == "gaussian":
                self.initial_states = np.clip(
                    self._rng.normal(
                        loc=0.0, scale=0.5 * (highs - lows), size=(n_rand_initial_states, self._state_dim)
                    ),
                    lows,
                    highs,
                )
            else:
                raise ValueError("Unsupported init_dist: choose 'uniform' or 'gaussian'")

        if init_mode == "seeded":
            self._index_order = self._rng.permutation(len(self.initial_states))
        elif init_mode == "sequential":
            self._index_order = np.arange(len(self.initial_states))
        else:
            self._index_order = None

    def reset_model(self):
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

    def close(self):
        super().close()
        if hasattr(self, "_temp_xml_path") and os.path.exists(self._temp_xml_path):
            os.remove(self._temp_xml_path)

    def _get_rew(self, pos_after: float, action):
        if getattr(self, "dense_reward", False):
            # Get the head's z position - handle API differences between mujoco versions
            try:
                # Modern MuJoCo Python API (≥2.3)
                head_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "head")
                if head_id >= 0:
                    z_head = self.data.geom_xpos[head_id][2]
                else:
                    # Fallback - try to find head from named bodies
                    body_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                                  for i in range(self.model.nbody)]
                    for i, name in enumerate(body_names):
                        if name and 'head' in name.lower():
                            z_head = self.data.xpos[i][2]
                            break
                    else:
                        # If we couldn't find anything with 'head', use torso height as proxy
                        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
                        z_head = self.data.xpos[torso_id][2]
            except (ImportError, AttributeError):
                # Legacy mujoco-py API
                head_id = self.model.body_name2id("head")
                z_head = self.data.body_xpos[head_id][2]
            # Calculate reward based on head height
            z_ref = self.init_qpos[2]
            head_ratio = z_head / z_ref
            main_reward = head_ratio
            reward_info = {
                "reward_head_ratio": head_ratio,
            }
        else:
            # Original reward: torso lifting rate
            uph_cost = (pos_after - 0) / self.model.opt.timestep
            main_reward = uph_cost + 1  # include the constant bonus
            reward_info = {
                "reward_linup": uph_cost,
            }

        # Control cost (same in both modes)
        quad_ctrl_cost = self._ctrl_cost_weight * np.square(self.data.ctrl).sum()

        # Impact cost (same in both modes)
        quad_impact_cost = self._impact_cost_weight * np.square(self.data.cfrc_ext).sum()
        min_impact_cost, max_impact_cost = self._impact_cost_range
        quad_impact_cost = np.clip(quad_impact_cost, min_impact_cost, max_impact_cost)

        # Total reward
        reward = main_reward - quad_ctrl_cost - quad_impact_cost

        # Add cost info to reward_info
        reward_info.update({
            "reward_quadctrl": -quad_ctrl_cost,
            "reward_impact": -quad_impact_cost,
        })

        return reward, reward_info
