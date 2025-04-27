from gymnasium.envs.mujoco.inverted_pendulum_v5 import InvertedPendulumEnv
import tempfile
import xml.etree.ElementTree as ET
import os


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
    def __init__(
        self,
        length: float = 0.6,
        pole_density: float = 1000.0,
        cart_density: float = 1000.0,
        xml_file: str = "./assets/inverted_pendulum.xml",
        **kwargs,
    ):
        modified_xml_file = modify_inverted_pendulum_xml(
            xml_file, length, pole_density, cart_density
        )

        super().__init__(xml_file=modified_xml_file, **kwargs)
        self._temp_xml_path = modified_xml_file

    def close(self):
        super().close()
        if hasattr(self, "_temp_xml_path") and os.path.exists(self._temp_xml_path):
            os.remove(self._temp_xml_path)
