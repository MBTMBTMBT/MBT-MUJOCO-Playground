import os
from importlib import resources as pkg_resources


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
