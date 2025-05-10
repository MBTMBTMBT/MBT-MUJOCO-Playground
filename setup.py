from setuptools import setup, find_packages

setup(
    name="custom_mujoco",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        "custom_mujoco.assets": ["*.xml"],
    },
)
