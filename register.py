from gymnasium.envs.registration import register


register(
    id="CustomInvertedPendulum",
    entry_point="custom_inverted_pendulum:CustomInvertedPendulum",
    kwargs={
        "render_mode": None,
        "length": 0.6,
        "pole_density": 1000.0,
        "cart_density": 1000.0,
        "xml_file": "./assets/inverted_pendulum.xml",
        "reset_noise_scale": 0.01,
    },
    max_episode_steps=500,
)