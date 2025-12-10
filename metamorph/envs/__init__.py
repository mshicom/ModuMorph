from gymnasium.error import RegistrationError
from gymnasium.envs.registration import register


def _safe_register(env_id: str, entry_point: str):
    try:
        register(id=env_id, entry_point=entry_point, max_episode_steps=1000)
    except RegistrationError:
        # Ignore if already registered with gymnasium
        pass


# Support both versioned and unversioned IDs for backward compatibility.
env_specs = [
    ("Unimal-v0", "metamorph.envs.tasks.task:make_env"),
    ("Unimal", "metamorph.envs.tasks.task:make_env"),
    ("GeneralWalker2D-v0", "metamorph.envs.tasks.gen_walker_2d:make_env"),
    ("GeneralWalker2D", "metamorph.envs.tasks.gen_walker_2d:make_env"),
    ("Modular-v0", "modular.ModularEnv:ModularEnv"),
    ("Modular", "modular.ModularEnv:ModularEnv"),
]

for env_id, entry_point in env_specs:
    _safe_register(env_id, entry_point)

CUSTOM_ENVS = [
    "Unimal",
    "Unimal-v0",
    "GeneralWalker2D",
    "GeneralWalker2D-v0",
    "Modular",
    "Modular-v0",
]