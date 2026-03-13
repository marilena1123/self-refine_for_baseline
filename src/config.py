import yaml
import os

_config = None


def load_config(config_path=None):
    global _config
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
        )
    with open(config_path) as f:
        _config = yaml.safe_load(f)
    return _config


def get_config():
    global _config
    if _config is None:
        load_config()
    return _config
