"""Load the central YAML config (configs/config.yaml) used by every script."""
import yaml


def load_config(path="configs/config.yaml"):
    """Read the YAML config at `path` and return it as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config file {path} is empty or not a mapping")
    return cfg
