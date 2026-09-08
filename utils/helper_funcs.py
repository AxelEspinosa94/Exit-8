import yaml


def load_config(category: str) -> dict:
    path = f"configs/{category}.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)
