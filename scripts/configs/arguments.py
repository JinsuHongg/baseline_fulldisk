import yaml
import argparse
from types import SimpleNamespace


def get_args(
    config_dir: str = "./baseline_fulldisk/scripts/configs/config_heliofm.yaml",
):
    parser = argparse.ArgumentParser(description="FullDiskModelTrainer")

    # General arguments
    parser.add_argument(
        "--config",
        type=str,
        default=config_dir,
        help="Path to YAML config file",
    )

    # Directory arguments
    parser.add_argument("--img_dir", type=str, default="/workspace/data/hmi_jpgs_512/")

    # Training arguments
    parser.add_argument("--model", type=str, default="Spectformer")
    parser.add_argument("--train_set", type=float, nargs="+", default=[1, 2])
    parser.add_argument("--test_set", type=int, default=4)
    parser.add_argument("--file_tag", type=str, default="spectformer")
    # Optimization arguments
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_lr", type=float, default=1e-6)
    parser.add_argument("--div_factor", type=int, default=100)
    parser.add_argument("--class_weight", type=list, default=[1, 3])
    parser.add_argument("--weight_decay", type=float, nargs="+", default=[0, 0.0001, 0.001])
    args = parser.parse_args()
    
    # Load YAML if provided
    if config_dir:
        with open(config_dir, "r") as f:
            yaml_config = yaml.safe_load(f)

    config = dict_to_namespace(yaml_config)

    return args, config

def add_arguments_from_config(parser, config, prefix=""):
    """Add argparse arguments from nested YAML config."""
    for key, value in config.items():
        full_key = f"{prefix}{key}" if prefix else key

        if isinstance(value, dict):
            # Recursively handle nested dictionaries
            add_arguments_from_config(parser, value, prefix=f"{full_key}.")
        else:
            arg_type = infer_type(value)
            parser.add_argument(
                f"--{full_key}",
                type=arg_type,
                default=value,
                help=f"Override {full_key} (default: {value})",
            )

def dict_to_namespace(d):
    """Recursively convert dictionary to argparse.Namespace (dot notation support)."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) if isinstance(item, dict) else item for item in d]
    else:
        return d            

def infer_type(value):
    """Infer the correct type for argparse based on the YAML value."""
    if isinstance(value, bool):
        return str_to_bool
    elif isinstance(value, int):
        return int
    elif isinstance(value, float):
        return float
    elif isinstance(value, list):
        return parse_list
    else:
        return str
    
def str_to_bool(value):
    """Convert string to boolean for argparse."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "yes", "1"):
        return True
    elif value.lower() in ("false", "no", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

def parse_list(value):
    """Convert string to list for argparse."""
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        return [v.strip().strip("'\"") for v in value[1:-1].split(",")]
    return [value]