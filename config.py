import json
import os
from itertools import product
import copy

def merge_config(base, override):
    """Recursively merge override dict into base dict.
    """Recursively merge override dict into base dict."""
    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            merge_config(base[k], v)
        else:
            base[k] = v
    return base

def load_config(path, user_specified=False):
    """Load configuration from file and merge with default config.
    
    Args:
        path: Path to configuration file
        user_specified: Whether the path was user-specified (default: False)
        
    Returns:
        Merged configuration dictionary
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if os.path.exists(path):
        with open(path, 'r') as f:
            file_cfg = json.load(f)
            merge_config(cfg, file_cfg)
    else:
        if user_specified:
            print(f"[WARN] Config file '{path}' not found. Using DEFAULT_CONFIG.")
    return cfg

# Example default config structure
DEFAULT_CONFIG = {
    'node_ids': [],
    'frequency': 868e6,
    'port_mappings': {},
    'lora_params': {
        'use_preset': False,
        'spreading_factor': 9,
        'bandwidth': 250,
        'coding_rate': 5
    },
    'rl_hyperparameters': {
        'epsilon': 0.3,  # Increased epsilon for more exploration
        'alpha': 1.0,  # Increased alpha for more exploration in LinUCB
        'gamma': 0.9,
        'lambda_reg': 0.1,  # Regularization parameter
        'poly_degree': 2  # Polynomial feature degree
    }
}

# Define possible values for each parameter
SPREADING_FACTORS = list(range(7, 13))  # 7 to 12 inclusive
BANDWIDTHS = [125, 250, 500]  # in kHz

# Cross product to generate all possible arms (no coding rate)
RL_ARMS = [
    {'spreading_factor': sf, 'bandwidth': bw}
    for sf, bw in product(SPREADING_FACTORS, BANDWIDTHS)
]
