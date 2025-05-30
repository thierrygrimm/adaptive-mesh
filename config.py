import json
from itertools import product

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

# Example default config structure
DEFAULT_CONFIG = {
    'node_ids': [],
    'frequency': 915e6,
    'port_mappings': {},
    'lora_params': {
        'spreading_factor': 7,
        'bandwidth': 125e3,
        'coding_rate': 5
    },
    'rl_hyperparameters': {
        'epsilon': 0.1,
        'alpha': 0.5,
        'gamma': 0.9
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
