import yaml
import numpy as onp
import jax
import jax.numpy as np
import argparse
import sys
import numpy as onp
import matplotlib.pyplot as plt
from jax.config import config
import os


# Set numpy printing format
onp.random.seed(0)
onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True)
onp.set_printoptions(precision=10)

# np.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True)
# np.set_printoptions(precision=5)


# yaml_filepath = os.path.realpath(os.path.join(os.path.dirname(__file__), '../modules/phase_field/pre-processing/yaml/default.yaml'))
# yaml_filepath = f'modules/phase_field/pre-processing/yaml/default.yaml'
yaml_filepath = os.path.realpath(os.path.join(os.path.dirname(__file__), 'pre-processing/yaml/default.yaml'))


# TODO: do some basic checks for the validity of parameters
with open(yaml_filepath) as f:
    args = yaml.load(f, Loader=yaml.FullLoader)
    print(f"YAML parameters:")
    # TODO: These are just default parameters
    print(yaml.dump(args, default_flow_style=False))
    print(f"These are default parameters")

args['root_path'] = os.path.dirname(__file__)

data_folders = ['mp4', 'neper', 'numpy', 'pdf', 'png', 'vtk', 'txt']

for data_folder in data_folders:
	os.makedirs(os.path.join(args['root_path'], 'data', data_folder), exist_ok=True)
