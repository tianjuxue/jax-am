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


yaml_filepath = os.path.realpath(os.path.join(os.path.dirname(__file__), '../pre-processing/yaml/default.yaml'))

with open(yaml_filepath) as f:
    args = yaml.load(f, Loader=yaml.FullLoader)
    # print(args)
