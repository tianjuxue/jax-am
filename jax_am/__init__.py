"""JAX-AM 

A GPU-based simulation toolbox for additive manufacturing.

Under GPL-3.0 license

"""

# TODO - public API to be spefified

from jax_am.phase_field import allen_cahn
from jax_am.phase_field import neper
from jax_am.phase_field import utils

import jax_am.cfd

__version__ = "0.0.1"

# __all__ = ()