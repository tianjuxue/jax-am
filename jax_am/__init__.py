"""JAX-AM

Description
-----------
    A GPU-accelerated simulation toolbox for additive manufacturing.

License
-------
    Under the GPL-3.0 License:

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 3
    of the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

Contact
-------
    Tianju Xue (https://tianjuxue.github.io/)

"""

# TODO: public API management
# __all__ = ()

from .logger_setup import setup_logger
# LOGGING
logger = setup_logger(__name__)


__version__ = "0.0.3"
