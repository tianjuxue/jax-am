import numpy as onp
import jax
import jax.numpy as np
import unittest


if __name__ == '__main__':
    testsuite = unittest.TestLoader().discover('.')
    unittest.TextTestRunner(verbosity=1).run(testsuite)