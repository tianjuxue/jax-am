import unittest
# from src.fem.tests import __path__ # also works
from . import __path__

suite = unittest.TestLoader().discover(__path__[0])
print(f"suite = {suite}")
unittest.TextTestRunner(verbosity=2).run(suite)









