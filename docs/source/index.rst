.. jax-am documentation master file, created by
   sphinx-quickstart on Thu Jun 30 11:08:35 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to JAX-AM's documentation!
==================================

**JAX-AM** is a Python library for numerical simulations in additive manufacturing.

Check out the :doc:`usage` section for further information, including
how to :ref:`installation` the project.

.. note::

   This project is under active development.

Contents
--------


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   myfile
   usage
   debug


Python codes
------------

To retrieve a list of random ingredients,
you can use the ``src.lumache.get_random_ingredients()`` function:

.. py:function:: src.lumache.get_random_ingredients(kind=None)
   :noindex:

   Return a list of random ingredients as strings.

   :param kind: Optional "kind" of ingredients.
   :type kind: list[str] or None
   :return: The ingredients list.
   :rtype: list[str]


Math formulas
-------------

.. math:: e^{i\pi} + 1 = 0
   :label: euler

Euler's identity, equation :math:numref:`euler`, was elected one of the
most beautiful mathematical formulas.




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
