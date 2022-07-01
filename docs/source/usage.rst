.. note::

   This is just a template! Page under contruction!

Usage
=====

.. _installation:

Installation
------------

To use src.lumache, first install it using pip:

.. code-block:: console

   (.venv) $ pip install src.lumache

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``src.lumache.get_random_ingredients()`` function:

.. autofunction:: src.lumache.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`src.lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: src.lumache.InvalidKindError

For example:

>>> import src.lumache
>>> src.lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']


Math formulas
-------------

.. math:: e^{i\pi} + 1 = 0
   :label: euler

Euler's identity, equation :math:numref:`euler`, was elected one of the
most beautiful mathematical formulas.


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


Others
------
Check out the :doc:`usage` section for further information, including
how to :ref:`installation` the project.
