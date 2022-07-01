.. note::

   This is just a template!

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
