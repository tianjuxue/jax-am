.. note::

   This is just a template! Page under contruction!

Template
========

.. _installation:

Installation
------------

To use template.lumache, first install it using pip:

.. code-block:: console

   (.venv) $ pip install template.lumache

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``template.lumache.get_random_ingredients()`` function:

.. autofunction:: template.lumache.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`template.lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: template.lumache.InvalidKindError

For example:

>>> import template.lumache
>>> template.lumache.get_random_ingredients()
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
you can use the ``template.lumache.get_random_ingredients()`` function:

.. py:function:: template.lumache.get_random_ingredients(kind=None)
   :noindex:

   Return a list of random ingredients as strings.

   :param kind: Optional "kind" of ingredients.
   :type kind: list[str] or None
   :return: The ingredients list.
   :rtype: list[str]


Others
------
Check out the :doc:`template` section for further information, including
how to :ref:`installation` the project.
