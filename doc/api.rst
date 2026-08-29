API Reference
=============

A ``check`` function performs a simple validation on a single input variable,
raises an error if the check fails, and does not modify input or return
anything. A ``validate`` function accepts many different input types or values,
applies optional constraints, and standardizes the output as a single
representation with known properties.

Every function is importable directly from the top-level package, for example
``from pyvista_validation import validate_array``. The recommended import binds
the whole package to a short name:

.. code-block:: python

   from pyvista_validation import validation

   validation.validate_array3([1, 2, 3])

.. currentmodule:: pyvista_validation

.. autosummary::
   :toctree: _autosummary

   check
   validate
