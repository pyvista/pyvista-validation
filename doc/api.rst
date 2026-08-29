API Reference
=============

A ``check`` function performs a simple validation on a single input variable,
raises an error if the check fails, and does not modify input or return
anything. A ``validate`` function accepts many different input types or values,
applies optional constraints, and standardizes the output as a single
representation with known properties.

Every function is importable directly from the top-level package, for example
``from input_validation import validate_array``.

.. currentmodule:: input_validation

.. autosummary::
   :toctree: _autosummary

   check
   validate
