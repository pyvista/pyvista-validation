pyvista-validation
==================

Validate and standardize array-like input.

These are the input validation functions developed for `PyVista
<https://github.com/pyvista/pyvista>`_, extracted into a standalone package so
any project can use them. Input validation methods can be used for checking
and/or validating that a variable has the correct type and/or value for use by
an algorithm. They are useful when writing custom Python methods that accept
flexible array-like input, wrapping ``VTK``, and/or when `Contributing to
PyVista <https://github.com/pyvista/pyvista/blob/main/CONTRIBUTING.rst>`_.

.. warning::

   The API of this package is unstable and likely to change between minor
   versions (for example ``0.1.0`` to ``0.2.0``). Pin the exact version you
   depend on, for example ``pyvista-validation==0.1.0``.

Installation
------------

.. code-block:: bash

   pip install pyvista-validation

NumPy is the only required dependency. ``VTK`` and ``SciPy`` are only needed
to validate their own object types, so they ship as extras:

.. code-block:: bash

   pip install pyvista-validation[vtk]    # accept vtkMatrix3x3, vtkMatrix4x4, vtkTransform
   pip install pyvista-validation[scipy]  # accept scipy.spatial.transform.Rotation
   pip install pyvista-validation[all]    # both

Neither is imported unless one of their objects is actually passed in.

Common use cases
----------------

Validate a 3-element vector:
    Use :func:`~pyvista_validation.validate.validate_array3`
Validate an Nx3 point or vector array:
    Use :func:`~pyvista_validation.validate.validate_arrayNx3`
Validate point or cell IDs:
    Use :func:`~pyvista_validation.validate.validate_arrayN_unsigned`
Validate a transformation matrix:
    Use :func:`~pyvista_validation.validate.validate_transform4x4`

.. toctree::
   :maxdepth: 2

   api
