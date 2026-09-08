"""Build the C accelerator, unless ``PYVISTA_VALIDATION_NO_EXTENSION`` asks for pure Python."""

from __future__ import annotations

import os
import sys

from setuptools import Extension
from setuptools import setup


def wanted(value: str | None, /) -> bool:
    """Return whether an environment value asks for a build without the accelerator."""
    return value is not None and value.strip().lower() not in {'', '0', 'false', 'off', 'no'}


if wanted(os.environ.get('PYVISTA_VALIDATION_NO_EXTENSION')):
    setup()
else:
    import numpy as np

    setup(
        ext_modules=[
            Extension(
                'pyvista_validation._fast',
                sources=['src/fast/module.c'],
                depends=[
                    'src/fast/fast.h',
                    'src/fast/array.c',
                    'src/fast/values.c',
                    'src/fast/checks.c',
                    'src/fast/validate.c',
                ],
                include_dirs=[np.get_include()],
                define_macros=[
                    # One wheel per platform serves every supported Python.
                    ('Py_LIMITED_API', '0x030A0000'),
                    ('NPY_NO_DEPRECATED_API', 'NPY_2_0_API_VERSION'),
                    # Built against NumPy 2, loadable on every NumPy the package supports.
                    ('NPY_TARGET_VERSION', 'NPY_1_21_API_VERSION'),
                ],
                # No fused multiply-add: the axes and rotation checks must round like NumPy
                extra_compile_args=['/O2']
                if sys.platform == 'win32'
                else ['-O3', '-ffp-contract=off'],
                py_limited_api=True,
                optional=True,
            )
        ],
        options={'bdist_wheel': {'py_limited_api': 'cp310'}},
    )
