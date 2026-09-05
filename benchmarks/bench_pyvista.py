"""Time PyVista operations that lean on the validation functions.

Needs PyVista installed against this checkout. Run once with
``PYVISTA_VALIDATION_ACCELERATE=0`` and ``--json before.json``, then again with
``--compare before.json`` to see what the C fast paths are worth to PyVista.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')

from bench import fmt
from bench import measure
import numpy as np
import pyvista as pv

POINTS = np.random.default_rng(0).random((100_000, 3))
SPHERE = pv.Sphere()
IMAGE = pv.ImageData(dimensions=(10, 10, 10))
TRANSFORM = pv.Transform().translate((1, 2, 3)).rotate_z(90)


def set_spacing() -> None:
    """Set the spacing of an image, which validates the vector."""
    IMAGE.spacing = (1.0, 2.0, 3.0)


CASES = {
    'Transform().translate((1, 2, 3))': (lambda: pv.Transform().translate((1, 2, 3)), (), {}),
    'Transform().translate().rotate_z().scale()': (
        lambda: pv.Transform().translate((1, 2, 3)).rotate_z(90).scale(2),
        (),
        {},
    ),
    'Transform.apply(points 1e5)': (TRANSFORM.apply, (POINTS,), {}),
    'ImageData(dimensions, spacing, origin)': (
        pv.ImageData,
        (),
        {'dimensions': (10, 10, 10), 'spacing': (1, 2, 3), 'origin': (0, 0, 0)},
    ),
    'ImageData.spacing = (1, 2, 3)': (set_spacing, (), {}),
    'Box(bounds)': (pv.Box, ((0, 1, 0, 1, 0, 1),), {}),
    'Line(pointa, pointb)': (pv.Line, ((0, 0, 0), (1, 1, 1)), {}),
    'Sphere(radius, center)': (pv.Sphere, (), {'radius': 0.5, 'center': (1, 2, 3)}),
    'PolyData(points 1e5)': (pv.PolyData, (POINTS,), {}),
    'mesh.translate((1, 2, 3))': (SPHERE.translate, ((1, 2, 3),), {}),
    'mesh.clip(normal, origin)': (SPHERE.clip, ('x',), {'origin': (0, 0, 0)}),
}


def main() -> None:
    """Run the benchmark and print or save the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=Path, help='write the timings to this file')
    parser.add_argument('--compare', type=Path, help='print a table of deltas against this file')
    options = parser.parse_args()

    results = {name: measure(*case, repeat=5) for name, case in CASES.items()}
    if options.json:
        options.json.write_text(json.dumps(results, indent=1))
    if options.compare:
        before = json.loads(options.compare.read_text())
        print('| PyVista call | Before (µs) | After (µs) | Speedup |')
        print('| --- | ---: | ---: | ---: |')
        for name, after in results.items():
            speedup = f'{before[name] / after:.1f}x'
            print(f'| `{name}` | {fmt(before[name])} | {fmt(after)} | {speedup} |')
    else:
        for name, value in results.items():
            print(f'{name:45s} {fmt(value):>8s} µs')


if __name__ == '__main__':
    main()
