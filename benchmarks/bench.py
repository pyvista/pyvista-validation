"""Time every public function on typical small input and, where it matters, on large arrays.

Run ``python benchmarks/bench.py --json before.json`` on one revision and
``python benchmarks/bench.py --compare before.json`` on another to get a table of deltas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import timeit

import numpy as np

import pyvista_validation as pv

N = 1_000_000
rng = np.random.default_rng(0)
ONES3 = np.ones(3)
POINTS = rng.random((N, 3))
SORTED = np.arange(N, dtype=float)
FLOATS = np.arange(N, dtype=float)
IDS = np.arange(N)
INT3 = np.array([1, 2, 3])

CASES: dict[str, tuple] = {
    'check_subdtype': (pv.check_subdtype, (np.zeros(3), np.floating), {}),
    'check_real': (pv.check_real, (ONES3,), {}),
    'check_sorted small': (pv.check_sorted, ([1, 2, 3],), {}),
    'check_sorted 1e6': (pv.check_sorted, (SORTED,), {}),
    'check_finite small': (pv.check_finite, (ONES3,), {}),
    'check_finite 1e6x3': (pv.check_finite, (POINTS,), {}),
    'check_integer small': (pv.check_integer, ([1.0, 2.0],), {}),
    'check_integer 1e6': (pv.check_integer, (FLOATS,), {}),
    'check_nonnegative small': (pv.check_nonnegative, (ONES3,), {}),
    'check_nonnegative 1e6': (pv.check_nonnegative, (FLOATS,), {}),
    'check_greater_than small': (pv.check_greater_than, (ONES3, 0), {}),
    'check_greater_than 1e6': (pv.check_greater_than, (FLOATS, -1), {}),
    'check_less_than small': (pv.check_less_than, (ONES3, 2), {}),
    'check_less_than 1e6': (pv.check_less_than, (FLOATS, N), {}),
    'check_range small': (pv.check_range, (ONES3, [0, 2]), {}),
    'check_range 1e6': (pv.check_range, (FLOATS, [0, N]), {}),
    'check_shape array': (pv.check_shape, (ONES3, 3), {}),
    'check_shape list': (pv.check_shape, ([1, 2, 3], [(), -1]), {}),
    'check_ndim': (pv.check_ndim, (ONES3, 1), {}),
    'check_number': (pv.check_number, (1.5,), {}),
    'check_string': (pv.check_string, ('abc',), {}),
    'check_sequence': (pv.check_sequence, ([1, 2],), {}),
    'check_iterable': (pv.check_iterable, ([1, 2],), {}),
    'check_instance': (pv.check_instance, (1, int), {}),
    'check_type': (pv.check_type, (1, int), {}),
    'check_iterable_items small': (pv.check_iterable_items, ([1, 2, 3], int), {}),
    'check_iterable_items 1e4': (pv.check_iterable_items, (list(range(10_000)), int), {}),
    'check_contains': (pv.check_contains, (['a', 'b'], 'a'), {}),
    'check_length': (pv.check_length, ([1, 2, 3],), {'exact_length': 3}),
    'validate_array ndarray': (pv.validate_array, (ONES3,), {}),
    'validate_array list': (pv.validate_array, ([1, 2, 3],), {}),
    'validate_array list constraints': (
        pv.validate_array,
        ([1, 2, 3],),
        {'must_be_in_range': [0, 5], 'dtype_out': float, 'to_tuple': True},
    ),
    'validate_array 1e6x3 finite': (
        pv.validate_array,
        (POINTS,),
        {'must_have_shape': (-1, 3), 'must_be_finite': True},
    ),
    'validate_number int': (pv.validate_number, (1,), {}),
    'validate_number np.float64': (pv.validate_number, (np.float64(1.5),), {}),
    'validate_data_range': (pv.validate_data_range, ([0, 1],), {}),
    'validate_arrayNx3 list': (pv.validate_arrayNx3, ([[1, 2, 3]],), {}),
    'validate_arrayNx3 1e6x3 finite': (pv.validate_arrayNx3, (POINTS,), {'must_be_finite': True}),
    'validate_arrayN list': (pv.validate_arrayN, ([1, 2, 3],), {}),
    'validate_arrayN 1e6': (pv.validate_arrayN, (IDS,), {}),
    'validate_arrayN_unsigned list': (pv.validate_arrayN_unsigned, ([0, 1, 2],), {}),
    'validate_arrayN_unsigned 1e6': (pv.validate_arrayN_unsigned, (IDS,), {}),
    'validate_array3 list': (pv.validate_array3, ([1, 2, 3],), {}),
    'validate_array3 ndarray': (pv.validate_array3, (ONES3,), {}),
    'validate_array3 broadcast': (pv.validate_array3, (1.0,), {'broadcast': True}),
    'validate_dimensionality int': (pv.validate_dimensionality, (2,), {}),
    'validate_dimensionality alias': (pv.validate_dimensionality, ('2D',), {}),
    'validate_axes matrix': (pv.validate_axes, (np.eye(3),), {}),
    'validate_axes vectors': (pv.validate_axes, ([1, 0, 0], [0, 1, 0], [0, 0, 1]), {}),
    'validate_rotation': (pv.validate_rotation, (np.eye(3),), {}),
    'validate_transform4x4 4x4': (pv.validate_transform4x4, (np.eye(4),), {}),
    'validate_transform4x4 3x3': (pv.validate_transform4x4, (np.eye(3),), {}),
    'validate_transform3x3': (pv.validate_transform3x3, (np.eye(3),), {}),
}

try:
    from vtkmodules.vtkCommonMath import vtkMatrix3x3
    from vtkmodules.vtkCommonMath import vtkMatrix4x4
    from vtkmodules.vtkCommonTransforms import vtkTransform
except ModuleNotFoundError:
    pass
else:
    CASES['validate_transform4x4 vtkMatrix4x4'] = (pv.validate_transform4x4, (vtkMatrix4x4(),), {})
    CASES['validate_transform4x4 vtkTransform'] = (pv.validate_transform4x4, (vtkTransform(),), {})
    CASES['validate_transform3x3 vtkMatrix3x3'] = (pv.validate_transform3x3, (vtkMatrix3x3(),), {})
try:
    from scipy.spatial.transform import Rotation
except ModuleNotFoundError:
    pass
else:
    CASES['validate_transform3x3 Rotation'] = (
        pv.validate_transform3x3,
        (Rotation.identity(),),
        {},
    )
    CASES['validate_rotation Rotation'] = (pv.validate_rotation, (Rotation.identity(),), {})


def measure(function, args, kwargs, *, repeat: int = 7) -> float:
    """Return the best per-call time in microseconds."""
    timer = timeit.Timer(lambda: function(*args, **kwargs))
    number, _ = timer.autorange()
    return min(timer.repeat(repeat, number)) / number * 1e6


def fmt(value: float) -> str:
    """Format microseconds with a sensible number of digits."""
    return f'{value:.2f}' if value < 100 else f'{value:.0f}'


def main() -> None:
    """Run the benchmark and print or save the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=Path, help='write the timings to this file')
    parser.add_argument('--compare', type=Path, help='print a table of deltas against this file')
    parser.add_argument('--only', help='substring filter on the case names')
    options = parser.parse_args()

    results = {
        name: measure(*case)
        for name, case in CASES.items()
        if options.only is None or options.only in name
    }
    if options.json:
        options.json.write_text(json.dumps(results, indent=1))
    if options.compare:
        before = json.loads(options.compare.read_text())
        print('| Call | Before (µs) | After (µs) | Speedup |')
        print('| --- | ---: | ---: | ---: |')
        for name, after in results.items():
            if name in before:
                print(
                    f'| `{name}` | {fmt(before[name])} | {fmt(after)} | {before[name] / after:.1f}x |'
                )
    else:
        for name, value in results.items():
            print(f'{name:40s} {fmt(value):>8s} µs')


if __name__ == '__main__':
    main()
