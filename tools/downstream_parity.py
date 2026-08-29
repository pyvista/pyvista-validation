"""Compare this package's public API against PyVista's vendored copy.

Run in CI against PyVista's main branch so that validation changes landing
upstream are noticed here rather than at the next release.
"""

from __future__ import annotations

import inspect
import sys

from pyvista.core import _validation as upstream

import pyvista_validation

# _validate_color_sequence builds pyvista.plotting Color objects, so it stayed
# with PyVista when this package was split out.
KNOWN_MISSING = frozenset({'_validate_color_sequence'})


def public_callables(module: object) -> set[str]:
    """Return the names of the public callables a module exports."""
    return {
        name
        for name in dir(module)
        if not name.startswith('_') and callable(getattr(module, name))
    }


def main() -> int:
    """Report any upstream validation API this package fails to match."""
    ours = public_callables(pyvista_validation)
    theirs = public_callables(upstream)

    problems = [
        f'missing: PyVista exports {name!r} but this package does not'
        for name in sorted(theirs - ours - KNOWN_MISSING)
    ]
    for name in sorted(ours & theirs):
        ours_sig = inspect.signature(getattr(pyvista_validation, name))
        theirs_sig = inspect.signature(getattr(upstream, name))
        if ours_sig != theirs_sig:
            problems.append(f'signature: {name}{theirs_sig} upstream, {ours_sig} here')

    if extra := sorted(ours - theirs):
        print(f'Only in this package: {", ".join(extra)}')
    if problems:
        print('\n'.join(problems), file=sys.stderr)
        return 1
    print(f'API parity OK ({len(ours & theirs)} functions checked).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
