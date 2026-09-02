#!/bin/bash
# Refresh the cached inventories used as fallbacks when a remote is unreachable.
set -euo pipefail
cd "$(dirname "$0")"
curl -sSL https://docs.python.org/3/objects.inv >python-objects.inv
curl -sSL https://numpy.org/doc/stable/objects.inv >numpy-objects.inv
curl -sSL https://docs.scipy.org/doc/scipy/objects.inv >scipy-objects.inv
curl -sSL https://docs.pyvista.org/objects.inv >pyvista-objects.inv
