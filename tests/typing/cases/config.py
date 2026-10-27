"""Typing cases for the check switches."""

from __future__ import annotations

from type_assert import assert_types

from pyvista_validation import Config
from pyvista_validation import config

assert_types(config.enabled, bool)
assert_types(config.sorted, bool)
assert_types(config.to_dict(), dict[str, bool])
assert_types(Config.from_dict({'sorted': False}), Config)
assert_types(Config().override(sorted=False, enabled=True).__enter__(), Config)
