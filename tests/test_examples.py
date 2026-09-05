"""Run the examples in the docstrings and the README, which document the behaviour."""

from __future__ import annotations

import doctest
from pathlib import Path
import re

import pytest

import pyvista_validation
from pyvista_validation import _cast_array
from pyvista_validation import check
from pyvista_validation import validate

MODULES = (pyvista_validation, check, validate, _cast_array)
DOCTESTS = [
    test for module in MODULES for test in doctest.DocTestFinder().find(module) if test.examples
]
README = Path(__file__).parents[1] / 'README.md'


def run(test: doctest.DocTest) -> None:
    """Run one doctest and fail with its report if any example disagrees."""
    report: list[str] = []
    results = doctest.DocTestRunner().run(test, out=report.append)
    assert results.failed == 0, ''.join(report)


@pytest.mark.parametrize('test', DOCTESTS, ids=lambda test: test.name)
def test_docstring_examples(test):
    run(test)


def test_readme_examples():
    # The fences around the code blocks are not part of the examples
    text = re.sub(r'^```.*$', '', README.read_text(), flags=re.MULTILINE)
    run(doctest.DocTestParser().get_doctest(text, {}, README.name, str(README), 0))
