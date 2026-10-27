"""Switches that turn the runtime checks off."""

from __future__ import annotations

from contextlib import contextmanager
import os
from typing import TYPE_CHECKING

from pyvista_validation import _accelerate

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Mapping
    from typing import TypedDict

    from typing_extensions import Unpack

    class _Switches(TypedDict, total=False):
        """The keyword arguments of :meth:`Config.override`."""

        enabled: bool
        axes: bool
        contains: bool
        finite: bool
        greater_than: bool
        instance: bool
        integer: bool
        iterable: bool
        iterable_items: bool
        length: bool
        less_than: bool
        ndim: bool
        nonnegative: bool
        number: bool
        range: bool
        real: bool
        rotation: bool
        sequence: bool
        shape: bool
        sorted: bool
        string: bool
        subdtype: bool
        type: bool


CHECKS = (
    'axes',
    'contains',
    'finite',
    'greater_than',
    'instance',
    'integer',
    'iterable',
    'iterable_items',
    'length',
    'less_than',
    'ndim',
    'nonnegative',
    'number',
    'range',
    'real',
    'rotation',
    'sequence',
    'shape',
    'sorted',
    'string',
    'subdtype',
    'type',
)
_SWITCHES = ('enabled', *CHECKS)


class Config:
    """Switches that turn the runtime checks off, globally or one kind at a time.

    A switch set to ``False`` skips its check in the ``check`` function and in the
    ``validate`` options that use it; :attr:`enabled` set to ``False`` skips them all.
    Casts and reshapes always run, so valid input gives the same output. Invalid input
    is not reported. ``PYVISTA_VALIDATION_CHECKS=false`` starts with :attr:`enabled` off.

    Examples
    --------
    Skip the sorting checks within a block.

    >>> from pyvista_validation import check_sorted, config
    >>> with config.override(sorted=False):
    ...     check_sorted([3, 1, 2])
    [3, 1, 2]

    Outside the block, the check runs again.

    >>> check_sorted([3, 1, 2])
    Traceback (most recent call last):
        ...
    ValueError: Array with 3 elements must be sorted in ascending order. Got:
        array([3, 1, 2])

    Skip every check within a block.

    >>> from pyvista_validation import validate_array
    >>> with config.override(enabled=False):
    ...     validate_array([-1, 2, 3], must_be_nonnegative=True)
    array([-1,  2,  3])

    """

    __slots__ = ('_values',)

    def __init__(self) -> None:
        self._values: dict[str, bool] = dict.fromkeys(_SWITCHES, True)

    def __repr__(self) -> str:
        """Show every switch."""
        values = ', '.join(f'{name}={value}' for name, value in self._values.items())
        return f'{type(self).__name__}({values})'

    def __eq__(self, other: object) -> bool:
        """Compare every switch."""
        if not isinstance(other, Config):
            return NotImplemented
        return self._values == other._values

    __hash__ = None  # type: ignore[assignment]

    @classmethod
    def from_dict(cls, values: dict[str, bool], /) -> Config:
        """Create a configuration from a dictionary of switches.

        Parameters
        ----------
        values : dict[str, bool]
            Switch names and values, as returned by :meth:`to_dict`. Switches that
            are not given keep their default.

        Returns
        -------
        Config
            A new configuration, separate from :data:`pyvista_validation.config`.

        """
        new = cls()
        new._update(values)
        return new

    def to_dict(self) -> dict[str, bool]:
        """Return every switch as a dictionary.

        Returns
        -------
        dict[str, bool]
            Switch names and values.

        """
        return dict(self._values)

    @contextmanager
    def override(self, **switches: Unpack[_Switches]) -> Iterator[Config]:
        """Set the given switches for the duration of a ``with`` block.

        Parameters
        ----------
        **switches : bool
            Switch names and the values to use within the block.

        Yields
        ------
        Config
            This configuration.

        """
        previous = dict(self._values)
        try:
            self._update(switches)
            yield self
        finally:
            self._values = previous
            _publish(self)

    def _update(self, values: Mapping[str, object], /) -> None:
        """Set several switches, raising for an unknown name or a value that is not a bool."""
        for name, value in values.items():
            if name not in self._values:
                msg = f'{name!r} is not a validation switch. Use one of {_SWITCHES}.'
                raise AttributeError(msg)
            self._set(name, value)

    def _set(self, name: str, value: object, /) -> None:
        """Set one switch, raising unless the value is a bool."""
        if not isinstance(value, bool):
            msg = f'`{name}` must be a bool, got {type(value).__name__}.'
            raise TypeError(msg)
        self._values[name] = value
        _publish(self)

    @property
    def enabled(self) -> bool:
        """Whether any check runs. ``False`` skips every check."""
        return self._values['enabled']

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._set('enabled', value)

    @property
    def axes(self) -> bool:
        """Whether :func:`validate_axes` checks the axes' geometry."""
        return self._values['axes']

    @axes.setter
    def axes(self, value: bool) -> None:
        self._set('axes', value)

    @property
    def contains(self) -> bool:
        """Whether :func:`check_contains` runs."""
        return self._values['contains']

    @contains.setter
    def contains(self, value: bool) -> None:
        self._set('contains', value)

    @property
    def finite(self) -> bool:
        """Whether :func:`check_finite` and ``must_be_finite`` run."""
        return self._values['finite']

    @finite.setter
    def finite(self, value: bool) -> None:
        self._set('finite', value)

    @property
    def greater_than(self) -> bool:
        """Whether :func:`check_greater_than` runs."""
        return self._values['greater_than']

    @greater_than.setter
    def greater_than(self, value: bool) -> None:
        self._set('greater_than', value)

    @property
    def instance(self) -> bool:
        """Whether :func:`check_instance` runs."""
        return self._values['instance']

    @instance.setter
    def instance(self, value: bool) -> None:
        self._set('instance', value)

    @property
    def integer(self) -> bool:
        """Whether :func:`check_integer` and ``must_be_integer`` run."""
        return self._values['integer']

    @integer.setter
    def integer(self, value: bool) -> None:
        self._set('integer', value)

    @property
    def iterable(self) -> bool:
        """Whether :func:`check_iterable` runs."""
        return self._values['iterable']

    @iterable.setter
    def iterable(self, value: bool) -> None:
        self._set('iterable', value)

    @property
    def iterable_items(self) -> bool:
        """Whether :func:`check_iterable_items` runs."""
        return self._values['iterable_items']

    @iterable_items.setter
    def iterable_items(self, value: bool) -> None:
        self._set('iterable_items', value)

    @property
    def length(self) -> bool:
        """Whether :func:`check_length` and the ``must_have_*length`` options run."""
        return self._values['length']

    @length.setter
    def length(self, value: bool) -> None:
        self._set('length', value)

    @property
    def less_than(self) -> bool:
        """Whether :func:`check_less_than` runs."""
        return self._values['less_than']

    @less_than.setter
    def less_than(self, value: bool) -> None:
        self._set('less_than', value)

    @property
    def ndim(self) -> bool:
        """Whether :func:`check_ndim` and ``must_have_ndim`` run."""
        return self._values['ndim']

    @ndim.setter
    def ndim(self, value: bool) -> None:
        self._set('ndim', value)

    @property
    def nonnegative(self) -> bool:
        """Whether :func:`check_nonnegative` and ``must_be_nonnegative`` run."""
        return self._values['nonnegative']

    @nonnegative.setter
    def nonnegative(self, value: bool) -> None:
        self._set('nonnegative', value)

    @property
    def number(self) -> bool:
        """Whether :func:`check_number` runs."""
        return self._values['number']

    @number.setter
    def number(self, value: bool) -> None:
        self._set('number', value)

    @property
    def range(self) -> bool:
        """Whether :func:`check_range` and ``must_be_in_range`` run."""
        return self._values['range']

    @range.setter  # noqa: A003
    def range(self, value: bool) -> None:
        self._set('range', value)

    @property
    def real(self) -> bool:
        """Whether :func:`check_real` and ``must_be_real`` run."""
        return self._values['real']

    @real.setter
    def real(self, value: bool) -> None:
        self._set('real', value)

    @property
    def rotation(self) -> bool:
        """Whether :func:`validate_rotation` checks orthogonality and handedness."""
        return self._values['rotation']

    @rotation.setter
    def rotation(self, value: bool) -> None:
        self._set('rotation', value)

    @property
    def sequence(self) -> bool:
        """Whether :func:`check_sequence` runs."""
        return self._values['sequence']

    @sequence.setter
    def sequence(self, value: bool) -> None:
        self._set('sequence', value)

    @property
    def shape(self) -> bool:
        """Whether :func:`check_shape` and ``must_have_shape`` run."""
        return self._values['shape']

    @shape.setter
    def shape(self, value: bool) -> None:
        self._set('shape', value)

    @property
    def sorted(self) -> bool:
        """Whether :func:`check_sorted` and ``must_be_sorted`` run."""
        return self._values['sorted']

    @sorted.setter  # noqa: A003
    def sorted(self, value: bool) -> None:
        self._set('sorted', value)

    @property
    def string(self) -> bool:
        """Whether :func:`check_string` runs."""
        return self._values['string']

    @string.setter
    def string(self, value: bool) -> None:
        self._set('string', value)

    @property
    def subdtype(self) -> bool:
        """Whether :func:`check_subdtype` and ``must_have_dtype`` run."""
        return self._values['subdtype']

    @subdtype.setter
    def subdtype(self, value: bool) -> None:
        self._set('subdtype', value)

    @property
    def type(self) -> bool:
        """Whether :func:`check_type` runs."""
        return self._values['type']

    @type.setter  # noqa: A003
    def type(self, value: bool) -> None:
        self._set('type', value)


def _publish(source: Config, /) -> None:
    """Apply the live configuration to the checks, in Python and in the C extension."""
    if source is not config:
        return
    on = source._values['enabled']
    enforced._values = {name: on and value for name, value in source._values.items()}
    _accelerate.skip(name for name in CHECKS if not enforced._values[name])


# The switches the checks read, with `enabled` applied to each.
enforced = Config()
config = Config()
_publish(config)
if _accelerate.disabled(os.environ.get('PYVISTA_VALIDATION_CHECKS')):  # pragma: no cover
    config.enabled = False
