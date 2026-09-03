"""A tiny stand-in for the small slice of the pytest API this suite uses.

This repo's dependency set is deliberately just torch. If pytest is installed,
`pytest -v` runs the tests normally. If it is not, `python tests/run_tests.py`
injects this module as `pytest` and runs the *same* test file unmodified.

Supported: `raises`, `fixture`, `skip`, `mark.skipif`, `mark.parametrize`,
and a module-level `pytestmark`. Nothing else -- keep the tests inside that
subset, or install real pytest.
"""


class Skipped(Exception):
    """Raised to signal a skipped test (mirrors pytest.skip.Exception)."""

    def __init__(self, reason: str = ""):
        super().__init__(reason)
        self.reason = reason


def skip(reason: str = ""):
    raise Skipped(reason)


skip.Exception = Skipped


class _Raises:
    """Context manager mirroring `pytest.raises`, including `.value`."""

    def __init__(self, expected):
        self.expected = expected
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError(f"DID NOT RAISE {self.expected}")
        if issubclass(exc_type, Skipped):
            return False  # let skips propagate
        if not issubclass(exc_type, self.expected):
            return False  # unexpected type -> propagate as a real failure
        self.value = exc
        return True  # suppress: this is the expected exception


def raises(expected):
    return _Raises(expected)


def fixture(*args, **kwargs):
    """Mark a function as a fixture so the runner does not collect it as a test.

    The only fixture in this suite is an autouse reset hook, which the runner
    applies itself; the body is not invoked through this shim.
    """
    def decorate(fn):
        fn._is_fixture = True
        return fn

    if args and callable(args[0]):
        return decorate(args[0])
    return decorate


class _Mark:
    @staticmethod
    def skipif(condition, reason=""):
        def decorate(fn):
            existing = list(getattr(fn, "_skipifs", []))
            existing.append((bool(condition), reason))
            fn._skipifs = existing
            return fn

        return decorate

    @staticmethod
    def parametrize(argnames, argvalues):
        names = ([a.strip() for a in argnames.split(",")]
                 if isinstance(argnames, str) else list(argnames))

        def decorate(fn):
            existing = list(getattr(fn, "_params", []))
            existing.append((names, list(argvalues)))
            fn._params = existing
            return fn

        return decorate


mark = _Mark()
