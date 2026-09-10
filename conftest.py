"""Pytest bootstrap.

Its mere presence at the project root makes pytest add the root to ``sys.path``
(prepend import mode), so ``import modules...`` / ``import app`` resolve whether
the suite is run as ``pytest`` (console script) or ``python -m pytest``. Without
it, a bare ``pytest`` invocation — which is what CI runs — fails to import the
top-level modules.
"""
