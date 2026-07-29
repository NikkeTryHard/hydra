"""WP-02x contract-suite helpers.

The BUILD §1 package-selection machinery and the canonical report writer
previously lived here; they were consolidated into the repo-root
``tests/conftest.py`` during the WP-03A/WP-03C single-source cutover because
pytest scopes ``pytest_runtest_logreport`` hookimpls to their conftest's
directory subtree, which silently produced empty reports whenever a matrix
row selected modules outside ``tests/contracts/`` (first observed with
``pixi run test-conformance --package WP-03C`` over ``tests/engines/``).

This module intentionally defines no pytest hooks anymore. The checklist
field registry (``CHECKLIST_FIELDS_BY_MODULE``) now lives beside the writer
in the root conftest, with identical contents and report shape.
"""
