Non-physics development tests
=============================

This folder contains small, internal tests that validate tooling and infrastructure
used by the LFM project (logging, runners, helpers). These are not physics
validation tests and can be run independently of the Tier suites.

Guidelines
- Keep each test self-contained and fast (<1 minute ideally)
- Prefer no external dependencies beyond the repo
- Name files with the standard `test_*.py` pattern so they can be discovered by pytest

How to run
- Run only these dev tests:
  - Optional:
    - python -m pytest -q devtests
- Or run an individual test module:
  - Optional:
    - python devtests/test_lfm_logger.py

Note: The main physics test harnesses live as Tier scripts and under the root-level
`test_*.py` files. This folder is explicitly for non-physics/dev sanity checks.
