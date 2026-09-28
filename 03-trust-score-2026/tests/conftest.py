import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "llm: calls a real LLM provider; skipped unless RUN_LLM_TESTS=1 and a key is set"
    )


def pytest_collection_modifyitems(config, items):
    if os.getenv("RUN_LLM_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="LLM test: set RUN_LLM_TESTS=1 and provider key to run")
    for item in items:
        if "llm" in item.keywords:
            item.add_marker(skip)
