#!/usr/bin/env python3
import pytest
import tempfile

class TestDataset:

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        tmp_dir = tempfile.TemporaryDirectory()
        yield
        tmp_dir.cleanup()
