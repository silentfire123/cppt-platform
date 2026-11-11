from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from scripts import run_backend


class BackendEnvironmentArgsTest(unittest.TestCase):
    def test_storage_backend_override_is_forwarded_to_wsl(self) -> None:
        with patch.dict(os.environ, {"XUNTUI_STORAGE_BACKEND": "sqlite"}, clear=False):
            self.assertIn(
                "XUNTUI_STORAGE_BACKEND=sqlite",
                run_backend.backend_environment_args(),
            )


if __name__ == "__main__":
    unittest.main()
