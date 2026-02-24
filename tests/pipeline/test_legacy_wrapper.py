from __future__ import annotations

import unittest
from pathlib import Path


class TestLegacyWrapper(unittest.TestCase):
    def test_wrapper_contains_deprecation_and_target(self) -> None:
        wrapper = Path("/home/yushanhui/workspace/GEM/scripts/llava_scripts/finetune.sh")
        self.assertTrue(wrapper.exists())
        text = wrapper.read_text(encoding="utf-8")
        self.assertIn("[DEPRECATED]", text)
        self.assertIn("legacy/llava_scripts/finetune.sh", text)


if __name__ == "__main__":
    unittest.main()
