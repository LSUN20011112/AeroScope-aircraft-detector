import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from runtime_assets import get_model_url, sha256_file


class RuntimeAssetTests(unittest.TestCase):
    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as temp_string:
            path = Path(temp_string) / "sample.bin"
            path.write_bytes(b"AeroScope")
            expected = hashlib.sha256(b"AeroScope").hexdigest()
            self.assertEqual(sha256_file(path), expected)

    def test_environment_model_url_takes_precedence(self):
        with patch.dict(os.environ, {"MODEL_URL": "https://example.com/model.pt"}):
            self.assertEqual(get_model_url(), "https://example.com/model.pt")


if __name__ == "__main__":
    unittest.main()
