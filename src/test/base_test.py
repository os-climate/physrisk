import os
import pathlib
import shutil
import tempfile
import unittest

from dotenv import load_dotenv


class TestWithCredentials(unittest.TestCase):
    """Test that attempts to load contents of credentials.env into environment variables (if present)"""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        dotenv_dir = os.environ.get("CREDENTIAL_DOTENV_DIR", os.getcwd())
        dotenv_path = pathlib.Path(dotenv_dir) / "credentials.env"
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path, override=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)
