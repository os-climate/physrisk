from pathlib import Path
from dotenv import load_dotenv


def load_credentials():
    """Function to load credentials from a credentials.env file stored in the physrisk folder at the top level
    (i.e. same level as pyproject.toml)
    """
    path = Path(__file__).parents[3] / "credentials.env"
    load_dotenv(path)
