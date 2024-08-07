import os
import pathlib
import shutil
import tarfile
import tempfile

import pandas as pd
import pytest
from dotenv import load_dotenv


@pytest.fixture(scope="function")
def test_dir():
    # Setup
    test_dir = tempfile.mkdtemp()
    yield test_dir
    # Teardown
    shutil.rmtree(test_dir)


@pytest.fixture(
    scope="function",
)
def load_credentials():
    dotenv_dir = os.environ.get("CREDENTIAL_DOTENV_DIR", os.getcwd())
    dotenv_path = pathlib.Path(dotenv_dir) / "credentials.env"
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path, override=True)


@pytest.fixture(
    scope="function",
)
def wri_power_plant_assets():
    """
    Load the WRI Global Power Plant Database dataset.

    This function extracts the `global_power_plant_database.csv` file from the
    `wri_global_power_plant_database.tbz2` archive and loads it into a pandas DataFrame.

    The dataset is sourced from the World Resources Institute's (WRI) Global Power Plant Database.
    The original dataset and more information can be found at:
    https://datasets.wri.org/dataset/globalpowerplantdatabase

    License:
        The dataset is provided under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
        This means you are free to:
        - Share: copy and redistribute the material in any medium or format
        - Adapt: remix, transform, and build upon the material for any purpose, even commercially.

        Under the following terms:
        - Attribution: You must give appropriate credit, provide a link to the license, and indicate if changes were made.
          You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

        More details about the license can be found at: https://creativecommons.org/licenses/by/4.0/

    Note:
        The original README file and a copy of the CC BY 4.0 license are included in the tar.bz2 archive.

    Returns:
        pandas.DataFrame: A DataFrame containing the Global Power Plant Database.
    """
    with tarfile.open(
        "./tests/api/wri_global_power_plant_database.tbz2", "r:bz2"
    ) as tf:
        with tf.extractfile("global_power_plant_database.csv") as f:
            asset_list = pd.read_csv(f, low_memory=False)

    return asset_list
