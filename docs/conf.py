# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Migrate as much of the configuration as possible into pyproject.toml
from sphinx_pyproject import SphinxConfig

# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../src/"))

config = SphinxConfig("../pyproject.toml", globalns=globals())
