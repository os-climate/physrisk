# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../src/"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PhysicalRisk"
copyright = "2023, OSC"
author = "OSC"

# The full version, including alpha/beta/rc tags
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# The master toctree document.
master_doc = "index"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_toolbox.installation",
    "sphinx_toolbox.latex",
    "sphinx.ext.autosummary",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.autosectionlabel",
    "sphinx_design",
    "sphinx.ext.intersphinx",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Not show module
python_module_index = False

# Summary
autosummary_generate = True

# Docstrings of private methods
autodoc_default_options = {"members": True, "undoc-members": True, "private-members": False}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_logo = "images/OS-Climate-Logo.png"
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# Don't show the code
html_show_sourcelink = True

html_theme_options = {
    # 'logo_only': False,
    # 'display_version': False,
    # Table of contents options
    "collapse_navigation": False,
}

html_sidebars = {
    "**": [
        "globaltoc.html",  # Índice general
        # 'localtoc.html',   # Índice local para cada archivo
        "searchbox.html",  # Cuadro de búsqueda
    ]
}

# This setting ensures that each section in your documentation is automatically assigned
# a unique label based on the document it belongs to.
autosectionlabel_prefix_document = True

# show the members in the order they appear in the source code, you can use the autodoc_member_order option.
autodoc_member_order = "bysource"
