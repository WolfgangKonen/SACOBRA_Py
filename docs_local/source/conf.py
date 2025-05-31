# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path
#
#print(sys.path)
#sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path('..', '..', 'src').resolve()))
sys.path.insert(0, str(Path('..', '..', 'demo').resolve()))
#print(sys.path)

project = 'SACOBRA_Py'
copyright = '2025, Wolfgang Konen'
author = 'Wolfgang Konen'
release = '0.8'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'furo'
html_static_path = ['_static']
