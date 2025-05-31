# Configuration file for the Sphinx documentation builder.

import sys
from pathlib import Path
#
#print(sys.path)
#sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path('..', '..', 'src').resolve()))
sys.path.insert(0, str(Path('..', '..', 'demo').resolve()))
#print(sys.path)

# -- Project information

project = 'SACOBRA_Py'
copyright = '2025, Wolfgang Konen'
author = 'Wolfgang Konen'
release = '0.8'


# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
