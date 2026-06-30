"""Sphinx configuration for the EvoGrad documentation."""
import os
import sys
from importlib.metadata import PackageNotFoundError, version as _pkg_version

# Make the package importable for autodoc (also installed on Read the Docs).
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "EvoGrad"
author = "Beatrice F. R. Citterio, Daniele M. Papetti, Giovanna Maria Dimitri, Andrea Tangherloni"
copyright = "2026, the EvoGrad authors"

try:
    release = _pkg_version("evograd-diff")
except PackageNotFoundError:
    release = "0.2.1"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",       # Google/NumPy-style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",               # Markdown support
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True   # a few docstrings (e.g. utils.device) use NumPy style
napoleon_include_init_with_doc = False
# Render "Attributes:" sections as :ivar: fields on the class instead of separate
# object descriptions, so they don't collide with autodoc'd @property members
# (removes the "duplicate object description" warnings).
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_title = f"EvoGrad {release}"
html_static_path = []  # add "_static" if custom CSS/assets are introduced
