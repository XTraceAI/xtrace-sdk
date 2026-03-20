# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
import tomllib

sys.path.insert(0, os.path.abspath('../src'))

with open(os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml'), 'rb') as _f:
    _meta = tomllib.load(_f)

project = 'XTrace SDK'
copyright = '2026, Liwen O.@XTrace Inc.'
author = 'Liwen O.'
release = _meta['project']['version']
version = '.'.join(release.split('.')[:2])  # major.minor shown in sidebar

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "autoapi.extension",
    "sphinx_copybutton",
    "myst_parser",
]
# Let Sphinx resolve stdlib references like abc.ABC, Sequence, etc.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),  # ← None, not {}
    # add third-party libs as needed, e.g.:
    # "numpy": ("https://numpy.org/doc/stable/", None),
}


# Make type rendering less brittle
autodoc_typehints = "description"     # or "both"
autodoc_typehints_format = "short"    # avoid super long fully-qualified names

# If you're using Google/NumPy docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

autoapi_type = "python"
autoapi_dirs = ["../src/xtrace_sdk"]  # path to your package
autoapi_keep_files = False
autoapi_ignore = [
    "*/xtrace_types*",   # pure type defs — TypedDict fields trip up autoapi RST rendering
    "*/cli/*",           # CLI is internal tooling; relative imports break autoapi parsing
]
autoapi_add_toctree_entry = True      # auto-injects into the last toctree in index.rst (API Reference)
suppress_warnings = [
    "autoapi.python_import_resolution",  # expected: xtrace_types excluded from autoapi
    "myst.header",                        # CHANGELOG.md starts at H2 by design (RST page provides H1)
    "ref.python",                         # XTraceIntegration re-exported in __init__.py — dual cross-ref target is fine
]

templates_path = ['_templates']
html_static_path = ['_static']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', "api_rst/**"]

# Suppress nitpicky link errors for known-bad targets
nitpick_ignore = [
    ("py:class", "abc.ABC"),                         # resolved by intersphinx; keep if still noisy
]

# Broader, regex-based ignores to get CI green while you refactor types
nitpick_ignore_regex = [
    (r"py:class", r"collections\.abc\..*"),          # until you normalize types
    (r"py:class", r"Goldwasser_MicaliEncryptedNumber"),
    (r"py:class", r"DocumentCollection"),
    (r"py:.*",    r"[01]"),                          # weird numeric refs—likely docstring typos
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'

pygments_style = 'friendly'
pygments_dark_style = 'monokai'

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "announcement": "Announcement placeholder.",
    "light_css_variables": {
        # Brand — purple accent throughout
        "color-brand-primary":          "#6d28d9",   # purple-700: nav highlights, hover
        "color-brand-content":          "#5b21b6",   # purple-800: links, inline code text
        # Inline code — light purple tint, brand-coloured text
        "color-inline-code-background": "#f5f3ff",
        # Monospace font stack
        "font-stack--monospace": (
            "'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Menlo', monospace"
        ),
    },
    "dark_css_variables": {
        # Brand — lighter purples readable on dark backgrounds
        "color-brand-primary":          "#a78bfa",   # violet-400
        "color-brand-content":          "#c4b5fd",   # violet-300
        # Inline code
        "color-inline-code-background": "#1e1540",
        # Monospace font stack
        "font-stack--monospace": (
            "'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Menlo', monospace"
        ),
    },
}
