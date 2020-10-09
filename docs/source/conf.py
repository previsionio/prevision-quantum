# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'prevision-quantum-nn'
copyright = '2020, Prevision.io'
author = 'Prevision.io'

templates_path = ["_templates"]

# The full version, including alpha/beta/rc tags
release = '1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'autoapi.extension',
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]

master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autoapi_dirs = ["../../prevision_quantum_nn"]
autoapi_template_dir = '_templates'
# pygments_style="monokai"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "press"
html_theme_options = {
    # Title shown in the top left. (Default: ``project`` value.)
    # "navbar_title": "Prevision-qnn",

    # Links to shown in the top bar. (Default: top-level ``toctree`` entries.)
    # "navbar_links": [
    #     ("Install", "install"),
    #     ("Quickstart", "quickstart"),
    #     ("Tutorials", "tutorials"),
    #     ("Benchmarks", "benchmarks"),
    #     ("About", "about"),
    # ],

    # If ``github_link`` is set, a GitHub icon will be shown in the top right.
    "external_links": [ ( "Github", "https://github.com/MichelNowak1/prevision-qnn",) ]

    # Text to show in the footer of every page.
    # "footer_text": "Prevision.io",

    # Google Analytics ID. (Optional.)
    # "analytics_id": "UA-XXXXX-X"
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [ 'css/custom.css', ]
