"""pyvista-validation documentation configuration."""

from __future__ import annotations

from datetime import datetime
from importlib.metadata import version as _metadata_version

extensions = [
    'notfound.extension',
    'numpydoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
]

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'

project = 'pyvista-validation'
copyright = f'{datetime.now().year}, PyVista'  # noqa: A001, DTZ005
author = 'The PyVista developers'
release = version = _metadata_version('pyvista-validation')

language = 'en'
exclude_patterns = ['_build']
pygments_style = 'sphinx'

# -- API generation ---------------------------------------------------------

autosummary_generate = True
# The class-members table numpydoc adds is redundant with the autosummary pages.
numpydoc_show_class_members = False

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'python': ('https://docs.python.org/3/', None),
    'pyvista': ('https://docs.pyvista.org/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# -- Options for HTML output --------------------------------------------------

html_theme = 'sphinx_book_theme'
html_context = {
    'github_user': 'pyvista',
    'github_repo': 'input-validation',
    'github_version': 'main',
    'doc_path': 'doc',
}
html_show_sourcelink = False
html_baseurl = 'https://validation.pyvista.org/'

html_theme_options = {
    'show_prev_next': False,
    'github_url': 'https://github.com/pyvista/input-validation',
    'collapse_navigation': True,
    'use_edit_page_button': True,
    'navigation_with_keys': False,
    'show_navbar_depth': 1,
    # Kept in sync with the icon links on docs.pyvista.org, minus the ones
    # that do not apply to this package.
    'icon_links': [
        {
            'name': 'Slack Community',
            'url': 'https://communityinviter.com/apps/pyvista/pyvista',
            'icon': 'fab fa-slack',
        },
        {
            'name': 'Support',
            'url': 'https://github.com/pyvista/pyvista/discussions',
            'icon': 'fa fa-comment fa-fw',
        },
        {
            'name': 'PyPI',
            'url': 'https://pypi.org/project/pyvista-validation',
            'icon': 'fa-brands fa-python',
        },
    ],
}

html_static_path = ['_static']
