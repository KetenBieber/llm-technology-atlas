project = "LLM Technology Atlas"
author = "KetenBieber"
release = "2026.09"

extensions = [
    "sphinxcontrib.mermaid",
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_rtd_theme",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
language = "zh_CN"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

pygments_style = "sphinx"
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": -1,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
}
html_title = "LLM Technology Atlas"
html_show_sourcelink = False
html_copy_source = False
html_search_language = "zh"
html_show_search_summary = False
html_static_path = ["_static"]
html_css_files = ["custom.css"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "tasklist",
]
myst_heading_anchors = 4
copybutton_exclude = ".linenos, .gp, .go"
