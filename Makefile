PYTHON ?= python
SPHINXBUILD ?= $(PYTHON) -m sphinx
SOURCEDIR = docs
BUILDDIR = site

.PHONY: html html-full clean

html:
	$(PYTHON) tools/build_site_sources.py
	$(PYTHON) tools/check_public_hygiene.py docs/generated
	$(SPHINXBUILD) -b html -W --keep-going $(SOURCEDIR) $(BUILDDIR)
	$(PYTHON) tools/check_math_render.py docs/generated $(BUILDDIR)
	$(PYTHON) tools/check_static_links.py $(BUILDDIR)

html-full:
	$(PYTHON) tools/build_site_sources.py
	$(PYTHON) tools/check_public_hygiene.py docs/generated
	$(SPHINXBUILD) -E -a -b html -W --keep-going $(SOURCEDIR) $(BUILDDIR)
	$(PYTHON) tools/check_math_render.py docs/generated $(BUILDDIR)
	$(PYTHON) tools/check_static_links.py $(BUILDDIR)

clean:
	$(SPHINXBUILD) -M clean $(SOURCEDIR) $(BUILDDIR)
