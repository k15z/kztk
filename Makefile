docs:
	pip install --upgrade .
	pdoc --overwrite --html --html-dir docs kztk

gh-pages:
	git subtree push --prefix docs/kztk origin gh-pages
