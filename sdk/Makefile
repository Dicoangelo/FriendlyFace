.PHONY: build clean check publish test

build:
	python -m build

clean:
	rm -rf dist/ *.egg-info/ build/

check: build
	twine check dist/*

test:
	pytest tests/ -v

publish: check
	twine upload dist/*
