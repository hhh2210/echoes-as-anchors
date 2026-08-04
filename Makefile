.PHONY: dist

dist:
	uv build --sdist --clear --out-dir dist
	uv build --wheel dist/*.tar.gz --out-dir dist
	python3 -c "import glob, zipfile; names = zipfile.ZipFile(glob.glob('dist/*.whl')[0]).namelist(); assert not any(name.startswith('src/') for name in names), names"
