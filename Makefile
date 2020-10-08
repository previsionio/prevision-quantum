install:
	python3 setup.py install

develop:
	python3 setup.py develop

clean:
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -type d | xargs rm -fr
	rm -fr docs/build/
	rm -fr *.egg *.egg-info/ dist/ build/
	rm -fr build/
	rm -fr dist/
