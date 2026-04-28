train: 
	pip install -r ml/requirements.txt
	python -m ml.src.train

test: 
	pip install -r ml/requirements.txt 
	python -m ml.src.test
