.PHONY: data

PYTHON_INTERPRETER = python3

install: 
	${PYTHON_INTERPRETER} -m pip install -r requirements.txt

data:
	${PYTHON_INTERPRETER} -m src.data.make_datasets data/external data/processed

test:
	pytest