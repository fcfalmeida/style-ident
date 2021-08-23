.PHONY: data clean

PYTHON_INTERPRETER = python3

install: 
	${PYTHON_INTERPRETER} -m pip install -r requirements.txt

data:
	${PYTHON_INTERPRETER} -m src.data.make_datasets data/interim data/processed

crossera:
	${PYTHON_INTERPRETER} -m src.data.make_crossera data/external/chroma data/interim

test:
	pytest --cov-report term-missing --cov=src tests/

clean:
	rm data/interim/*
	rm data/processed/* \