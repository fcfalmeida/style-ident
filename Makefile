.PHONY: data clean

PYTHON_INTERPRETER = python3

install: 
	${PYTHON_INTERPRETER} -m pip install -r requirements.txt

data:
	${PYTHON_INTERPRETER} -m src.data.make_datasets data/external/chroma data/processed

crossera:
	${PYTHON_INTERPRETER} -m src.data.make_crossera_full data/external/chroma data/external/chroma

test:
	pytest --cov-report term-missing --cov=src tests/

clean:
	rm data/processed/*