.PHONY: crossera data trainsets clean test

PYTHON_INTERPRETER = python3

install: 
	${PYTHON_INTERPRETER} -m pip install -r requirements.txt

crossera:
	${PYTHON_INTERPRETER} -m src.data.make_crossera data/external/chroma data/interim

data:
	${PYTHON_INTERPRETER} -m src.data.make_datasets data/interim data/interim/chroma_resolutions

trainsets:
	${PYTHON_INTERPRETER} -m src.data.make_trainsets data/interim/chroma_resolutions data/processed

test:
	pytest --cov-report term-missing --cov=src tests/

clean:
	rm -f data/interim/*.csv
	rm -f data/interim/chroma_resolutions/*.csv
	rm -f data/processed/*.csv