.PHONY: crossera weiss_feats trainsets train visualize_lda test clean

PYTHON_INTERPRETER = python3

install: 
	${PYTHON_INTERPRETER} -m pip install -r requirements.txt

crossera:
	${PYTHON_INTERPRETER} -m src.data.make_crossera data/external/chroma data/interim/crossera

weiss_feats:
	${PYTHON_INTERPRETER} -m src.data.make_weiss_feats data/interim/crossera data/interim/weiss_feats

hcdf_segmentation:
	${PYTHON_INTERPRETER} -m src.data.make_hcdf_segmentation data/interim/crossera data/interim/tis_feats

trainsets:
	${PYTHON_INTERPRETER} -m src.data.make_trainsets data/interim/weiss_feats data/processed/weiss

train:
	${PYTHON_INTERPRETER} -m src.models.weiss data/processed/weiss models/weiss

visualize_lda:
	${PYTHON_INTERPRETER} -m src.tools.visualize_lda data/processed/weiss/chroma-nnls_full.csv

test:
	pytest --cov-report term-missing --cov=src tests/

clean:
	rm -f data/interim/**/*.csv
	rm -f data/processed/**/*.csv