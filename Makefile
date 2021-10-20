.PHONY: crossera weiss_feats trainsets train visualize_lda test clean

PYTHON_INTERPRETER = python3

install: 
	${PYTHON_INTERPRETER} -m pip install -r requirements.txt

crossera:
	${PYTHON_INTERPRETER} -m src.data.make_crossera data/external/chroma data/interim/crossera

weiss_feats:
	${PYTHON_INTERPRETER} -m src.data.make_weiss_feats data/interim/crossera data/interim/weiss_feats

hcdf_segmentation:
	${PYTHON_INTERPRETER} -m src.data.make_hcdf_segmentation data/interim/crossera data/interim/hcdf_segmented

tis_feats:
	${PYTHON_INTERPRETER} -m src.data.make_tis_feats data/interim/hcdf_segmented data/interim/tis_feats

weiss_trainset:
	${PYTHON_INTERPRETER} -m src.data.make_weiss_trainset data/interim/weiss_feats data/processed/weiss

tis_trainset:
	${PYTHON_INTERPRETER} -m src.data.make_tis_trainset data/interim/tis_feats data/processed/tis

weiss_train:
	${PYTHON_INTERPRETER} -m src.models.weiss data/processed/weiss models/weiss

tis_train:
	${PYTHON_INTERPRETER} -m src.models.weiss data/processed/tis models/tis

plot_lda:
	${PYTHON_INTERPRETER} -m src.tools.plot_lda $(dataset)

plot_hcdf:
	${PYTHON_INTERPRETER} -m src.tools.plot_hcdf data/interim/crossera/chroma-nnls_full.csv $(piece)

test:
	pytest --cov-report term-missing --cov=src tests/

clean:
	rm -f data/interim/**/*.csv
	rm -f data/processed/**/*.csv