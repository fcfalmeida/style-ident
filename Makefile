.PHONY: crossera weiss_feats trainsets train visualize_lda test clean

PYTHON_INTERPRETER = python3

install: 
	${PYTHON_INTERPRETER} -m pip install -r requirements.txt

crossera:
	${PYTHON_INTERPRETER} -m src.data.commands.make_crossera data/external/chroma data/interim/crossera

weiss_feats:
	${PYTHON_INTERPRETER} -m src.data.commands.make_weiss_feats data/interim/crossera data/interim/$(pipeline) $(pipeline)

hcdf_segmentation:
	${PYTHON_INTERPRETER} -m src.data.commands.make_hcdf_segmentation data/interim/crossera data/interim/hcdf_segmented

tis_feats:
	${PYTHON_INTERPRETER} -m src.data.commands.make_tis_feats data/interim/hcdf_segmented data/interim/$(pipeline) $(pipeline)

combine_feats:
	${PYTHON_INTERPRETER} -m src.data.commands.combine_pipeline_output $(pipelines)

trainset:
	${PYTHON_INTERPRETER} -m src.data.commands.make_trainset data/interim/$(pipeline) data/processed/$(pipeline)

train:
	${PYTHON_INTERPRETER} -m src.models.weiss data/processed/$(pipeline) models/$(pipeline)

plot_lda:
	${PYTHON_INTERPRETER} -m src.tools.plot_lda $(dataset) "$(title)"

plot_hcdf:
	${PYTHON_INTERPRETER} -m src.tools.plot_hcdf data/interim/crossera/chroma-nnls_full.csv $(piece)

plot_feature:
	${PYTHON_INTERPRETER} -m src.tools.plot_feature $(dataset) "$(feature)"

test:
	pytest --cov-report term-missing --cov=src tests/

clean:
	rm -f data/interim/**/*.csv
	rm -f data/processed/**/*.csv