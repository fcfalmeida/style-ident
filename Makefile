.PHONY: crossera res_feats all_res_feats hcdf_segmentation combine_feats \
	combine_feats_multiple trainset all_trainsets train \
	plot_lda plot_hcdf plot_feature test clean

PYTHON_INTERPRETER = python3

install: 
	${PYTHON_INTERPRETER} -m pip install -r requirements.txt

crossera:
	${PYTHON_INTERPRETER} -m src.data.commands.make_crossera data/external/chroma data/external/crossera

res_feats:
	${PYTHON_INTERPRETER} -m src.data.commands.make_res_feats $(pipeline)

all_res_feats:
	${PYTHON_INTERPRETER} -m src.data.commands.make_all_res_feats

hcdf_segmentation:
	${PYTHON_INTERPRETER} -m src.data.commands.make_hcdf_segmentation data/external/crossera data/external/hcdf_segmented

segmented_feats:
	${PYTHON_INTERPRETER} -m src.data.commands.make_segmented_feats $(pipeline)

all_segmented_feats:
	${PYTHON_INTERPRETER} -m src.data.commands.make_all_segmented_feats

combine_feats:
	${PYTHON_INTERPRETER} -m src.data.commands.combine_pipeline_output $(pipelines)

combine_feats_multiple:
	${PYTHON_INTERPRETER} -m src.data.commands.combine_features_multiple

trainset:
	${PYTHON_INTERPRETER} -m src.data.commands.make_trainset data/interim/$(pipeline) data/processed/$(pipeline)

all_trainsets:
	${PYTHON_INTERPRETER} -m src.data.commands.make_all_trainsets

train:
	${PYTHON_INTERPRETER} -m src.models.weiss data/processed/$(pipeline) models/$(pipeline)

plot_lda:
	${PYTHON_INTERPRETER} -m src.tools.plot_lda $(dataset) "$(title)"

plot_hcdf:
	${PYTHON_INTERPRETER} -m src.tools.plot_hcdf data/external/crossera/chroma-nnls_full.csv $(piece)

plot_feature:
	${PYTHON_INTERPRETER} -m src.tools.plot_feature $(dataset) "$(feature)"

test:
	pytest --cov-report term-missing --cov=src tests/

clean:
	rm -rf data/interim/*
	rm -rf data/processed/*