.PHONY: crossera res_feats all_res_feats hcdf_segmentation combine_feats \
	combine_feats_multiple trainset all_trainsets train \
	plot_lda plot_hcdf plot_feature extract_chroma test clean

PYTHON_INTERPRETER = python3

install: 
	${PYTHON_INTERPRETER} -m pip install -r requirements.txt

crossera:
	${PYTHON_INTERPRETER} -m src.data.commands.make_crossera data/external/chroma data/external/crossera

res_feats:
	${PYTHON_INTERPRETER} -m src.data.commands.make_res_feats $(dataset) $(pipeline)

all_res_feats:
	${PYTHON_INTERPRETER} -m src.data.commands.make_all_res_feats $(dataset)

hcdf_segmentation:
	${PYTHON_INTERPRETER} -m src.data.commands.make_hcdf_segmentation $(dataset)

segmented_feats:
	${PYTHON_INTERPRETER} -m src.data.commands.make_segmented_feats $(dataset) $(pipeline)

all_segmented_feats:
	${PYTHON_INTERPRETER} -m src.data.commands.make_all_segmented_feats $(dataset)

combine_feats:
	${PYTHON_INTERPRETER} -m src.data.commands.combine_pipeline_output $(dataset) $(pipelines)

combine_feats_multiple:
	${PYTHON_INTERPRETER} -m src.data.commands.combine_features_multiple $(dataset)

trainset:
	${PYTHON_INTERPRETER} -m src.data.commands.make_trainset $(dataset) $(pipeline)

all_trainsets:
	${PYTHON_INTERPRETER} -m src.data.commands.make_all_trainsets $(dataset)

train:
	${PYTHON_INTERPRETER} -m src.models.weiss $(dataset) $(pipeline)

train_all:
	${PYTHON_INTERPRETER} -m src.data.commands.train_all $(dataset)

plot_lda:
	${PYTHON_INTERPRETER} -m src.tools.plot_lda $(dataset) "$(title)"

plot_hcdf:
	${PYTHON_INTERPRETER} -m src.tools.plot_hcdf data/external/crossera/chroma-nnls_full.csv $(piece)

plot_feature:
	${PYTHON_INTERPRETER} -m src.tools.plot_feature $(dataset) "$(feature)"

extract_chroma:
	${PYTHON_INTERPRETER} -m src.tools.extract_chroma $(dataset)

test:
	pytest --cov-report term-missing --cov=src tests/

clean:
	rm -rf data/interim/*
	rm -rf data/processed/*