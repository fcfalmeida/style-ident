import click
import src.data.commands.make_res_feats as make_res_feats
from src.data.pipelines.catalogue import res_pipelines


@click.command()
@click.argument('dataset', type=str)
def main(dataset):
    for pipeline_name in res_pipelines.keys():
        make_res_feats.execute(dataset, pipeline_name)


if __name__ == '__main__':
    main()
