import click
import src.data.commands.make_res_feats as make_res_feats
from src.data.pipelines.catalogue import res_pipelines


@click.command()
def main():
    for pipeline_name in res_pipelines.keys():
        make_res_feats.execute(pipeline_name)


if __name__ == '__main__':
    main()
