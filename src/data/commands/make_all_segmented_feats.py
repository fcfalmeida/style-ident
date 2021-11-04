import click
import src.data.commands.make_segmented_feats as make_segmented_feats
from src.data.pipelines.catalogue import segmented_pipelines


@click.command()
def main():
    for pipeline_name in segmented_pipelines.keys():
        make_segmented_feats.execute(pipeline_name)


if __name__ == '__main__':
    main()
