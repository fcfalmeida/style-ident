import click
import pathlib
from src.data.constants.others import INTERIM_DIR, PROCESSED_DIR
import src.data.commands.make_trainset as make_trainset


@click.command()
def main():
    for path in pathlib.Path(INTERIM_DIR).iterdir():
        if path.is_dir():
            output_path = f'{PROCESSED_DIR}/{path.name}'
            make_trainset.execute(str(path), output_path)

            print(f'Created {output_path} trainset')


if __name__ == '__main__':
    main()
