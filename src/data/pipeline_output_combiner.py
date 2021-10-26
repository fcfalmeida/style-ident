import pandas as pd
import pathlib
from src.data.constants import INTERIM_DIR


class PipelineOutputCombiner:
    @classmethod
    def combine(cls, pipelines: list[str], type: str) -> pd.DataFrame:

        combined_data = None

        for p in pipelines:
            files = pathlib.Path(f'{INTERIM_DIR}/{p}').glob(f'*{type}*')

            for f in files:
                data = pd.read_csv(f, dtype={'piece': str})

                if combined_data is None:
                    combined_data = data.copy()
                else:
                    combined_data = combined_data.join(data, rsuffix='_')

        return combined_data