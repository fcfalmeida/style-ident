import pandas as pd
import pathlib
from src.data.constants.others import INTERIM_DIR


class PipelineOutputCombiner:
    @classmethod
    def combine(
            cls, dataset: str, pipelines: list[str],
            type: str, index_col: str) -> pd.DataFrame:

        combined_data = None

        for p in pipelines:
            files = pathlib.Path(
                f'{INTERIM_DIR}/{dataset}/{p}').glob(f'*{type}*')

            for f in files:
                data = pd.read_csv(
                    f, dtype={'piece': str}, index_col=index_col
                )

                if combined_data is None:
                    combined_data = data.copy()
                else:
                    combined_data = combined_data.join(data, rsuffix='_')

        return combined_data
