import pandas as pd
from numpy.typing import ArrayLike
from src.features.feature_extractor import FeatureExtractor
from src.data.constants import CHROMA_COLS, INTERVAL_COLS, CHORD_MAJ, CHORD_MIN, CHORD_DIM, CHORD_AUG, CHORD_COLS

class TemplateBased(FeatureExtractor):
    TEMPLATES = {
        INTERVAL_COLS[0]: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        INTERVAL_COLS[1]: [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        INTERVAL_COLS[2]: [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        INTERVAL_COLS[3]: [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        INTERVAL_COLS[4]: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        INTERVAL_COLS[5]: [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        CHORD_MAJ: [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        CHORD_MIN: [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        CHORD_DIM: [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        CHORD_AUG: [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    }

    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        data_copy = data.copy()

        for template_key in self.TEMPLATES.keys():
            data_copy[template_key] = self._calc_template_feat(data[CHROMA_COLS].values, template_key)

        cols = []
        cols.extend(INTERVAL_COLS)
        cols.extend(CHORD_COLS)
        
        return data_copy[cols]

    def _calc_template_feat(self, chroma_vector: ArrayLike, template_key: str) -> ArrayLike:
        template = self.TEMPLATES[template_key]

        likelihood_sum = 0
        for q in range(12):
            likelihood = 1

            for k in range(12):
                chroma_value = chroma_vector[:, (q + k) % 12]

                if template[k]:
                    likelihood *= chroma_value

            likelihood_sum += likelihood

        return likelihood_sum