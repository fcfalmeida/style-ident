from src.data.pipelines.feature_pipeline import FeaturePipeline
from src.data.constants.feature_groups import HARM_RHYTHM_FEATS
from src.features.harm_rhythm import HarmRhythm


class HarmRhythmPipeline(FeaturePipeline):
    def __init__(self) -> None:
        super().__init__(
            [HarmRhythm()],
            HARM_RHYTHM_FEATS
        )
