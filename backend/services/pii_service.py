from typing import Optional

from ..pii.config import PseudonymConfig
from ..pii.pipeline import RedactionPipeline

_pipeline: Optional[RedactionPipeline] = None


def get_pii_pipeline() -> RedactionPipeline:
    global _pipeline
    if _pipeline is None:
        config = PseudonymConfig.from_env()
        _pipeline = RedactionPipeline(config)
    return _pipeline
