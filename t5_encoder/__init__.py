from transformers.models.auto.modeling_auto import (
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    _LazyAutoMapping
)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

from .modeling_mt5 import MT5ForTokenClassification, MT5ForSequenceClassification

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES["mt5"] = "MT5ForTokenClassification"
MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES["mt5"] = "MT5ForSequenceClassification"

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
)

MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
)

from transformers.models import mt5

setattr(mt5, "MT5ForTokenClassification", MT5ForTokenClassification)
setattr(mt5, "MT5ForSequenceClassification", MT5ForSequenceClassification)

__version__ = "0.1"

__all__ = [
    "MT5ForTokenClassification",
    "MT5ForSequenceClassification"
]
