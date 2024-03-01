from dataclasses import dataclass, fields
from enum import Enum
from typing_extensions import Any, Dict
import sys

py310 = sys.version_info.minor >= 10 or sys.version_info.major > 3

class RecordType(str, Enum):
    PAIR = 'pair'
    TRIPLET = 'triplet'
    SCORED_PAIR = 'scored_pair'


@dataclass(**({"slots": True} if py310 else {}))
class PairRecord:
    text: str
    text_pos: str


@dataclass(**({"slots": True} if py310 else {}))
class TripletRecord:
    text: str
    text_pos: str
    text_neg: str


@dataclass(**({"slots": True} if py310 else {}))
class ScoredPairRecord:
    sentence1: str
    sentence2: str
    label: float


# * Order matters
record_type_cls_map: Dict[RecordType, Any] = {
    RecordType.SCORED_PAIR: ScoredPairRecord,
    RecordType.TRIPLET: TripletRecord,
    RecordType.PAIR: PairRecord,
}


def infer_record_type(record: dict) -> RecordType:
    record_type_field_names_map = {
        record_type: [field.name for field in fields(record_cls)] for record_type, record_cls in record_type_cls_map.items()
    }
    for record_type, field_names in record_type_field_names_map.items():
        if all(field_name in record for field_name in field_names):
            return record_type
    raise ValueError(f'Unknown record type, record: {record}')
