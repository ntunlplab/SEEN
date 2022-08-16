from enum import IntEnum as BaseIntEnum


class IntEnum(BaseIntEnum):
    @classmethod
    def labels(cls):
        return [v for _, v in sorted(cls._member_map_.items(), key=lambda x: x[1])]

    @classmethod
    def names(cls):
        return [k for k, _ in sorted(cls._member_map_.items(), key=lambda x: x[1])]


class EventLabel(IntEnum):
    consistent = 0
    additional = 1
    forgotten = 2
    inconsistent = 3
    unforgotten = 4


class PreRetoldLabel(IntEnum):
    forgotten = 0
    unforgotten = 1


class PostRetoldLabel(IntEnum):
    consistent = 0
    inconsistent = 1
    additional = 2
