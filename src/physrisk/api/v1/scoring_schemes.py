from enum import Enum


class Category(int, Enum):
    ERROR = -3  # an unexpected error: should not happen
    NO_DATA = -2  # no data available to determine vulnerability, e.g. scenario/year selected for which there is no coverage for asset and hazard combination
    NO_VULNERABILITY = -1  # for the type of asset, the model assumes no vulnerability to the hazard, e.g. model assumes real estate asset has no vulnerability to chronic heat (if shrink-swell subsidence tackled separately)
    VERY_LOW = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERY_HIGH = 4


def map_to_original_category(score: int) -> int:
    if score in [Category.ERROR, Category.NO_DATA, Category.NO_VULNERABILITY]:
        return int(OriginalCategory.NODATA)
    elif score == Category.VERY_LOW:
        return int(OriginalCategory.LOW)
    elif score == Category.LOW:
        return int(OriginalCategory.LOW)
    elif score == Category.MEDIUM:
        return int(OriginalCategory.MEDIUM)
    elif score == Category.HIGH:
        return int(OriginalCategory.HIGH)
    elif score == Category.VERY_HIGH:
        return int(OriginalCategory.REDFLAG)
    else:
        raise ValueError(f"Unexpected score value: {score}")


# original scoring scheme for reference
class OriginalCategory(int, Enum):
    NODATA = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    REDFLAG = 4
