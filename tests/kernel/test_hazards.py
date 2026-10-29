import pytest

from physrisk.kernel.hazards import Wind, hazard_class


def test_hazard_class_returns_only_hazard_types():
    assert hazard_class("Wind") is Wind

    with pytest.raises(AttributeError, match="unknown hazard class"):
        hazard_class("HazardKind")
