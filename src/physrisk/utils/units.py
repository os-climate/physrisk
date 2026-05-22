ADIM_UNITS = {"index"}


def needs_conversion(source_units: str, target_units: str | None) -> bool:
    if source_units == "default" or target_units in (None, "default"):
        return False

    if source_units == target_units:
        return False

    if source_units in ADIM_UNITS or target_units in ADIM_UNITS:
        raise ValueError(
            f"Cannot convert between incompatible units: {source_units} and {target_units}"
        )

    return True
