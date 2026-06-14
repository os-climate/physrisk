UNITLESS = {"index", ""}


def needs_conversion(source_units: str, target_units: str | None) -> bool:
    """Return whether values should be converted between two unit labels.

    Units marked as ``default`` and missing target units do not require
    conversion. Equal units also do not require conversion. Unitless indicators (e.g. index)
    cannot be converted to or from dimensional units, and attempting to do so raises an error.

    Args:
        source_units: Units attached to the source values.
        target_units: Units expected by the target consumer, or ``None`` if
            unspecified.

    Returns:
        ``True`` when unit conversion should be attempted, otherwise ``False``.

    Raises:
        ValueError: If source and target are provided, source is not ``default``, and only one is unitless.
    """
    if source_units == "default" or target_units in (None, "default"):
        return False

    if source_units == target_units:
        return False

    if source_units in UNITLESS or target_units in UNITLESS:
        raise ValueError(
            f"Conversion is not defined between incompatible units: {source_units} and {target_units}"
        )

    return True
