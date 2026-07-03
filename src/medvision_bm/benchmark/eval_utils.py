def parse_sample_indices(s: str) -> list:
    """Parse a ``--sample_indices`` string into a list of integer indices.

    Two formats are accepted:

    - ``[start:stop]`` -> ``list(range(start, stop))``
    - ``[start,stop]`` or ``[start,stop,step]`` ->
      ``list(range(start, stop[, step]))``

    Surrounding square brackets are optional and stripped before parsing.

    Args:
        s (str): The raw ``--sample_indices`` argument, e.g. ``"[0:10]"`` or
            ``"0,10,2"``.

    Returns:
        list: The list of integer indices described by ``s``.

    Raises:
        ValueError: If the string matches neither accepted format (a colon form
            without exactly two parts, or a comma form without two or three
            parts).
    """
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if ":" in s:
        parts = s.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid sample_indices format '{s}'. Use [start:stop] or [start,stop,step]."
            )
        return list(range(int(parts[0]), int(parts[1])))
    else:
        parts = [int(x) for x in s.split(",")]
        if len(parts) not in (2, 3):
            raise ValueError(
                f"Invalid sample_indices format '{s}'. Use [start:stop] or [start,stop,step]."
            )
        return list(range(*parts))
