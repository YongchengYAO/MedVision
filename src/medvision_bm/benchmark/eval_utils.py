def parse_sample_indices(s: str) -> list:
    """Parse --sample_indices string into a list of integer indices.

    Accepted formats:
      [idx1:idx2]          -> list(range(idx1, idx2))
      [idx1,idx2,interval] -> list(range(idx1, idx2, interval))
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
