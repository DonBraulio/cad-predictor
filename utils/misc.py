def str_to_seconds(time_str):
    """
    Convert time string as HH:MM:ss to number of seconds as float
    """
    if not time_str:
        return 0.0
    elif time_str.count(":") != 2:
        raise ValueError(f"Time format is not HH:MM:ss in '{time_str}'")
    return sum(x * float(t) for x, t in zip([3600, 60, 1], time_str.split(":")))
