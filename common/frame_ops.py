def create_sliding_windows(num_samples, window_size, stride):
    """Split samples into sliding windows so that each sample is used exactly once.

    Returns:
        window_ranges: (start, end) of each window (absolute indices, end exclusive)
        used_ranges: (start, end) of the samples used from each window (relative to the window)
    """
    if window_size < 1 or not 1 <= stride <= window_size:
        raise ValueError(
            f"window_size must be >= 1 and stride must be in [1, window_size]: "
            f"window_size={window_size}, stride={stride}"
        )
    if num_samples <= 0:
        return [], []
    if num_samples <= window_size:
        return [(0, num_samples)], [(0, num_samples)]

    left_context = (window_size - stride) // 2
    last_start = num_samples - window_size
    window_start_indices = list(range(0, last_start + 1, stride))
    if window_start_indices[-1] != last_start:
        window_start_indices.append(last_start)

    window_ranges = [(i_start, i_start + window_size) for i_start in window_start_indices]
    used_ranges = []
    used_start = 0
    for window_count, i_start in enumerate(window_start_indices):
        if window_count == len(window_start_indices) - 1:
            used_end = num_samples
        else:
            used_end = i_start + left_context + stride
        used_ranges.append((used_start - i_start, used_end - i_start))
        used_start = used_end

    return window_ranges, used_ranges
