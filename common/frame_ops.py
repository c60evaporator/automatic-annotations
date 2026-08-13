def create_sliding_windows(num_samples, window_size, stride):
    # Create a sliding windows
    left_context = (window_size - stride) // 2
    if num_samples <= window_size:
        window_start_indices = [0]
    else:
        last_start = num_samples - window_size
        window_start_indices = list(
            range(0, last_start + 1, stride)
        )
        if window_start_indices[-1] != last_start:
            window_start_indices.append(last_start)
    window_ranges = [(i_start, i_start + window_size) for i_start in window_start_indices]
    used_ranges = []
    for window_count, i_start in enumerate(window_start_indices):
        if window_count == 0:
            used_ranges.append((0, left_context + stride))
        elif window_count == len(window_start_indices) - 1:
            used_ranges.append((left_context, window_size))
        else:
            used_ranges.append((left_context, left_context + stride))
    
    return window_ranges, used_ranges
