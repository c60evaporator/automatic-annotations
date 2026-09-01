def get_skipped_sample_indices(num_samples, sample_interval):
    selected_sample_indices = list(range(0, num_samples, sample_interval))
    if selected_sample_indices and selected_sample_indices[-1] != num_samples - 1:
        selected_sample_indices.append(num_samples - 1)
    return selected_sample_indices

def get_sliding_window(num_samples, window_size, stride):
    """Create a sliding window indices of samples in a scene.
    Args:
        num_samples (int): The number of samples in the scene.
        window_size (int): The size of the sliding window.
        stride (int): The stride of the sliding window.
    Returns:
        window_ranges (list of tuple): A list of tuples, each containing the start and end indices of a sliding window.
        used_ranges (list of tuple): A list of tuples, each containing the start and end indices of the used range within each sliding window.
    """
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
