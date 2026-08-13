from os import PathLike

import torch

from video_depth_anything.video_depth import VideoDepthAnything


def load_model(encoder_name: str,
               checkpoint_dir: str | PathLike[str],
               metric: bool = True,
               device: str = "cuda") -> VideoDepthAnything:
    """
    Load the VideoDepthAnything model from the specified path.

    Args:
        encoder_name (str): Name of the encoder ('vits', 'vitb', 'vitl').
        checkpoint_dir (str | PathLike[str]): Path to the directory containing the model checkpoint file.
        metric (bool): Whether to load the model with metric scale or not. Default is True.
        device (str): Device to load the model on ('cuda' or 'cpu').

    Returns:
        VideoDepthAnything: Loaded model instance.
    """
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    checkpoint_name = 'metric_video_depth_anything' if metric else 'video_depth_anything'

    video_depth_anything = VideoDepthAnything(**model_configs[encoder_name], metric=metric)
    video_depth_anything.load_state_dict(torch.load(f'{checkpoint_dir}/{checkpoint_name}_{encoder_name}.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(device)

    return video_depth_anything
