import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import numpy as np

from ..visualize.pointcloud import plot_pointcloud

def plot_depth_map(
    depth_map: np.ndarray,
    platform: str = "plotly",
    ax = None,
    fig = None,
    cmap: str = "turbo",
    title: str = None,
    unit: str | None = "m"
) -> go.Figure | plt.Axes:
    """
    Visualize a depth map using Plotly.

    Args:
        depth_map (np.ndarray): HxW array of depth values.
        platform (str): Visualization platform, either 'matplotlib' or 'plotly'.
        ax: Matplotlib axis object (used if platform is 'matplotlib').
        fig: Matplotlib figure object (used if platform is 'matplotlib').
        cmap (str): Colormap for visualization.
        title (str): Title of the plot.
        unit (str | None): Unit of depth values, e.g., 'm' for meters. If None, no unit will be displayed.

    Returns:
        go.Figure: A Plotly figure object for the depth map visualization.
    """
    H, W = depth_map.shape

    if platform == "matplotlib":
        if fig is not None:
            raise ValueError("If platform is 'matplotlib', fig must be None.")

        if ax is None:
            ax = plt.gca()

        im = ax.imshow(depth_map, cmap=cmap)
        ax.axis("off")
        if title:
            ax.set_title(title)
        plt.colorbar(im, ax=ax)
        return ax

    elif platform == "plotly":
        if ax is not None:
            raise ValueError("If platform is 'plotly', ax must be None.")

        if fig is None:
            fig = go.Figure()
        
        fig.add_trace(
            go.Heatmap(
                z=depth_map,
                x=np.arange(W),
                y=np.arange(H),
                colorscale=cmap.capitalize(),
                hovertemplate=(
                    "x: %{x}<br>"
                    "y: %{y}<br>"
                    f"depth: %{{z:.3f}} {unit if unit else ''}"
                    "<extra></extra>"
                ),
                colorbar=dict(title=f"Depth ({unit})" if unit else "Depth"),
            )
        )
        fig.update_xaxes(constrain="domain")
        fig.update_yaxes(autorange="reversed",
                         scaleanchor="x",
                         scaleratio=1)
        fig.update_layout(
            xaxis_title="x [pixel]",
            yaxis_title="y [pixel]",
        )
        return fig

def plot_depth_with_original_image(
    depth_map: np.ndarray,
    original_image: np.ndarray,
    platform: str = "plotly",
    cmap: str = "turbo",
    title: str = None,
    unit: str | None = "m",
):
    """
    Visualize a depth map alongside the original image using Plotly.

    Args:
        depth_map (np.ndarray): HxW array of depth values.
        original_image (np.ndarray): HxWx3 array of RGB values for the original image.
        platform (str): Visualization platform, either 'matplotlib' or 'plotly'.
        cmap (str): Colormap for the depth map visualization.
        title (str): Title of the plot.
        unit (str | None): Unit of depth values, e.g., 'm' for meters. If None, no unit will be displayed.
    """
    if platform == "matplotlib":
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(original_image)
        axes[0].axis("off")
        axes[0].set_title("Original Image")
        plot_depth_map(depth_map, platform="matplotlib", ax=axes[1], cmap=cmap, title="Depth Map", unit=unit)
        fig.suptitle(title if title else "Depth Map with Original Image")
        plt.tight_layout()

    if platform == "plotly":
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Original Image", "Depth Map"))
        # Original image
        fig.add_trace(go.Image(z=original_image),row=1, col=1)
        
        # Depth map
        depth_fig = plot_depth_map(depth_map, platform="plotly", cmap=cmap, title="Depth Map", unit=unit)
        for trace in depth_fig.data:
            fig.add_trace(trace, row=1, col=2)
        
        # Add axis settings
        fig.update_xaxes(constrain="domain", row=1, col=1)
        fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1, row=1, col=1)
        fig.update_xaxes(constrain="domain", row=1, col=2)
        fig.update_yaxes(autorange="reversed", scaleanchor="x2", scaleratio=1, row=1, col=2)
        
        fig.update_layout(title_text=title if title else "Depth Map with Original Image",
                          width=1000, height=500)
        fig.show()


def plot_pseudo_lidar_with_ground_truth(
    pseudo_lidar_points: np.ndarray,
    ground_truth_points: np.ndarray,
    pseudo_lidar_colors: list[float] = [1.0, 0.0, 0.0],
    ground_truth_colors: list[float] = [0.117647, 0.564706, 1.0],
    axis_translation: np.ndarray | None = None,
    axis_quaternion: np.ndarray | None = None,
) -> go.Figure:
    """
    Visualize pseudo-LiDAR points and ground truth points in a 3D plot.

    Args:
        pseudo_lidar_points (np.ndarray): Nx3 array of pseudo-LiDAR with global coordinates.
        ground_truth_points (np.ndarray): Mx3 array of ground truth points with global coordinates.
        pseudo_lidar_colors (list[float], optional): RGB color for pseudo-LiDAR points. Defaults to [1.0, 0.0, 0.0].
        ground_truth_colors (list[float], optional): RGB color for ground truth points. Defaults to [0.117647, 0.564706, 1.0].
        axis_translation (np.ndarray | None): Position of the axis origin in
            the rendering coordinate system, as ``(x, y, z)``. Used only when
            ``show_axes=True``. If None, the origin is placed at ``(0, 0, 0)``.
        axis_quaternion (np.ndarray | None): Quaternion ``(w, x, y, z)``
            representing the rotation from the axis-local frame to the
            rendering coordinate system. Applied to both the rendered axes and
            the camera view direction. If None, no rotation is applied.

    Returns:
        go.Figure: A Plotly 3D figure containing the pseudo-LiDAR and ground truth points.
    """
    points = np.vstack((pseudo_lidar_points, ground_truth_points))
    colors = np.vstack((np.tile(pseudo_lidar_colors, (pseudo_lidar_points.shape[0], 1)),
                        np.tile(ground_truth_colors, (ground_truth_points.shape[0], 1))))
    plot_fig = plot_pointcloud(points, colors=colors, opacity=0.8, point_size=1.0,
                              axis_translation=axis_translation,
                              axis_quaternion=axis_quaternion)

    return plot_fig
