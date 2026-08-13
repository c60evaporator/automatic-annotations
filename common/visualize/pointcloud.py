import open3d as o3d
from open3d.visualization.draw_plotly import get_plotly_fig
import plotly.graph_objects as go
import numpy as np

from ..geometry.transform import normalize_quaternion, quaternion_to_rotation_matrix


def plot_pointcloud(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    opacity: float = 0.4,
    point_size: float = 0.5,
    down_sample_size: int = None,
    fig_width: int = 640,
    fig_height: int = 480,
    title: str = None,
    show_axes: bool = True,
    axis_translation: np.ndarray | None = None,
    axis_quaternion: np.ndarray | None = None,
) -> go.Figure:
    """
    Visualize a point cloud using Open3D and Plotly.

    Args:
        points (np.ndarray): Nx3 array of point coordinates.
        colors (np.ndarray | None): Nx3 array of RGB colors (0-1 range) for each point. If None, points will be white.
        opacity (float): Opacity of the points in the plot.
        point_size (float): Size of the points in the plot.
        down_sample_size (int | None): If provided, the point cloud will be downsampled to this number of points by voxel downsampling. If None, no downsampling is applied.
        fig_width (int): Width of the plot canvas in pixels.
        fig_height (int): Height of the plot canvas in pixels.
        title (str): Title of the plot.
        show_axes (bool): Whether to show the axes lines in the plot. If False, axes will be hidden.
        axis_translation (np.ndarray | None): Position of the axis origin in
            the rendering coordinate system, as ``(x, y, z)``. Used only when
            ``show_axes=True``. If None, the origin is placed at ``(0, 0, 0)``.
        axis_quaternion (np.ndarray | None): Quaternion ``(w, x, y, z)``
            representing the rotation from the axis-local frame to the
            rendering coordinate system. Applied to both the rendered axes and
            the camera view direction. If None, no rotation is applied.

    Returns:
        go.Figure: A Plotly figure object for the point cloud visualization.
    """
    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points) 

    # Create colors for the points if provided, otherwise use a default color
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.paint_uniform_color([0.117647, 0.564706, 1.0])  # Default to mediumaquamarine if no colors provided

    # Downsample the point cloud if requested
    if down_sample_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size=down_sample_size)

    # Create a Plotly figure from the Open3D PointCloud
    fig = get_plotly_fig(
        [pcd],
        width=fig_width,
        height=fig_height,
    )

    # Calculate the rotation
    axis_rotation = np.eye(3, dtype=np.float64)
    if axis_quaternion is not None:
        axis_rotation = quaternion_to_rotation_matrix(
            normalize_quaternion(axis_quaternion)
        )
    # 軸の姿勢に合わせて、既定のカメラ位置と上方向も回転する。
    camera_eye = axis_rotation @ np.array([-1.0, -0.5, 1.5])
    camera_up = axis_rotation @ np.array([0.0, 0.0, 1.0])

    # Set the camera view and layout
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            camera=dict(
                eye=dict(
                    x=camera_eye[0],
                    y=camera_eye[1],
                    z=camera_eye[2],
                ),
                up=dict(
                    x=camera_up[0],
                    y=camera_up[1],
                    z=camera_up[2],
                ),
                center=dict(x=0, y=0, z=0,),
            ),
        ),
        title=title,
    )

    # Add axes if requested
    if show_axes:
        axis_length = 5.0  # [m]
        axis_origin = np.zeros(3, dtype=np.float64)
        if axis_translation is not None:
            axis_origin = np.asarray(axis_translation, dtype=np.float64)
            if axis_origin.shape != (3,):
                raise ValueError(
                    "axis_translation must have shape (3,), "
                    f"got {axis_origin.shape}"
                )

        # 軸方向だけを回転し、最後に軸原点の位置を加える。
        axis_vectors = axis_length * axis_rotation
        x_axis = axis_origin + axis_vectors[:, 0]
        y_axis = axis_origin + axis_vectors[:, 1]
        z_axis = axis_origin + axis_vectors[:, 2]
        fig.add_trace(
            go.Scatter3d(x=[axis_origin[0], x_axis[0]],
                         y=[axis_origin[1], x_axis[1]],
                         z=[axis_origin[2], x_axis[2]],
                         mode="lines",
                         line=dict(color="red", width=4),
                         name="X",
                         showlegend=False)
        )
        fig.add_trace(
            go.Scatter3d(x=[axis_origin[0], y_axis[0]],
                         y=[axis_origin[1], y_axis[1]],
                         z=[axis_origin[2], y_axis[2]],
                         mode="lines",
                         line=dict(color="green", width=4),
                         name="Y",
                         showlegend=False)
        )
        fig.add_trace(
            go.Scatter3d(x=[axis_origin[0], z_axis[0]],
                         y=[axis_origin[1], z_axis[1]],
                         z=[axis_origin[2], z_axis[2]],
                         mode="lines",
                         line=dict(color="blue", width=4),
                         name="Z",
                         showlegend=False)
        )
    
    # Update marker size and opacity for the point cloud
    fig.update_traces(
        marker=dict(
            size=point_size,
            opacity=opacity,
        ),
        selector=dict(type="scatter3d"),
    )
    return fig
