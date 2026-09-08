import open3d as o3d
import numpy as np

def remove_radius_outliers(
    points: np.ndarray, nb_points: int, radius: float
) -> np.ndarray:
    """Radius Outlier Removal.

    半径 radius 内に nb_points 個の近傍を持たない点を落とす。
    深度推定の点群は物体の輪郭から尾を引くように誤差が出るため、
    クラスタリングの前段でこれを落としておく。
    """
    if points.shape[0] == 0 or nb_points < 1:
        return points

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    _, keep = cloud.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return points[np.asarray(keep, dtype=np.int64)] if keep else np.empty((0, 3))


def largest_dbscan_cluster(
    points: np.ndarray, eps: float, min_samples: int
) -> np.ndarray:
    """DBSCAN で最大クラスタだけを残す.

    マスクが背景を巻き込むと、対象物体の手前や奥に別の塊ができる。
    最大クラスタに絞ることで、尾状に伸びたノイズを落とす。
    """
    if points.shape[0] == 0:
        return points

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    labels = np.asarray(
        cloud.cluster_dbscan(eps=eps, min_points=min_samples, print_progress=False)
    )
    valid = labels >= 0
    if not valid.any():
        # すべてノイズ判定なら、絞り込まずに返す（点を失うより情報を残す）
        return points
    counts = np.bincount(labels[valid])
    return points[labels == int(np.argmax(counts))]
