"""pycolmap のスタブ（Dockerfile_cpu と scripts/setup_mac.sh で使う）.

pycolmap には aarch64 Linux 向けの wheel も sdist も無く、macOS の wheel は torch と別の
libomp を同梱しているため、torch と同じプロセスに読み込むと OMP: Error #15 で落ちる。
Depth-Anything-3 は COLMAP 出力でしか使わないが、depth_anything_3.api の
import 時に読み込まれるため、import だけ通るようにする。
属性へアクセスした時点で ImportError を送出する。
"""


def __getattr__(name: str):
    raise ImportError(f"pycolmap is not installed in this image (pycolmap.{name})")
