"""webapp と inference で共有するモジュール.

プロジェクトルートの `common/` を両コンテナへ read-only でマウントし、
PYTHONPATH に親ディレクトリ（/opt/shared）を通して import する。

  from common.mask_rle import decode_rle
  from common.box_ops import box_iou

ここに置くのは「両方のコンテナで実際に使うもの」だけにすること。
片側でしか使わない処理（推論の NMS、トラック照合など）を持ち込むと、
共有コードが肥大して変更の影響範囲が読めなくなる。
"""
