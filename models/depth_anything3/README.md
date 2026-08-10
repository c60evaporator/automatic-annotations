## Installation



## 解説

### pseudo-LiDAR化

各画素 ((u,v)) とdepth (z) を、カメラ内部パラメータで3D点（カメラ座標）へ戻します。

```math
X=\frac{(u-c_x)z}{f_x},\qquad
Y=\frac{(v-c_y)z}{f_y},\qquad
Z=z
```

SAMマスク内部の点だけを取り出し、3D boxをfitします。

ただしマスク境界には背景深度が混ざるため、以下が重要です。

* マスクのerosion
* depth discontinuityによる境界除去
* 中央領域を優先
* depth中央値または下位分位点の利用
* ground planeとの接点補正

OVM3D-Detも、open-vocabulary 2Dモデルとpseudo-LiDARで3D pseudo-labelを生成し、ノイズの多いboxに対してpseudo-LiDAR erosionとサイズpriorによる補正を導入しています。([arXiv][9])
