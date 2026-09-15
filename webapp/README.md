## Prerequisites

- Docker & Docker Compose
- Chrome or Firefox browser
- nuScenes formatted dataset

## Getting Started

### 1. Create `.env`

以下の.envファイルをnuscenes-annotatorフォルダ直下に作成してください

```.env
# ホストサーバーのUID/GIDをidコマンドで調べて記載してください
HOST_UID=1000
HOST_GID=1000
HTTP_PROXY=[Your proxy path]
HTTPS_PROXY=[Your proxy path]
HOST_DATA_ROOT=

# ホストサーバーのGPUに対応するCUDAアーキテクチャリストを記載 (参考https://en.wikipedia.org/wiki/CUDA#GPUs_supported).
TORCH_CUDA_ARCH_LIST=8.9


# 使用する物理GPU番号（nvidia-smi の index）。コンテナ内では常に 0 に振り直される。シングルGPUなら0を指定
GPU_DEVICE_ID=0
```

特に`GPU_DEVICE_ID`は使用したいGPU番号に合わせて適宜変更してください

### 2. データセットを準備

.envの`HOST_DATA_ROOT`に指定したフォルダの下に、以下のようにnuScenes形式のデータセットを設置します（`<dataroot>`は好きなフォルダ名でOKで、後でデータインポート時にフォルダ名を使用する。複数の`<dataroot>`フォルダを作ることも可能）。

```
HOST_DATA_ROOT/<dataroot>/
    ├── v1.0-mini/            ← --version で指定。ここに *.json がある。v1.0-trainvalでもOK
    │     ├── log.json, scene.json, sample.json, ...
    ├── samples/              ← キーフレームのセンサーデータ
    ├── sweeps/               ← 非キーフレーム
    └── maps/
        ├── <hash>.png            ← basemap
        └── expansion/            ← Map Expansion（任意）
            └── boston-seaport.json
```

### 3. 重みダウンロード

GroundingDINOとSAM2の重みを、`nuscenes-annotator/checkpoints`フォルダにダウンロードします（DA3の重みはHuggingFaceから自動ダウンロードされるので手動でのダウンロードは不要です）

```bash
cd nuscenes-annotator/checkpoints
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### 4. コンテナビルド

以下コマンドでコンテナをビルドします（TORCH_CUDA_ARCH_LISTが間違っているとビルドは通るがコンテナ内でのGPU推論が無効化されて激遅になるので注意してください）

```bash
cd nuscenes-annotator
docker compose build --no-cache
```

### 5. データインポート

以下コマンドでJSON形式のメタデータをDBにインポートします。`<dataroot>`は先ほど指定したデータセット全体が格納されたフォルダ名、`<version>`はその下層のjsonファイルが格納されたフォルダ名（例: `v1.0-mini`）

```bash
docker compose run --rm webapp \
    python -m app.json_conversion.to_nusc_db \
    --name <好きなデータセット名> --version <version> --dataroot <dataroot>
```

## Usage

以下コマンドでアプリ立ち上げ

```bash
cd nuscenes-annotator
docker compose up
```

Chrome, Firefox等のブラウザで、サーバーのアドレス（ローカルホストの場合`http://localhost`。デフォルトではポート80=ポート番号指定不要）を開く

画面は以下の6つのページ（左側のタブで切替）に分かれている。基本的には上から下のタブに向かって操作を実施していけばOK

|ページ名|役割|
|---|---|
|main|データセットを選択する|
|Select Scene|アノテーション対象のシーンを選択する|
|Detect BBoxes|Grounding DINOで各物体のバウンディングボックスとラベルを推定する|
|Track Instances|SAM2で各物体のインスタンスマスクとトラッキングIDを推定する|
|Estimate Depth|Depth-Anything-3で単眼深度推定を実施してインスタンスごとの点群を取得し、この点群に3Dバウンディングボックスをフィッティングする|
|Export|アノテーション結果を含むメタデータをJSON形式でエクスポートする|

### mainページ

データセットを選択するページ。中央のテーブルからアノテーションしたいデータセットを選んで「このデータセットを使う」で確定。
ここで選択されたデータセットは、画面リロードされるまでセッションで維持される

### Select Sceneページ

シーンを選択するページ。中央のテーブルからアノテーションしたいシーンを選んで「このシーンをアノテーション」で確定
ここで選択されたシーンは、画面リロードされるまでセッションで維持される

### Detect BBoxesページ

Grounding DINOで各物体のバウンディングボックスとラベルを推定するページ。

#### Grounding DINOの推論フローとパラメータ

- Sample Interval間隔のキーフレーム → カメラ → カテゴリグループ[^category_group]のループで以下を実施
    - カテゴリグループ内の各ラベル[^nusc_label]に前処理[^groundingdino_preprocess]（アンダースコアをスペースに置き換える等）を実施してリスト化
    - ラベルのリストをプロンプトとして渡してGrounding DINOで推論

UI上では、「Inference Parameters」エクスパンダーで以下パラメータを変更可能

|名称|内容|デフォルト値|
|---|---|---|
|Sample Interval|推論を実施するキーフレームの間隔|4（2秒）|
|Score Threshold|推論したバウンディングボックスを採択するスコアの閾値（実際にはカテゴリグループごとの`DET2D_DEFAULT_SCORE_THRESHOLDS`に掛けた値が使用される）|1.0|
|NMS Threshold|NMSでバウンディングボックスを合体するIoUの閾値（同一ラベルのNMSはカテゴリグループごとの`DET2D_NMS_SAME_CLASS_IOUS`に掛けた値が使用され、別ラベルのNMSは`DET2D_NMS_CROSS_CLASS_IOU`に掛けた値が使用される）|1.0|

「Run Inference」ボタンを押すと推論が実行される。初回実行時はモデルのロードに時間が掛かる（1分ほど）ことに注意

[^category_group]: 似たラベルをグルーピングしてこのカテゴリグループごとに一括推論している。カテゴリグループは`nuscenes-annotator/webapp/app/core/config.py`の`LABEL_TO_CATEGORY_GROUP`で指定可能

[^nusc_label]: 判定対象のラベルとnuScenesカテゴリとの変換は`nuscenes-annotator/webapp/app/core/config.py`の`NUSC_CATEGORY_TO_LABEL`、`LABEL_TO_NUSC_CATEGORY`で指定可能。

[^groundingdino_preprocess]: GroundingDINOはトークナイザの扱いがシビアなので、実装上の工夫をいくつか加えている。詳細は`nuscenes-annotator/inference/app/models_impl/groundingdino_predict.py`参照

#### バウンディングボックス推論結果の表示UI

推論結果はDBに保存され、「保存済みの推論結果」エクスパンダーで選択可能。選択した推論結果がページ下部のカメラ画像上に表示される。右ペインで以下のように表示を変えられる

|名称|内容|
|---|---|
|Show boxesチェックボックス|バウンディングボックスのカメラ画像上への表示有無|
|Compare with GTチェックボックス|チェックするとGround Truthと推論ボックスのプロット画像が左右に並べて表示される。チェックしないと推論ボックスのみが表示される|
|Min scoreスライダー|このScoreを超えたバウンディングボックスのみが画像上に表示される|
|Box textラジオボタン|バウンディングボックスの上にテキスト表示される内容（"Label"ならラベル、"Score"なら推論スコア、Noneなら表示なし）|

### Track Instancesページ

SAM2で各物体のインスタンスマスクとトラッキングIDを推定するページ。プロンプトとして前のページで検出したバウンディングボックスを使用

#### SAM2の推論フローとパラメータ

推論の流れは以下

- カメラ → Sample Interval間隔のキーフレームのループで以下を実施
    - 入力データにはDetect BBoxesページで指定したSample Interval分のキーフレームに加え、キーフレーム間を後述の`Sweeps per Sample`で当分割した非キーフレーム画像も挿入して使用。
    - 入力データの最初の画像に、Detect BBoxesページで検出した各物体のバウンディングボックスをプロンプトとして渡す
    - 全入力データに対してトラッキング推論（伝播）を実施し、インスタンスマスクと暫定トラッキングIDを得る
    - 今回の推論の最初の画像のインスタンスマスクと、前回の推論（今回の推論がSample4～8のデータを用いている場合、前回の推論はSample0～4のデータを用いた推論が該当）の最後のキーフレームに伝播されたインスタンスマスクを比較し、IoUが一定以上であればトラッキングIDを引き継いで割り振りなおす。IoUが一定以上のインスタンスマスクが存在しなければ新たなトラッキングIDを割り振る

「Inference Parameters」エクスパンダーで以下パラメータを変更可能

|名称|内容|デフォルト値|
|---|---|---|
|Sweeps per Sample|トラッキングの入力データに使用するキーフレームあたりのフレーム数（キーフレーム自身も含む）|2|
|IoU Threshold|前回の推論と比較してトラッキングIDを引き継ぐためのIoUの下限|0.5|
|IoU Method|上記トラッキングID引継ぎIoU計算にマスクと外接バウンディングボックスどちらを用いるか|"box"（外接バウンディングボックス）|
|IoU Label Match|上記トラッキングID引継ぎIoU判定に、ラベルの一致を条件として加えるか（"label"ならラベル一致必須、"category_group"ならカテゴリグループ一致が条件、"none"ならラベル一致は見ずIoUのみで判定）|"label"|

「Run Inference」ボタンを押すと推論が実行される。初回実行時はモデルのロードに時間が掛かる（1分ほど）ことに注意

#### インスタンスセグメンテーション・トラッキング結果の表示UI

推論結果はDBに保存され、「保存済みの推論結果」エクスパンダーで選択可能。選択した推論結果がページ下部のカメラ画像上に表示される。右ペインで以下のように表示を変えられる

|名称|内容|
|---|---|
|Compare propagationチェックボックス|チェックすると前の推論の最後の画像の伝播インスタンスマスクと、今回の推論の最初の画像のインスタンスマスクを左右に並べて表示（トラッキングIDの引継ぎのIoU判定を視覚的に確認できる。Sample Intervalごとのボックスプロンプトを与えたキーフレームのみが表示対象となる）。チェックしないと今回の推論結果のみが表示される|
|Show boxesラジオボタン|画像上に表示するバウンディングボックスの種類（"Prompt"なら与えたボックスプロンプト、"Instance"ならインスタンスマスクの外接矩形、Noneなら表示なし）|
|Colorラジオボタン|インスタンスマスク・バウンディングボックスの表示色の決め方（"Label"ならラベル、"Track ID"ならトラッキングID）|
|Instance textラジオボタン|インスタンスマスクの上にテキスト表示される内容（"Label"ならラベル、"Track ID"ならトラッキングID、Noneなら表示なし）|

### Estimate Depthページ

Depth-Anything-3で単眼深度推定を実施してインスタンスごとの点群を取得し、この点群に3Dバウンディングボックスをフィッティングするページ

#### Depth-Anything-3＆Box Fittingの推論フローとパラメータ

推論の流れは以下

- カメラ → キーフレームのループで以下の単眼深度推定を実施
    - キーフレーム画像を入力してDepth-Anything-3で推論し、深度画像（DA3モデルによりリサイズされている）を得る
    - 得た深度画像を元の画像のサイズのスケールに変換する
- キーフレーム → カメラのループで以下のBox Fittingを実施
    - 各インスタンスマスクに前処理（クロージング処理）を実施
    - 深度画像を各インスタンスマスクで切り出す
    - 切り出した深度画像をカメラパラメータを用いて点群に変換し、インスタンスごとの点群を得る
    - インスタンスごとの点群に対してROR → DBSCAN（最大クラスタ以外を削除）を適用してノイズ除去
    - [こちらの方法](https://arxiv.org/pdf/2302.01034)でインスタンスごと点群に3D bounding boxをフィッティング
    - 3D bounding boxの高さは、インスタンスごと点群のz_percentilesパラメータで指定したZ軸の上位・下位パーセント点で決める

「Inference Parameters」エクスパンダーで以下パラメータを変更可能（タブで分けられている）

|タブ|名称|内容|デフォルト値|
|---|---|---|---|
|General|Dilation|インスタンスマスクのクロージング処理の膨張カーネルサイズ|5|
|General|Erosion|インスタンスマスクのクロージング処理の収縮カーネルサイズ|5|
|Depth Estimation|ROR nb_points|RORノイズ除去処理のnb_pointsパラメータ（大きいほど削除される点群が増える）|8|
|Depth Estimation|ROR Radius|RORノイズ除去処理のradiusパラメータ（小さいほど削除される点群が増える）|0.8|
|Depth Estimation|DBSCAN eps|DBSCANノイズ除去処理のepsパラメータ（小さいほど削除される点群が増える）|1.0|
|Depth Estimation|DBSCAN min_samples|DBSCANノイズ除去処理のmin_samplesパラメータ（大きいほど削除される点群が増える）|12|
|LiDAR Pointcloud|LiDAR Sweeps|表示するLiDAR点群のキーフレームあたりsweep数（LiDAR点群は表示にのみ使用しておりアノテーションには使用していない）|5|
|BOX Fitting|Method|Box Fittingに使用する手法（現状1種類しかない）|"convex_hull_moa"|
|BOX Fitting|angle_step_deg|Box Fittingアルゴリズムでyaw角度推定する際の損失計算の間隔（小さいほど精度上がるが所要時間が増える）|0.5|
|BOX Fitting|z_percentiles|Boxの高さ推定に使用するパーセント点（上下両方指定）|[1.0, 99.0]|

「Run Inference」ボタンを押すと推論が実行される。初回実行時はモデルのロードに時間が掛かる（1分ほど）ことに注意

#### Depth-Anything-3＆Box Fitting結果の表示UI

推論結果はDBに保存され、「保存済みの推論結果」エクスパンダーで選択可能。選択した推論結果がページ下部のUI表示部に表示される。

UI表示部は以下の3つのタブで切り替えられる

##### Depth Estimationタブ

推定した深度画像を6カメラ分表示（容量削減のため保存時に縮小リサイズしているため少し表示が粗い）

右ペインで以下のように表示を変えられる

|名称|内容|
|---|---|
|Show masksラジオボタン|深度画像上にインスタンスマスクを重ねるかどうかを指定（"None"ならマスク表示なし、"Orignal"ならクロージング処理前のマスクを表示、"Closed"ならクロージング処理後のマスクを表示）|
|Mask Colorラジオボタン|インスタンスマスクの表示色の決め方。Show Masksが"None"以外のとき有効（"Label"ならラベル、"Track ID"ならトラッキングID）|
|Mask textラジオボタン|インスタンスマスクの上にテキスト表示される内容。Show Masksが"None"以外のとき有効（"Label"ならラベル、"Track ID"ならトラッキングID、Noneなら表示なし）|

##### PointCloudタブ

深度画像をプロジェクションして得られたインスタンスごと点群を、点群ビュー上に3D表示（点群は生点群をすべて表示すると重いため、点数に合わせてボクセルサンプリングで間引いて表示していることに注意）

右ペインで以下のように表示を変えられる

|名称|内容|
|---|---|
|Raw LiDARチェックボックス|LiDAR点群を灰色で重ねて表示するかどうか|
|LiDAR Groundチェックボックス|LiDAR点群の地面判定（Patchwork++）された部分を茶色で重ねて表示するかどうか|
|Raw Depthチェックボックス|深度画像からプロジェクションした全点群を水色で重ねて表示するかどうか|
|Depth Instacnesチェックボックス|深度画像からプロジェクションしたインスタンスごと点群を表示するかどうか（このビューの表示のメインとなる）|
|Instance Pointsラジオボタン|インスタンスごと点群の表示対象。Filter ParamsエクスパンダーでROR/DBSCANを再実行したときのみ有効※（"Raw"なら元のインスタンスごと点群、"ROR/DBSCAN"ならROR/DBSCAN適用後の点群）|
|Instance Colorラジオボタン|インスタンスごと点群の表示色の決め方（"Label"ならラベル、"Track ID"ならトラッキングID）|
|Camerasチェックボックス|チェックしたカメラのインスタンスごと点群のみが表示される|

※ROR/DBSCANを再実行していないときは、推論時パラメータでのROR/DBSCANが適用された点群が表示される

以下の3つのボタンで、視点を切り替えられる

|名称|視点|
|---|---|
|Global Viewボタン|車両の向きに関係なくグローバル座標に固定された視点方向で表示（グローバル座標に対する方向は`nuscenes-annotator/webapp/app/core/config.py`の`DEFAULT_GLOBAL_POINTCLOUD_EYE`と`DEFAULT_GLOBAL_POINTCLOUD_UP`で指定可能）|
|Top Viewボタン|車両の真上から見たBEV視点（恐らくこれが一番使いやすい）|
|Forward Viewボタン|車両の前方向に向かってみた視点|

また、Filter Paramsエクスパンダー内のパラメータを切り替えて「Apply」を押すことで、ROR、DBSCANによる（自動アノテーションに適用するためには、画面最上部のInference Parametersエクスパンダーでパラメータを再指定して「Run Inference」ボタンで推論再実行が必要）

##### Box Fittingタブ

インスタンスごと点群に[こちらの方法](https://arxiv.org/pdf/2302.01034)でyaw角度推定した3Dバウンディングボックスを、BEV視点で表示

右ペインで以下のように表示を変えられる

|名称|内容|
|---|---|
|Compare with GTチェックボックス|チェックした場合、Ground Truthのバウンディングボックスと推定されたバウンディングボックスが左右に並べて表示される。チェックしていない場合、推定されたバウンディングボックスのみが表示される|
|LiDAR Groundチェックボックス|LiDAR点群の地面判定（Patchwork++）された部分を茶色で重ねて表示するかどうか|
|Depth Instacnesチェックボックス|深度画像からプロジェクションしたインスタンスごと点群を一緒に表示するかどうか|
|Convex-Hullチェックボックス|yaw角度推定の途中経過として使用するConvex-hull多角形を表示するかどうか|
|Instance Colorラジオボタン|バウンディングボックスおよび点群の表示色の決め方（"Label"ならラベル、"Track ID"ならトラッキングID）|

### Exportページ

- アノテーション結果を含むメタデータをJSON形式でエクスポートするページ
- 選択したシーンのみエクスポートするか、データセット全体をエクスポートするか選べる
- エクスポート結果にはアノテーション（sample_annotation.jsonおよびinstance.json）だけでなく、他のすべてのメタデータを含む
- 「エクスポート」ボタンを押すとjsonに書き出され、「ダウンロード」ボタンでブラウザからzip形式でダウンロードできる

## システム構成

以下の2つのDockerコンテナをdocker-compose.ymlで起動している

|コンテナ名|役割|
|---|---|
|webapp|Webアプリ＋データセットや結果を保持するDB|
|inference|推論を実施して結果を返すWeb APIサーバー|

### webappコンテナ

以下のテックスタックを使用（Docerkfile、requirements.txtも参照）

- Frontend/Backend: Streamlit (Python)
- DB: SQLite
- ORM: SQL Alchemy (`nuscenes-annotator/webapp/app/models`フォルダにスキーマ格納)
- Migration: Alembic

### inferenceコンテナ

以下のテックスタックを使用（Docerkfile、requirements.txtも参照）

- 推論プラットフォーム: CUDA + PyTorch（バージョン互換がシビアなので以下に記載）
    - Python: 3.11.11
    - CUDA: 12.4.1
    - Ubuntu 22.04
    - PyTorch: 2.5.1
    - TorchVision: 0.20.1
- Web API: FastAPI (`http://<サーバーのIPアドレス>:8000/docs`でSwaggerUIを開ける)
- AIモデルの重み
    - GroundingDINO: [こちらのリンク](https://github.com/idea-research/groundingdino#luggage-checkpoints)のGroundingDINO-B
    - SAM2: [こちらのリンク](https://github.com/facebookresearch/sam2#sam-21-checkpoints)のsam2.1_hiera_large
    - Depth-Anything-3: [こちらのリンク](https://github.com/bytedance-seed/depth-anything-3)のDA3METRIC-LARGE
