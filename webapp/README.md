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

### 3. コンテナビルド

#### Linuxの場合（CUDA）

以下コマンドでコンテナをビルドします（TORCH_CUDA_ARCH_LISTが間違っているとビルドは通るがコンテナ内でのGPU推論が無効化されて激遅になるので注意してください）

```bash
docker compose build --no-cache
```

#### Apple Silicon Macの場合（MPS）

MacではCUDAイメージが使えず、またDockerコンテナからはMetal（MPS）が使えません。そのため`docker-compose.yml`ではなく`docker-compose.mac.yml`を使い、以下のどちらかの構成で動かします（データインポート等の`docker compose`コマンドにも`-f docker-compose.mac.yml`を付けてください）。

`.env`には`GPU_DEVICE_ID`/`TORCH_CUDA_ARCH_LIST`は不要です。深度マップ等の保存先は`HOST_DERIVED_ROOT`で変更できます（既定は`webapp/derived`）。また[uv](https://docs.astral.sh/uv/)とlibusb（open3dが使用）が必要です（`brew install uv libusb`）

以下コマンドで推論サーバーの必要パッケージ類をインストールします

```bash
./inference/scripts/setup_mac.sh
```

以下コマンドでwebコンテナをビルドします

```bash
docker compose -f docker-compose.mac.yml build
```

#### Apple Silicon Macの場合（CPU推論）

以下ベースイメージをビルドしてから、`cpu`プロファイルで起動します（推論はかなり遅くなります）

```bash
cd ../docker && bash cpu_devel_ubuntu2204.sh
```

### 4. データインポート

以下コマンドでJSON形式のメタデータをDBにインポートします。`<dataroot>`は先ほど指定したデータセット全体が格納されたフォルダ名、`<version>`はその下層のjsonファイルが格納されたフォルダ名（例: `v1.0-mini`）

```bash
docker compose run --rm webapp \
    python -m app.json_conversion.to_nusc_db \
    --name <好きなデータセット名> --version <version> --dataroot <dataroot>
```

## Usage

### サーバーの立ち上げ

まずWebサーバーと推論サーバーを立ち上げます。プラットフォームに応じて以下のように方法は変わります

#### Linuxの場合（CUDA）

以下コマンドでアプリ立ち上げ

```bash
docker compose up
```

#### Apple Silicon Macの場合（MPS）

以下コマンドで推論サーバーをホスト起動します

```bash
./inference/scripts/run_mac.sh
```

別ターミナルでDockerでwebappを起動します

```bash
docker compose -f docker-compose.mac.yml up
```

MPS未対応の演算は`PYTORCH_ENABLE_MPS_FALLBACK=1`によりCPUで実行されます

#### Apple Silicon Macの場合（CPU推論）

```bash
INFERENCE_BASE_URL=http://inference:8000 docker compose -f docker-compose.mac.yml --profile cpu up --build
```

### ブラウザでアプリを開く

Chrome, Firefox等のブラウザで、サーバーのアドレス（ローカルホストの場合`http://localhost`。デフォルトではポート80=ポート番号指定不要）を開きます。

## 画面の見方

画面は以下の6つのページ（左側のタブで切替）に分かれています。基本的には上から下のタブに向かって操作を実施していけばOKです

|ページ名|役割|
|---|---|
|main|データセットを選択する|
|Scene Selection|アノテーション対象のシーンを選択する|
|Detection2D|Grounding DINOで各物体のバウンディングボックスとラベルを推定する|
|Instance Tracking|SAM2で各物体のインスタンスマスクとトラッキングIDを推定する|
|SAM3 Vocab Tracking|SAM3で画像から直接インスタンスマスクとトラッキングIDを得る（未実装）|
|Depth Bixfitting|Depth-Anything-3で単眼深度推定を実施してインスタンスごとの点群を取得し、この点群に3Dバウンディングボックスをフィッティングする|
|Export|アノテーション結果を含むメタデータをJSON形式でエクスポートする|

### mainページ

データセットを選択するページ。中央のテーブルからアノテーションしたいデータセットを選んで「このデータセットを使う」で確定。
ここで選択されたデータセットは、画面リロードされるまでセッションで維持される

<img src=webapp/docs/images/0_main.png width=50%>

### Select Sceneページ

シーンを選択するページ。中央のテーブルからアノテーションしたいシーンを選んで「このシーンをアノテーション」で確定
ここで選択されたシーンは、画面リロードされるまでセッションで維持される

<img src=webapp/docs/images/1_scene_selection.png width=50%>

### Detect BBoxesページ

Grounding DINOで各物体のバウンディングボックスとラベルを推定するページ。

<img src=webapp/docs/images/2_detection2d.png width=50%>

#### Grounding DINOの推論フローとパラメータ

以下のフローで推論を実行しています。[推論フローの詳細はこちら](inference/CLAUDE.md#推論の実施フロー)を参照して下さい

- Detection2D: Sample Interval間隔のキーフレーム → カメラ → カテゴリグループ[^category_group]のループでGroundingDINOによる2D物体検出の推論を実施（カテゴリグループ内の各ラベル[^nusc_label]ごとに割り当てたすべてのサブラベル[^sub_label]、にアンダースコアをスペースに置き換える等の前処理[^groundingdino_preprocess]を適用してピリオド区切りで結合したプロンプトを使用）。これにより2Dバウンディングボックスとラベルを得る
- Re-Classification: GroundingDINOで得られた各バウンディングボックスのうち指定したラベル[^siglip2_label]のものを、ボックス周囲の切り出し画像でSigLIP2により再判定。これにより各バウンディングボックスのラベル判定の精度を向上させる
- Duplicate Suppression: 再判定後の最終ラベルが等しく内包に近い関係にあるボックス同士をIoS（Intersection over Smaller, 2つのボックスが重なった面積÷2つのうち小さい方の領域の面積）で閾値判定して結合[^2dbox_ios_deletion]。これにより同じ物体を重複検出したボックスを削除する
- Whole-Image Fallback: そのカメラ・フレームのボックス検出数が0の場合、画像全体をリサイズ（Whole-Image Resize Ratioで割合指定）してSigLIP2で推論を実施。これにより画面全体を覆っていてGroundingDINOでは検出困難な物体を検出する

UI上では、「Inference Parameters」エクスパンダー内の以下のタブで区切られたパラメータを調整可能。[具体的なパラメータ一覧はこちら](inference/CLAUDE.md#uiで指定可能なパラメータ)を参照

|タブ名|内容|
|---|---|
|Detection2D|GroundingDINOによる2D物体検出関係のパラメータ（実施フレーム間隔、スコア閾値、NMS閾値）|
|Re-Classification|SigLIP2による再判定関係のパラメータ（ボックス周囲画像切り出し時のクロップ率）|
|Duplicate Suppression|IoSによる重複ボックス削除関係のパラメータ（重複判定のIoS閾値）|
|Whole-Image Fallback|ボックス未検出時の画像全体SigLIP2再判定関係のパラメータ（スコア閾値とリサイズ率）|

「Run Inference」ボタンを押すと推論が実行される。初回実行時はモデルのロードに時間が掛かる（1分ほど）ことに注意

[^category_group]: 似たラベルをグルーピングしてこのカテゴリグループごとに一括推論している。カテゴリグループは`nuscenes-annotator/webapp/app/core/config.py`の`LABEL_TO_CATEGORY_GROUP`辞書で指定可能

[^nusc_label]: 判定対象のラベルとnuScenesカテゴリとの変換は`nuscenes-annotator/webapp/app/core/config.py`の`NUSC_CATEGORY_TO_LABEL`、`LABEL_TO_NUSC_CATEGORY`辞書で指定可能

[^sub_label]: 各ラベルごとのサブラレベルは`nuscenes-annotator/webapp/app/core/config.py`の`LABEL_TO_SUBLABEL`辞書で指定可能

[^groundingdino_preprocess]: GroundingDINOはトークナイザの扱いがシビアなので、実装上の工夫をいくつか加えている。詳細は`nuscenes-annotator/inference/app/models_impl/groundingdino_predict.py`参照

[^siglip2_label]: 入力するGroundingDINOのラベルと再判定時のラベル候補の変換は、`nuscenes-annotator/webapp/app/core/config.py`の`RE_CLASSIFICATION_CANDIDATES`辞書で指定可能。この辞書のvalueがさらに辞書となっており、その辞書のkeyがラベル候補、valueがそのラベル候補と判定されたときに最終的に割り当てるラベルを表す。valueがNoneならそのボックスを削除

[^2dbox_ios_deletion]: IoS閾値を超えた同ラベルのボックス同士を総当たりでマッチングし、`nuscenes-annotator/webapp/app/core/config.py`の`DET2D_IOS_DELETE_DIRECTION`に応じてラベルごとに大きい方または小さい方を残す

#### バウンディングボックス推論結果の表示UI

推論結果はDBに保存され、「保存済みの推論結果」エクスパンダーで選択可能。選択した推論結果がページ下部のカメラ画像上に表示される。右ペインで以下のように表示を変えられる

|名称|内容|
|---|---|
|Compare with GTチェックボックス|チェックするとGround Truthと推論ボックスのプロット画像が左右に並べて表示される。チェックしないと推論ボックスのみが表示される|
|Show boxesチェックボックス|バウンディングボックスのカメラ画像上への表示有無|
|Min scoreスライダー|このScoreを超えたバウンディングボックスのみが画像上に表示される|
|Box textラジオボタン|バウンディングボックスの上にテキスト表示される内容（"SubLabel"ならGroudingDINOで検出された生のラベルであるサブラベル、"Detection Label"ならSublabelを辞書変換したラベル、"Re-Classification"ならSigLIP2再判定後のラベル、"Label"なら最終的なラベル、"Score"なら推論スコア、Noneなら表示なし）|
|Labelsチェックボックス|チェックしたラベルのバウンディングボックスのみが表示される|

### Track Instancesページ

SAM2で各物体のインスタンスマスクとトラッキングIDを推定するページ。プロンプトとして前のページで検出したバウンディングボックスを使用する

<img src=webapp/docs/images/3_instance_tracking.png width=50%>

#### SAM2の推論フローとパラメータ

以下のフローで推論を実行しています。[推論フローの詳細はこちら](inference/CLAUDE.md#推論の実施フロー-1)を参照して下さい

- 入力とするDetection2Dの結果を選択
- Instance Tracking: カメラ → Sample Interval間隔のキーフレームのループで、Detection2Dのバウンディングボックスをプロンプトとして与え、SAM2によるインスタンスマスクのトラッキングを実施。トラッキングの入力データ（ブロックと呼ぶ）はSample Interval分のキーフレームに加え、キーフレーム間をSweeps per Sampleで等分割したフレームも挿入して使用。これにより各物体ごとのインスタンスマスクと、複数フレーム間でのインスタンスの一致を保持するtrack_idを得る。推論方法はTrack ID Inheritanceパラメータに応じて以下のように変わる
    - Track ID Inheritance="continuous_id"のとき: 最初のフレームにボックスプロンプトを与えるForward方向トラッキングのみ実施
    - Track ID Inheritance="forward_backward_matching"のとき: Forward方向トラッキングと、最後のフレームにボックスプロンプトを与えたBackward方向トラッキング両方を実施
- Track ID Inheritance: 以下のように前後のブロック間で検出されたインスタンスマスクのIoUマッチングを実施。これによりブロックの枠を超えて、シーン全体でtrack_idを紐付けられる
    - Track ID Inheritance="continuous_id"のとき: Forward方向トラッキングの最後のフレームのインスタンスマスクと、次のブロックの最初のフレームのインスタンスマスクをIoUマッチング
    - Track ID Inheritance="forward_backward_matching"のとき: Forward方向トラッキングとBackward方向トラッキングの各フレームの時空間IoUマッチングを実施

UI上では、「Inference Parameters」エクスパンダー内の以下のタブで区切られたパラメータを調整可能。[具体的なパラメータ一覧はこちら](inference/CLAUDE.md#uiで指定可能なパラメータ-1)を参照

|タブ名|内容|
|---|---|
|Instance Tracking|SAM2によるインスタンスマスクのトラッキング関係のパラメータ（Sweep per Sample）|
|Track ID Inheritance|IoUマッチングによるブロック間のtrack_id引き継ぎ処理関係のパラメータ（IoU閾値、IoU計算方法、ラベル一致の有無、Track IDの引き継ぎ）|

「Run Inference」ボタンを押すと推論が実行される。初回実行時はモデルのロードに時間が掛かる（1分ほど）ことに注意

#### インスタンスセグメンテーション・トラッキング結果の表示UI

推論結果はDBに保存され、「保存済みの推論結果」エクスパンダーで選択可能。選択した推論結果がページ下部のカメラ画像上に表示される。右ペインで以下のように表示を変えられる

|名称|内容|
|---|---|
|Compare propagationチェックボックス|チェックすると前の推論の最後の画像の伝播インスタンスマスクと、今回の推論の最初の画像のインスタンスマスクを左右に並べて表示（トラッキングIDの引継ぎのIoU判定を視覚的に確認できる。Sample Intervalごとのボックスプロンプトを与えたキーフレームのみが表示対象となる）。チェックしないと今回の推論結果のみが表示される|
|Show boxesラジオボタン|画像上に表示するバウンディングボックスの種類（"Prompt"なら与えたボックスプロンプト、"Instance"ならインスタンスマスクの外接矩形、Noneなら表示なし）|
|Colorラジオボタン|インスタンスマスク・バウンディングボックスの表示色の決め方（"Label"ならラベル、"Track ID"ならトラッキングID）|
|Instance textラジオボタン|インスタンスマスクの上にテキスト表示される内容（"Label"ならラベル、"Track ID"ならトラッキングID、Noneなら表示なし）|
|Labelsチェックボックス|チェックしたラベルのマスクのみが表示される|

### Depth Boxfittingページ

Depth-Anything-3やMapAnythingで単眼深度推定を実施してインスタンスごとの点群を取得し、この点群に3Dバウンディングボックスをフィッティングする

<img src=webapp/docs/images/4_depth_boxfitting.png width=50%>

#### Depth Boxfittingの推論フローとパラメータ

以下のフローで推論を実行しています。[推論フローの詳細はこちら](inference/CLAUDE.md#推論の実施フロー-2)を参照して下さい

- 入力とするInstance Trackingの結果を選択
- Depth Estimation: カメラ → キーフレームのループで、単眼モデル（da3metric-large）またはPose-conditionedモデルによる深度推定を実施し、深度画像を得る
- LiDAR Pointcloud: キーフレームのループで、LiDAR点群の読み込みとPatchwork++による地面点群の除去を実施
- Mask Closing: インスタンスマスクにクロージング処理を適用し、境界部を削除
- Instance Mask Projection: 深度画像をマスク投影してカメラパラメータで点群化したインスタンスごとDepth点群、およびLiDAR点群をマスク投影したインスタンスごとLiDAR点群を得る
- Noise Removal: インスタンスごとDepth点群およびインスタンスごとLiDAR点群に、それぞれROR, DBSCAN, L-shape Filter, Distance Filterによるノイズ除去処理を適用する
- Depth LiDAR Merge: インスタンスごとDepth点群とインスタンスごとLiDAR点群をマージ（条件によってはどちらかのみを用いることもある）し、インスタンスごと点群を得る
- Hull Based Inter-cam Merge: 同フレームで異なるカメラのインスタンスごと点群間で、凸包の重なり率が高いものに新たなtrack_id（hull_merged_track_id）を割り当てる。これによりカメラを跨いで同一インスタンスを結合する
- Box Fitting: [こちらの方法](https://arxiv.org/pdf/2302.01034)でインスタンスごと点群に3D bounding boxをフィッティングし、ボックスの高さと底面も分位点等の方法で推定する。
- Orientation Estimation: hull_merged_track_id＆フレームごとに[PartHOE](https://github.com/zhaojieting/Part_HOE)または[OrientAnything](https://github.com/SpatialVision/Orient-Anything)を用いて物体の向きを推定し、Box Fittingで推定していたボックス角度を補正する
- Orientation Correction: hull_merged_track_idごとに所属する全てのフレームで角度のクラスタリングを実施し、最大クラスタ以外のフレームの角度を最大クラスタのフレームに基づき補正する。これにより周囲フレームからの角度ずれが大きい（角度推定が間違っている可能性が高い）フレームを補正できる
- Box Based Inter-cam Merge: hull_merged_track_id＆フレームごとに、カメラが異なる＆角度が近い＆合体後のボックスの長さと幅が規定サイズの一定比率以下となるインスタンス同士に、新たなtrack_id（box_merged_track_id）を割り当てる。これによりカメラを跨いだ同一インスタンスのうち、Hull Based Inter-cam Mergeで結合し損ねたものを結合する
- Height/Width/Length Estimation: box_merged_track_idごとに所属する全てのフレームを考慮し、分位点および見えている面に基づくボックス補正を実施する。これにより同一インスタンスのボックスサイズをフレーム間で一定に揃える
- Track Smoother: box_merged_trackごとに、インスタンスの動作予測に基づくボックス中心位置の補正（静止 or カルマンスムーザー）を実施する。これにより周囲フレームから推定される動きと整合しないボックスの位置を補正する。また動きの整合性に基づき、途中フレームが抜けたインスタンス同士の結合を実施し、新たなtrack_id（global_track_id）を割り当てる
- Global Height/Width/Length Estimation: box_merged_trackを2個以上結合したglobal_track_idに対して、所属する全てのフレームを考慮し、分位点に基づくボックス補正を実施する。これによりTrack Smootherの結果を考慮した上で同一インスタンスのボックスサイズをフレーム間で一定に揃える

UI上では、「Inference Parameters」エクスパンダー内の以下のタブで区切られたパラメータを調整可能。[具体的なパラメータ一覧はこちら](inference/CLAUDE.md#uiで指定可能なパラメータ-2)を参照

|タブ名|内容|
|---|---|
|General|インスタンスごと点群に対する処理（Mask Closing、Noise Removal）のうち、Depth点群とLiDAR点群共通で適用するパラメータ|
|Depth Estimation|Depth点群に関する処理（Depth Estimation, Mask Closing, Noise Removal）に適用するパラメータ|
|LiDAR Pointcloud|LiDAR点群に関する処理（LiDAR Pointcloud, Mask Closing, Noise Removal, Depth LiDAR Merge）に適用するパラメータ|
|Inter-cam Merge|カメラを跨いだ同一インスタンス結合（Hull Based Inter-cam Merge, Box Based Inter-cam Merge）に適用するパラメータ|
|Box Fitting|3D bounding boxのフィッティング（Box Fitting, Orientation Estimation, Orientation Correction, Height/Width/Length Estimation）に適用するパラメータ|
|Track Smoother|インスタンスの動作予測に基づくボックス中心位置の補正（Track Smoother）に適用するパラメータ|

「Run Inference」ボタンを押すと推論が実行される。初回実行時はモデルのロードに時間が掛かる（1分ほど）ことに注意

#### Depth Boxfitting結果の表示UI

推論結果はDBに保存され、「保存済みの推論結果」エクスパンダーで選択可能。選択した推論結果がページ下部のUI表示部に表示される。

UI表示部は以下の5つのタブで切り替えられる

##### Depth Estimationタブ

推定した深度画像を6カメラ分表示（容量削減のため保存時に縮小リサイズしているため少し表示が粗い）

右ペインで以下のように表示を変えられる

|名称|内容|
|---|---|
|Show masksラジオボタン|深度画像上にインスタンスマスクを重ねるかどうかを指定（"None"ならマスク表示なし、"Orignal"ならクロージング処理前のマスクを表示、"Closed"ならクロージング処理後のマスクを表示）|
|Mask Colorラジオボタン|インスタンスマスクの表示色の決め方。Show Masksが"None"以外のとき有効（"Label"ならラベル、"Track ID"ならInstance TrackingのトラッキングID）|
|Instance textラジオボタン|インスタンスマスクの上にテキスト表示される内容。Show Masksが"None"以外のとき有効（"Label"ならラベル、"Track ID"ならInstance TrackingのトラッキングID、Noneなら表示なし）|

##### PointCloudタブ

深度画像をプロジェクションして得られたインスタンスごと点群を、点群ビュー上に3D表示（点群は生点群をすべて表示すると重いため、点数に合わせてボクセルサンプリングで間引いて表示していることに注意）

右ペインで以下のように表示を変えられる

|名称|内容|
|---|---|
|Raw LiDARチェックボックス|LiDAR点群を灰色で重ねて表示するかどうか|
|LiDAR Groundチェックボックス|LiDAR点群の地面判定（Patchwork++）された部分を茶色で重ねて表示するかどうか|
|Raw Depthチェックボックス|深度画像からプロジェクションした全点群を水色で重ねて表示するかどうか|
|Depth Instacnesチェックボックス|深度画像からプロジェクションしたインスタンスごと点群を表示するかどうか（このビューの表示のメインとなる）|
|Instance Pointsラジオボタン|インスタンスごと点群の表示対象。Filter ParamsエクスパンダーでROR/DBSCANを再実行したときのみ有効※（"Raw"なら元のインスタンスごと点群、"ROR/DBSCAN"ならROR/DBSCAN適用後の点群、"L-shape Filter"ならL-shape Filter適用後の点群、"Distance Filter"ならDistance Filter適用後の点群）|
|Depth Instanceラジオボタン|インスタンスごとDepth点群の表示位置（"raw"なら元々のDepth点群、"LiDAR corrected"ならLiDAR点群で位置補正した後の点群。後者はMax LiDAR Points to use Depth条件で削除されたDepth点群は表示されない）|
|L-shape Filterチェックボックス|L-shape Filterの点群削除に使用されるL字平面を表示するかどうか|
|Fitted Boxesラジオボタン|点群上に重ねて表示する3Dバウンディングボックスの種類（Noneなら表示しない、"First Fit"ならBox Fittingでフィッティングした直後のボックス、"Local Corrected"なら高さ・底面補正とOrientation Estimationでの角度補正後、"Orientation Corrected"ならOrientation Correctionによる複数フレームクラスタリングでの角度補正後、"Box Merged"ならBox Based Inter-cam MergeとHeight/Width/Length Estimationによる複数フレームを考慮したボックスサイズ補正後、"Final Box"ならTrack Smoother適用後の最終的なボックス）|
|Instance Colorラジオボタン|インスタンスごと点群およびバウンディングボックスの表示色の決め方（"Label"ならラベル、"Track ID"ならInstance TrackingのトラッキングID、"Hull Merged"ならHull Based Inter-cam Mergeによる複数カメラ同一インスタンス結合後のhull_merged_track_id、"Box Merged"ならBox Based Inter-cam Mergeによる複数カメラ同一インスタンス結合後のbox_merged_track_id、"Global Track"ならTrack Smootherによる分断トラック結合後の最終的なglobal_track_id）|
|Camerasチェックボックス|チェックしたカメラのインスタンスごと点群のみが表示される|

画面中央上部の3つのボタンで、以下のように視点を切り替えられます

|名称|視点|
|---|---|
|Global Viewボタン|車両の向きに関係なくグローバル座標に固定された視点方向で表示（グローバル座標に対する方向は`nuscenes-annotator/webapp/app/core/config.py`の`DEFAULT_GLOBAL_POINTCLOUD_EYE`と`DEFAULT_GLOBAL_POINTCLOUD_UP`で指定可能）|
|Top Viewボタン|車両の真上から見たBEV視点（恐らくこれが一番使いやすい）|
|Forward Viewボタン|車両の前方向に向かってみた視点|

また、Filter Paramsエクスパンダー内のパラメータを切り替えて「Apply」を押すことで、ROR、DBSCAN、L-shape Filter、Distance Filterのパラメータを変えた際の点群を再表示することができる

##### Box Fittingタブ

Box Fittingで推定してその後の処理で補正された3Dバウンディングボックスを、BEV視点で表示

右ペインで以下のように表示を変えられる

|名称|内容|
|---|---|
|Compare with GTチェックボックス|チェックした場合、Ground Truthのバウンディングボックスと推定されたバウンディングボックスが左右に並べて表示される。チェックしていない場合、推定されたバウンディングボックスのみが表示される|
|LiDAR Groundチェックボックス|LiDAR点群の地面判定（Patchwork++）された部分を茶色で重ねて表示するかどうか|
|Depth Instacnesチェックボックス|深度画像からプロジェクションしたインスタンスごと点群を一緒に表示するかどうか|
|Convex-Hullチェックボックス|yaw角度推定の途中経過として使用するConvex-hull多角形を表示するかどうか|
|Depth Instanceラジオボタン|インスタンスごとDepth点群の表示位置（"raw"なら元々のDepth点群、"LiDAR corrected"ならLiDAR点群で位置補正した後の点群。後者はMax LiDAR Points to use Depth条件で削除されたDepth点群は表示されない）|
|Fitted Boxesラジオボタン|表示するバウンディングボックスの種類（"First Fit"ならBox Fittingでフィッティングした直後のボックス、"Local Corrected"なら高さ・底面補正とOrientation Estimationでの角度補正後、"Orientation Corrected"ならOrientation Correctionによる複数フレームクラスタリングでの角度補正後、"Box Merged"ならBox Based Inter-cam MergeとHeight/Width/Length Estimationによる複数フレームを考慮したボックスサイズ補正後、"Final Box"ならTrack Smoother適用後の最終的なボックス）|
|Orientation Estimationチェックボックス|Orientation Estimationでのを表示するかどうか|
|Instance Colorラジオボタン|バウンディングボックスおよび点群の表示色の決め方（"Label"ならラベル、"Track ID"ならInstance TrackingのトラッキングID、"Hull Merged"ならHull Based Inter-cam Mergeによる複数カメラ同一インスタンス結合後のhull_merged_track_id、"Box Merged"ならBox Based Inter-cam Mergeによる複数カメラ同一インスタンス結合後のbox_merged_track_id、"Global Track"ならTrack Smootherによる分断トラック結合後の最終的なglobal_track_id）|

また、画面中央上部の以下エクスパンダーでは、複数カメラ同一インスタンス結合でのインスタンス結合判定の結果を確認できます
- Hull Merge Diagnosticsエクスパンダー: Hull Based Inter-cam MergeでのBEV視点凸包IoSマッチングによるトラック結合結果を表示
    - グループ: 結合判定後のトラック（hull_merged_track_id）をリスト表示。2個以上のmembersが存在するトラックが、本処理で結合されたものとなる
    - トラック対の判定: 凸包IoSマッチングの対象となった（凸包距離が3m以下）インスタンス点群の対をすべてリスト表示。status=acceptedの行が、マッチ判定されたトラック対（マッチ数がMin Match Frames以上のトラック対が最終的に結合される）
    - 点数不足で当てはめなかったグループ: 結合後の点数がMin Points to fit Box以下でBox Fitting対象外となったhull_merged_track_idを表示。この時点で削除される
- Box-Orient Merge Diagnosticsエクスパンダー: Box Based Inter-cam Mergeでの角度とボックス規定サイズの比較に基づくトラック結合結果を表示
    - 結合したグループ: 本処理で結合されたトラック（box_merged_track_id）と結合前のhull_merged_track_idの組をリスト表示
    - トラック対の判定: 結合の対象となった（凸包距離が3m以下）ボックスの対をすべてリスト表示。result=matchedの行が、マッチ判定されたトラック対（マッチ数がMin Match Frames以上のトラック対が最終的に結合される。これはstatus=acceptedの行に相当）

##### Track Smootherタブ

Track Smootherでの車両の動きを考慮したボックス位置補正結果を表示する。

画面中央上側のLabelラジオボタンで、表示対象のラベルを選択できる。表示結果は以下のように解釈できる

- 菱形◆で表され周囲に2本の平行な点線があるトラック: 「静止している」判定されたトラック
    - 菱形◆の点: トラック全体のXY位置の中央値。全てのトラックのボックス中心位置がこの位置に補正される
    - ○および△印の点: 各フレームにおける補正前のボックス中央位置、これらが◆の位置に補正される。△は初回平滑化時の外れ値を表す
    - 2本の平行な点線: トラック全体のXY位置中央値から幅方向に±Static Radiusの位置に引いた線。静止判定は以下の3条件で実施しており、最初の条件の基準を表す
        - トラックの進行角度中央値に垂直な方向のばらつき（90%分位点）< Static Radius
        - トラックの最初の3分の1と最後の3分の1それぞれの中央値の差 < Static Max Displacement
        - 平滑化した速さの中央値 < Static Max Speed
    - 2本の平行な実線: トラックの最初の3分の1と最後の3分の1の中央値の中間点から両中央値方向にStatic Max Displacement÷2だけ伸ばした位置に引いた線。上記3条件のうち2番目の条件の基準を表す
- ●や▲で表され実線で繋がっているトラック: 「動いている」判定されたトラック
    - 塗り潰し●および▲印の点: 各フレームにおける補正後のボックス中央位置。▲は初回平滑化時の外れ値を、●は非外れ値を表す
    - 白抜き○および△印の点: 各フレームにおける補正前のボックス中央位置、補正後の●や▲の位置まで矢印が引かれる。△は初回平滑化時の外れ値を、○は非外れ値を表す
- ×で表されているトラック: 短いトラック（フレーム数がMin Frames to Smooth以下）
- Select Sampleスライダで選択しているsampleのフレームは太く濃く表示される

右ペインで以下のように表示を変えられる

|名称|内容|
|---|---|
|Fitted Boxesラジオボタン|一緒に表示するバウンディングボックスの種類（Noneなら表示なし、"Box Merged"ならBox Based Inter-cam MergeとHeight/Width/Length Estimationによる複数フレームを考慮したボックスサイズ補正後、"Final Box"ならTrack Smoother適用後の最終的なボックス）|
|Instance Colorラジオボタン|バウンディングボックスおよび点群の表示色の決め方（"Label"ならラベル、"Box Merged"ならBox Based Inter-cam Mergeによる複数カメラ同一インスタンス結合後のbox_merged_track_id、"Global Track"ならTrack Smootherによる分断トラック結合後の最終的なglobal_track_id）|
|Global Trackマルチセレクト|表示対象のglobal_track_idを選択|

##### Box on Camタブ

フィッティングした3Dバウンディングボックスを各カメラ画像に投影して表示する。

右ペインで以下のように表示を変えられる

|名称|内容|
|---|---|
|Compare with GTチェックボックス|チェックした場合、Ground Truthのバウンディングボックスと推定されたバウンディングボックスが左右に並べて表示される。チェックしていない場合、推定されたバウンディングボックスのみが表示される|
|Instance Colorラジオボタン|バウンディングボックスおよび点群の表示色の決め方（"Label"ならラベル、"Track ID"ならInstance TrackingのトラッキングID、"Hull Merged"ならHull Based Inter-cam Mergeによる複数カメラ同一インスタンス結合後のhull_merged_track_id、"Box Merged"ならBox Based Inter-cam Mergeによる複数カメラ同一インスタンス結合後のbox_merged_track_id、"Global Track"ならTrack Smootherによる分断トラック結合後の最終的なglobal_track_id）|
|Instance Textラジオボタン|画像上でボックスの上に表示する文字の種類（Noneなら表示なし、"Label"ならラベル、"Track ID"ならInstance TrackingのトラッキングID、"Hull Merged"ならHull Based Inter-cam Mergeによる複数カメラ同一インスタンス結合後のhull_merged_track_id、"Box Merged"ならBox Based Inter-cam Mergeによる複数カメラ同一インスタンス結合後のbox_merged_track_id、"Global Track"ならTrack Smootherによる分断トラック結合後の最終的なglobal_track_id）|
|Labelsチェックボックス|チェックしたラベルのバウンディングボックスのみが表示される|

### Exportページ

- アノテーション結果を含むメタデータをJSON形式でエクスポートするページ
- 選択したシーンのみエクスポートするか、データセット全体をエクスポートするか選べる
- エクスポート結果にはアノテーション（sample_annotation.jsonおよびinstance.json）だけでなく、他のすべてのメタデータを含む
- 「エクスポート」ボタンを押すとjsonに書き出され、「ダウンロード」ボタンでブラウザからzip形式でダウンロードできる

<img src=webapp/docs/images/5_export.png width=50%>

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
