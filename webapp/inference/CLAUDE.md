# 推論のフロー

## Detection2D

Detection2Dでは、カメラ画像を入力としてGroundingDINOによる2D物体検出とSigLIP2によるラベルの再判定を実施し、物体のバウンディングボックスとラベルを得ます

### 推論の実施フロー

Detection2Dは、以下のフローで行われます

- Sample Interval間隔のキーフレーム → カメラ → カテゴリグループのループ（実行単位の詳細は後述）で以下手順でGroundingDINOによる推論を実施
    - カテゴリグループ内のラベルを`settings.LABEL_TO_CATEGORY_GROUP`に基づき取得してリスト化
    - カテゴリグループ内のラベルに紐づくサブラベルを`settings.LABEL_TO_SUBLABEL`に基づき取得してリスト化
    - 各サブラベル名のアンダースコアをスペースに置き換えたのち、リストをピリオド区切りで結合したテキストをプロンプトとする
    - プロンプトを渡してGrounding DINOで推論
    - 推論で得られたサブラベルを`settings.LABEL_TO_SUBLABEL`で逆引きしてラベルに戻す
- GroundingDINOで得られた各バウンディングボックスのラベルを、以下手順でSigLIP2により再判定
    - GroundingDINOのラベルに応じて`settings.RE_CLASSIFICATION_CANDIDATES`から再判定のラベル候補の辞書を取得（辞書のkeyがラベル候補、valueがそのラベル候補と判定されたときに最終的に割り当てるラベルを表す。valueがNoneならそのボックスを削除）
    - バウンディングボックスをRe-Classification Crop Marginだけ拡張して切り出し画像を作成
    - ラベル候補と切り出し画像をSigLIP2に入力してzero-shot classification推論を実施し、最もスコアの大きいラベルを採用
    - 採用されたラベルに紐づくvalueを`settings.RE_CLASSIFICATION_CANDIDATES`から参照し、最終的なラベルとする
- 再判定後の最終ラベルが等しく内包に近い関係にあるボックス同士を、以下フローで結合
    - 最終ラベルが等しいボックス同士を総当たりでIoSで閾値判定（1つのボックスに2つのボックスが内包される場合、両方のボックスに判定を適用したいため、Hungarianのように1対1で紐づけるマッチングではなく、同ラベルのボックス同士を総当たりでIoS判定することに注意）
    - 閾値を超えた場合、面積が小さい方or大きい方のボックスを削除する（大小どちらを削除するかは後述の`settings.DET2D_IOS_DELETE_DIRECTION`でラベルごとに決める）
- そのカメラ・フレームのボックス検出数が0の場合、画像全体をリサイズ（Whole-Image Resize Ratioで割合指定）してSigLIP2で推論を実施（Whole-Image再判定）。`settings.WHOLE_IMAGE_CANDIDATES`から推論のラベル候補の辞書を取得（辞書のkeyがラベル候補、valueがそのラベル候補と判定されたときに最終的に割り当てるラベルを表す）。推論のスコアがWhole-Image Score Threshold以下または割り当てられたラベル（辞書のvalue）がNoneなら結果を使用せず削除し、どちらでもない場合は割り当てられたラベルの全画面のボックスを追加する

`settings.DET2D_IOS_DELETE_DIRECTION`は以下のようにラベルをkeyとし、"small"ならIoSが閾値を超えたら小さい方のボックスを削除し、"big"なら大きい方のボックスを削除するようにします

```python
DET2D_IOS_DELETE_DIRECTION: dict[str, str] = {
        "car": "small",
        "truck": "small",
        "construction_vehicle": "small",
        "bus": "small",
        "trailer": "small",
        "barrier": "big",
        "traffic_cone": "big",
        "motorcycle": "big",
        "bicycle": "big",
        "pedestrian": "big",
    }
```

### 推論に使用するパラメータ

Detection2Dでは、以下のパラメータを指定できます

#### UIで指定可能なパラメータ

UI（Detection2Dページの画面上部のエクスパンダー）からは以下パラメータを指定できます

|パラメータ名|型|内容|デフォルト値|
|---|---|---|---|
|Sample Interval|int|推論を実施するsample（キーフレーム）の間隔|`settings.DET2D_DEFAULT_SAMPLE_INTERVAL`で指定|
|Score Threshold|float|検出したバウンディングボックスのscore閾値にかける倍率。実際に適用する閾値は、カテゴリグループごとに異なる`settings.DET2D_DEFAULT_SCORE_THRESHOLDS`にここで選択した倍率を掛けたものとなる|1.0|
|NMS Threshold|float|検出したバウンディングボックスでNMSを実施する際に、IoUの閾値にかける倍率。実際に適用する閾値は、同ラベル間のbox結合はカテゴリグループごとに異なる`settings.DET2D_NMS_SAME_CLASS_IOUS`にここで選択した倍率を掛けたもの、別ラベル間のbox結合は`settings.DET2D_NMS_CROSS_CLASS_IOU`にここで選択した倍率を掛けたものとなる|1.0|
|Re-Classification Crop Margin|float|SigLIP2の再判定用切り出し画像をGroudingDINOのバウンディングボックスから拡張する割合|`settings.DEFAULT_RECLASSIFICATION_CROP_MARGIN_RATIO`で指定|
|Whole-Image Score Threshold|float|Whole-Image再判定時のSigLIP2スコアの閾値。この閾値以下の結果は削除|`settings.DEFAULT_WHOLE_IMAGE_SCORE_THRESHOLD`で指定|
|Whole-Image Resize Ratio|float|Whole-Image再判定時の画像リサイズ率|`settings.DEFAULT_WHOLE_IMAGE_RESIZE_RATIO`で指定|
|IoS Delete Threshold|float|ボックス同士を総当たりでIoSで結合判定する際の、IoSの閾値。この閾値以上のボックス同士が結合され、`settings.DET2D_IOS_DELETE_DIRECTION`でラベルごとに指定した大小どちらかのボックスのみが保持される|`settings.DET2D_IOS_DELETE_THRESHOLD`で指定|

#### 設定ファイルから指定するパラメータ

設定ファイル`webapp/app/core/config.py`からは、以下のパラメータを設定できます

|設定名|型|内容|
|---|---|---|
|`settings.NUSC_CATEGORY_TO_LABEL`|dict[str, str]|nuScenes形式データセットのcategoryをDetection2Dのラベルに変換するdict。keyがnuScenesのcagtegory名、valueがDetection2Dのラベル|
|`settings.LABEL_TO_NUSC_CATEGORY`|dict[str, str]|Detection2DのラベルをnuScenes形式データセットのcategoryに変換するdict。keyがDetection2Dのラベル、valueがnuScenesのcagtegory名|
|`settings.LABEL_TO_SUBLABEL`|dict[str, list[str]]|Detection2DのラベルをGroundingDINO推論で使用するサブラベルに変換するdict。keyがDetection2Dのラベル、valueがサブラベル|
|`settings.LABEL_TO_CATEGORY_GROUP`|dict[str, str]|Detection2DのラベルをGrounidingDINO推論の実行単位であるカテゴリグループに変換するdict。keyがDetection2Dのラベル、valueがカテゴリグループ|
|`settings.DET2D_DEFAULT_SAMPLE_INTERVAL`|int|Sample Intervalパラメータのデフォルト値|
|`settings.DET2D_DEFAULT_SCORE_THRESHOLDS`|dict[str, float]|GroundingDINOのスコア閾値のカテゴリグループごとのデフォルト値。keyがカテゴリグループ、valueがスコア閾値のデフォルト値。実際の推論時はこの値に前述のScore Thresholdパラメータを掛けた値未満のバウンディングボックスを削除する|
|`settings.DET2D_NMS_SAME_CLASS_IOUS`|dict[str, float]|GroundingDINOの同ラベルNMSで使用する閾値のカテゴリグループごとのデフォルト値。keyがカテゴリグループ、valueがデフォルト値。実際の推論時はこの値に前述のNMS Thresholdパラメータを掛けた値以上の同ラベルバウンディングボックス同士をNMS結合する（実際にはしきい値を超えたら自動結合されるわけではなく、貪欲マッチングで結合対象を選ぶことに注意。またサブラベルではなくラベルで判定することにも注意）|
|`settings.DET2D_NMS_CROSS_CLASS_IOU`|float|GroundingDINOの別ラベルNMSで使用する閾値のデフォルト値。実際の推論時はこの値に前述のNMS Thresholdパラメータを掛けた値以上の別ラベルバウンディングボックス同士をNMS結合する（実際にはしきい値を超えたら自動結合されるわけではなく、貪欲マッチングで結合対象を選ぶことに注意）|
|`settings.RE_CLASSIFICATION_CANDIDATES`|dict[str, dict[str,str]]|Detection2Dのラベルごとに、SigLIP2によるラベル再判定のラベル候補と最終ラベルを指定するdict。keyがDetection2Dが付けたラベル、valueのkeyがラベル候補、valueのvalueが最終的に割り当てるラベルとなる。最終的に割り当てるラベルがNoneの場合、そのボックスは削除する|
|`settings.DEFAULT_RECLASSIFICATION_CROP_MARGIN_RATIO`|float|前述のRe-Classification Crop Marginパラメータのデフォルト値|
|`settings.RECLASSIFICATION_CROP_MARGIN_RATIO_MAX`|float|前述のRe-Classification Crop Marginパラメータの最大値|
|`settings.WHOLE_IMAGE_CANDIDATES`|dict[str, dict[str,str]]|Whole-Image再判定の、SigLIP2推論のラベル候補と最終ラベルを指定するdict。辞書のkeyがラベル候補、valueがそのラベル候補と判定されたときに最終的に割り当てるラベルを表す|
|`settings.DEFAULT_WHOLE_IMAGE_SCORE_THRESHOLD`|float|Whole-Image Score Thresholdパラメータのデフォルト値|
|`settings.WHOLE_IMAGE_SCORE_THRESHOLD_MAX`|float|Whole-Image Score Thresholdパラメータの最大値|
|`settings.DEFAULT_WHOLE_IMAGE_RESIZE_RATIO`|float|Whole-Image Resize Ratioパラメータのデフォルト値|
|`settings.WHOLE_IMAGE_RESIZE_RATIO_MAX`|float|Whole-Image Resize Ratioパラメータの最大値|
|`settings.DET2D_IOS_DELETE_THRESHOLD`|float|Whole-Image Resize Ratioパラメータのデフォルト値|
|`settings.DET2D_IOS_DELETE_THRESHOLD_MAX`|float|IoS Delete Thresholdパラメータの最大値|
|`settings.DET2D_IOS_DELETE_DIRECTION`|dict[str, str]|ボックス同士を総当たりでIoSで結合判定する際の、ラベルごとに大小どちらのボックスを削除するかを指定する辞書。keyがラベル名を表し、valueが"small"なら小さい方の、"big"なら大きい方のボックスを削除|
|`settings.DET2D_MAX_RUNS_PER_SCENE`|int|1シーンあたり保持するrunの上限|
|`settings.DET2D_MODEL_NAME`|str|runの記録に残すモデル名（重み名を指定。推論サーバー側の実体と合わせる必要がある）|

### 推論の実行単位

Detection2Dは、パラメータで指定したSample Intervalごとに実施する（間引き後のサンプル数N' × カメラ数C × カテゴリグループ数G回推論が実施される）。
例えばSample Interval=4、カテゴリグループ数3のとき、以下のように推論が行われる

- 1回目の推論: Sample0の画像を入力データとして与え、カテゴリグループ1に含まれる複数ラベルをプロンプトとして与えて推論を実施
- 2回目の推論: Sample0の画像を入力データとして与え、カテゴリグループ2に含まれる複数ラベルをプロンプトとして与えて推論を実施
- 3回目の推論: Sample0の画像を入力データとして与え、カテゴリグループ3に含まれる複数ラベルをプロンプトとして与えて推論を実施
- 4回目の推論: Sample4の画像を入力データとして与え、カテゴリグループ1に含まれる複数ラベルをプロンプトとして与えて推論を実施
- 5回目の推論（以下略）

### 推論モデルの実装

GroundingDINOのモデルは、HuggingFace版ではなくGitHubの公式リポジトリ版を使用し、重みgroundingdino_swinb_cogcoor.pthを手動ダウンロードし、マウントでコンテナ内の/opt/checkpointsフォルダ内に配置して使用する（READMEに事前に重みをダウンロードする必要がある旨を記載）。

GroundingDINOのモデルインスタンスは以下のように作成する

```python
GROUNDINGDINO_CONFIG_PATH = "/opt/third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
GROUNDINGDINO_WEIGHT_PATH = "/opt/checkpoints/groundingdino_swinb_cogcoor.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
# Build the GroundingDINO model
groundingdino_model = load_model(str(GROUNDINGDINO_CONFIG_PATH), str(GROUNDINGDINO_WEIGHT_PATH), device=device)
```

推論の実装例（フレーム、カメラ、カテゴリグループ数分だけ推論を実施し、結果をまとめて返す）

```python
# ラベル -> カテゴリグループの変換dict
LABEL_TO_CATEGORY_GROUP: dict[str, str] = {
    "car": "vehicle",
    "truck": "vehicle",
    "construction_vehicle": "vehicle",
    "bus": "vehicle",
    "trailer": "vehicle",
    "barrier": "road_object",
    "traffic_cone": "road_object",
    "motorcycle": "two_wheeler",
    "bicycle": "two_wheeler",
    "pedestrian": "pedestrian",
}
category_groups = set(list(LABEL_TO_CATEGORY_GROUP.values()))
# ラベル -> GroundingDINOでプロンプトに使用するサブラベルの変換dict
LABEL_TO_SUBLABEL: dict[str, str] = {
    "car": ["car", "van"],
    "truck": ["truck", "tank_truck"],
    "construction_vehicle": ["construction_vehicle"],
    "bus": ["bus"],
    "trailer": ["trailer"],
    "barrier": ["barrier"],
    "motorcycle": ["motorcycle"],
    "bicycle": ["bicycle"],
    "pedestrian": ["pedestrian"],
    "traffic_cone": ["traffic cone"],
}
###### データ読込処理（省略） ######
# Frame loop
for sample_index in proc_sample_indices:
    # Camera channel loop
    for camera_channel in CAMERA_CHANNELS:
        # Category group loop
        for category_group in category_groups:
            image = images[sample_index][camera_channel]
            box_threshold = DET2D_DEFAULT_SCORE_THRESHOLDS[category_group]
            labels = [k for k, v in LABEL_TO_CATEGORY_GROUP.items() if v == category_group]
            sublabels = [sublabel for label in labels for sublabel in LABEL_TO_SUBLABEL[label]]
            # Infer the image with GroundingDINO model
            predicted_boxes, caption = predict_multi_labels(
                model=groundingdino_model,
                image=image,
                labels=sublabels, # 実際にプロンプトとして使用するラベル=サブラベルを渡す
                box_threshold=box_threshold,
                same_class_nms_iou=0.6,
                cross_class_nms_iou=0.85,
                sublabel_to_label=sublabel_to_label # Reverse mapping of LABEL_TO_SUBLABEL
            )
```

カテゴリグループ内のラベルをそのままプロンプトとして渡すわけではなく、ラベルに紐づく複数（1個以上）のサブラベルをまとめてプロンプトに渡し、GroundingDINOの推論を実施します。

例えば上記の例の"vehicle"カテゴリグループにおいては、["car", "van", "truck", "tank_truck", "construction_vehicle", "bus", "trailer"]のリストがlabels引数として`predict_multi_labels`関数に渡されます。この関数内では、以下の処理によりリストをプロンプトに整形して推論を実行し

- リスト内の要素のアンダースコアをスペースで置換する（自然言語として適切な表現となり、推論の精度が上がる）
- リストをピリオド区切りの文字列に変換し、プロンプトとする（GroundingDINOの推奨に従う）
- リストの各要素（サブラベル）のトークン位置を保持しておく
- プロンプトをGroundingDINOに渡して推論を実行
- 推論で得られた各ボックスのトークンごとのスコアを、先ほど保持したトークン位置情報を基にサブラベルごとに合計し、各サブラベルのスコアを得る
- スコアが最大のサブラベルをそのボックスのサブラベルとし、逆引きdict`sublabel_to_label`を基にラベルに変換する
- ラベルが等しいボックス同士で`same_class_nms_iou`に基づきNMSを実施
- ラベルが異なるボックス同士で`cross_class_nms_iou`に基づきNMSを実施
- ボックスの座標を標準化座標からピクセル座標に変換

## Instance Tracking

Instance Trackingでは、Detection2Dのバウンディングボックスとカメラ画像を入力として、SAM2によるインスタンスセグメンテーションとトラッキングを実施し、各物体のキーフレームごとのインスタンスマスクとtrack_idを得ます

### 推論の実施フロー

Instance Trackingは、以下のフローで行われます

- 入力とするDetection2Dの結果を選択
- カメラ → 選択したDetection2DのSample Interval間隔のキーフレームのループ（実行単位の詳細は後述）で、以下手順でSAM2による推論を実施
    - 推論対象データとして、対象キーフレームからSample Interval後のキーフレームまで、各キーフレームあたりSweeps per Sampleフレームを使用する（例: Sample Interval=4, Sweeps per Samples=2のとき、全部で9フレームを推論用データとして使用。このひとまとまりの推論用フレームをブロックと呼ぶこととする）
    - Track ID Inheritance="continuous_id"のとき: 以下フローでForward方向のトラッキング推論とtrack_idの継承を実施
        - 最初のフレームにDetection2D推論結果をプロンプトとして与え、時間順方向（Forward方向）にSAM2のpropagation推論を実施
        - 得られた最後のフレームのインスタンスと、次のブロックの最初のフレームにボックスプロンプトを与えて得られたインスタンスをIoUマッチングし、マッチングしたtrack_idを次のブロックに引き継ぐ
    - Track ID Inheritance="forward_backward_matching"のとき: 以下フローでForward＆Backward方向のトラッキング推論とtrack_idの継承を実施
        - 最初のフレームにDetection2D推論結果をプロンプトとして与え、時間順方向（Forward方向）にSAM2のpropagation推論を実施
        - 最後のフレームにDetection2D推論結果をプロンプトとして与え、時間逆方向（Backward方向）にSAM2のpropagation推論を実施
        - Forward/Backward両トラッキングの各インスタンスの全フレームでの時空間IoUマッチング（Hungarian algorithm）を実施し、マッチしたインスタンス同士のtrack_idを次のブロックに引き継ぐ

### 推論に使用するパラメータ

Instance Trackingでは、以下のパラメータを指定できます

#### UIで指定可能なパラメータ

UI（Instance Trackingページの画面上部のエクスパンダー）からは以下パラメータを指定できます

|パラメータ名|型|内容|デフォルト値|
|---|---|---|---|
|Box Prompt|-|入力とするDetection2Dの結果を選択（`detection_2d_params`テーブルのレコードをラジオボタン付きでリスト表示）|`started_at`が最新のレコード|
|Sweeps per Sample|float|トラッキングに使用する画像のキーフレームあたりフレーム数（1ならキーフレームのみを使用）|`settings.DEFAULT_TRACKING_NUM_SWEEPS`で指定|
|IoU Threshold|float|track_idのブロック間引き継ぎに使用するインスタンス同士のHungarian algorithmによるIoUマッチング後のIoU閾値|`settings.DEFAULT_TRACKING_IOU_THRESHOLD`で指定|
|IoU Method|"Box" or "Mask"|上記IoUマッチングで使用するIoUの計算方法を、外径バウンディングボックス同士のIoUにするか、Mask IoUにするかを選択|`settings.DEFAULT_TRACKING_IOU_METHOD`で指定|
|IoU Label Match|"Label", "Category Group", or None|上記IoUマッチング時にラベルまたはカテゴリグループの一致も考慮するかを指定|`settings.DEFAULT_TRACKING_IOU_LABEL_MATCH`で指定|
|Track ID Inheritance|"continuous_id" or "forward_backward_matching"|IoUマッチングに使用する手法を指定する。両手法の詳細は後述の「推論モデルの実装」参照|`settings.DEFAULT_TRACK_ID_INHERITANCE`で指定|

#### 設定ファイルから指定するパラメータ

設定ファイル`webapp/app/core/config.py`からは、以下のパラメータを設定できます

|設定名|型|内容|
|---|---|---|
|`settings.DEFAULT_TRACKING_NUM_SWEEPS`|int|Sweeps per Sampleパラメータのデフォルト値|
|`settings.DEFAULT_TRACKING_IOU_THRESHOLD`|float|IoU Thresholdパラメータのデフォルト値|
|`settings.DEFAULT_TRACKING_IOU_METHOD`|"box" or "mask"|IoU Methodパラメータのデフォルト値|
|`settings.DEFAULT_TRACKING_IOU_LABEL_MATCH`|"label", "category_group" or None|IoU Label Matchパラメータのデフォルト値|
|`settings.DEFAULT_TRACK_ID_INHERITANCE`|"continuous_id" or "forward_backward_matching"|Track ID Inheritanceパラメータのデフォルト値|
|`settings.TRACKING_MAX_RUNS_PER_SCENE`|int|1シーンあたり保持するrunの上限|
|`settings.TRACKING_MODEL_NAME`|str|runの記録に残すモデル名（重み名を指定。推論サーバー側の実体と合わせる必要がある）|
|`settings.DEFAULT_TRACKING_MASK_SCORE_THRESHOLD`|float|現状使用していない（将来的にマスクにスコアを付けるトラッキングアルゴリズムを使用する場合のために準備）|
|`settings.TRACKING_STUB_DELAY_SEC`|float|モデル使用時には使用しない（スタブでの待ち時間を指定する）|

### 推論の実行単位

Instance Trackingは、Box Promptラジオボタンで指定した`detection_2d_params`（Detection2Dでの推論run単位でパラメータを保持するテーブル）のSample Intervalごとに実施する。このSample Interval間のフレームをまとめてトラッキングに利用する（キーフレームだけでなく、Sweeps per Sampleパラメータに応じて非キーフレームをほぼ等間隔となるよう選択して使用する。このトラッキングに利用するフレーム群をブロックと呼ぶこととする）。
例えばSample Interval=4のとき、以下のように推論が行われる

- 1つ目のブロック（Sample0から4）: Sample0からSample4までの画像をまとめたブロックを入力データとして与え、トラッキング推論を実施
- 2つ目のブロック（Sample4から8）: Sample4からSample8までの画像をまとめたブロックを入力データとして与え、トラッキング推論を実施。前回（1回目の推論）のtrack_idをどのように引き継ぐかは、後述の`track_id_inheritance`パラメータにより変わる
- 3回目の推論（以下略）

入力データとして与えるブロックは、以下の参考コードのように、キーフレームと前のキーフレームとの間をSweeps per Sampleパラメータ（参考コード中の`nsweeps`）で等分するするように選択される

```python
sweep_indices = sorted(dict.fromkeys(len(frames) - 1 - int(i)
                       for i in np.linspace(0, len(frames), nsweeps, endpoint=False)))
sweep_frames = [frames[i] for i in sweep_indices]
```

### 推論モデルの実装

SAM2のモデルは、HuggingFace版ではなくGitHubの公式リポジトリ版を使用し、重みsam2.1_hiera_large.ptを手動ダウンロードし、マウントでコンテナ内の/opt/checkpointsフォルダ内に配置して使用する（READMEに事前に重みをダウンロードする必要がある旨を記載）。

SAM2のモデルインスタンスは以下のように作成する

```python
from sam2.build_sam import build_sam2_video_predictor

SAM2_CONFIG_PATH = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT_PATH = "/opt/checkpoints/sam2.1_hiera_large.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam2_predictor = build_sam2_video_predictor(str(SAM2_CONFIG_PATH), str(SAM2_CHECKPOINT_PATH), device=device)
```

インスタンスセグメンテーション・トラッキングの推論はこの`sam2_predictor`インスタンスを用いて実施しますが、`track_id_inheritance`パラメータに"continuous_id"と"reverse_matching"どちらを指定するかで、各推論間のtrack_idの引き継ぎ方法を以下のように変えます（どちらも今回のSample Intervalから次のSample Intervalまでの画像フレームをまとめて推論に使用する）

- `track_id_inheritance="continuous_id"`: 最初のフレームにボックスプロンプト（Detection2Dの結果）を与えて時間順方向（Forward方向）にトラッキングし、得られた最後のフレームのインスタンスと、次のブロックの最初のフレームにボックスプロンプトを与えて得られたインスタンスをIoUマッチングし、マッチングしたtrack_idを次のブロックに引き継ぐ
- `track_id_inheritance="forward_backward_matching"`: 最初のフレームにボックスプロンプトを与えたForward方向トラッキングと、最後のフレームにボックスプロンプトを与えた時間逆方向（Backward方向）トラッキングを実施し、両トラッキングの各インスタンスの全フレームでの時空間IoUマッチングを実施してtrack_idを引き継ぐ

両手法の詳細を以下に示します

#### track_id_inheritance="continuous_id"

この方法では、ブロック内の最初のフレームにボックスプロンプト（Detection2Dの結果）を与えて時間方向に前から後ろ方向（Forward方向）にフレームをトラッキングし、トラッキングで得られた最後のフレームの全インスタンスと、次のブロックの推論の最初のフレームの全インスタンスをIoUでHungarianマッチングしてから`IoU Threshold`で閾値判定し、マッチングした場合track_idを次のブロックの推論に引き継ぎます。

例えばSample Interval=4のとき、以下のように推論が行われます。

- 1つ目のブロック（Sample0から4）: Sample0からSample4までの画像を入力データとして与え、Sample0のボックス（Detection2Dの結果）をプロンプトとして与えて推論（トラッキングpropagation）を実施
- 2つ目のブロック（Sample4から8）: Sample4からSample8までの画像を入力データとして与え、Sample4のボックス（Detection2Dの結果）をプロンプトとして与えて推論を実施。このとき、前回の推論（1回目の推論）で伝播により得られたSample4のインスタンスと、今回の推論（2回目の推論）で得られたSample4のインスタンス同士で貪欲マッチングを実施し、IoUが閾値（後述）以上のインスタンスが存在すれば1回目の推論のtrack_idを引き継ぎ、存在しなければ新たにtrack_idを割り振ります。
- 3回目の推論（以下略）

このケースでは、間引き後のサンプル数N' × カメラ数C回推論が実施されます。

推論の実装例

```python
# Camera channel loop
for camera_channel in CAMERA_CHANNELS:
    # Sample loop（Detection2DのSample Intervalに従う）
    for sample_index in proc_sample_indices:
        images = # sample_indexから次のsample_indexまでの画像をSweeps per Sampleに従い非キーフレームも含めて時間順リスト化（sample_index=0のときはキーフレームしか存在しないので注意）
        det2d_boxes = # sample_indexにおけるDetection2Dの検出バウンディングボックスを取得
        # Initialize the inference state for SAM2 with the loaded images
        inference_state = init_frame_state(sam2_predictor, images)
        # 一時的にtrack_idをアサイン(あとで前述した前sample_indexからのpropagation結果とのIoUマッチングにより再アサインする)
        for i_box in range(len(det2d_boxes)):
            det2d_boxes[i_box].track_id = i_box
        tmp_track_id_to_label = {box.track_id: box.label for box in det2d_boxes}
        # Reset and Add the box prompts
        sam2_predictor.reset_state(inference_state)
        predicted_instances = add_box_prompts(
            predictor=sam2_predictor,
            inference_state=inference_state,
            frame_idx=0, # 最初の画像にボックスプロンプトを追加
            box_prompts=det2d_boxes,
        )
        # propagate the tracking inference
        full_result_instances = propagate_inference(sam2_predictor, inference_state)

        # Identify the track_id based on IoU with propagated instances from the previous frame
        if i == 0:  # first sample in the sequence
            track_id_mapping = {i: i for i in range(len(det2d_boxes))}
            max_track_id = max(track_id_mapping.values())
        else:
            idx_to_track_id, max_track_id = assign_continuous_tracking_ids(
                current_predicted_instances=predicted_instances,
                prev_propagated_instances=prev_propagated_instances,
                max_track_id=max_track_id,
                iou_method=TRACKING_IOU_METHOD,
                iou_threshold=TRACKING_IOU_THRESHOLD,
                match_label=TRACKING_IOU_LABEL_MATCH
            )
            track_id_mapping = {instance.box.track_id: idx_to_track_id[predicted_inst_idx] for predicted_inst_idx, instance in enumerate(predicted_instances)}

        # Update the labels and track_ids
        for propagate_frame_idx, frame_result_instances in full_result_instances.items():
            for tmp_track_id, instance in frame_result_instances.items():
                instance.box.label = tmp_track_id_to_label[tmp_track_id]
                instance.box.track_id = track_id_mapping[tmp_track_id]
        for instance in predicted_instances:
            instance.box.track_id = track_id_mapping[instance.box.track_id]

        # Filter the results to only include the keyframes
        result_instances = {k: v for i_sweep, (k, v) in enumerate(full_result_instances.items()) if i_sweep in keyframe_indices}
```

#### track_id_inheritance="forward_backward_matching"

この方法では、ブロック内の最初のフレームにボックスプロンプト（Detection2Dの結果）を与えたForward方向トラッキング推論と、最後のフレームにボックスプロンプト（Detection2Dの結果）を与えたBackward方向トラッキング推論を比較して、以下の式で複数フレームでのIoU（時空間IoU）を求めます

```math
score(i, j) = \sum_k \frac{\text{IoU}_k(F_i, B_j)} {k} \\
F_i: Forward方向トラッキングのi番目のインスタンス \\
B_j: Backward方向トラッキングのj番目のインスタンス \\
k : F_i または B_j が存在するフレーム
```

求めた全ての$(i, j)$に対する$score(i, j)$の行列から、Hungarian algorithmでForwardとBackwardのインスタンスをマッチングしてから`IoU Threshold`で閾値判定し、紐づいた場合はForwardのtrack_idをBackwardに引き継ぐことで、次のブロックのForwardにもtrack_idが引き継がれます（Backwardのプロンプトボックスは次のブロックのForwardと等しいため）

この方法では間引き後のサンプル数N' × カメラ数C回 × 2回推論が実施されます（ForwardとBackward両方向で実施するため、track_id_inheritance="continuous_id"の2倍の推論数となる）。

## Depth Boxfitting

Depth Boxfittingでは、Instance Trackingのインスタンスマスクとtrack_id、およびカメラ画像・LiDAR点群を入力として、Depth-Anything-3による深度推定と、複数カメラ間での同一インスタンス結合、および[こちらの手法](https://arxiv.org/abs/2302.01034)によるボックス位置推定を実施し、各物体の3Dバウンディングボックスとinstance_idを得ます

### 推論の実施フロー

Depth Boxfittingは、以下のフローで行われます

- 入力とするInstance Trackingの結果を選択
- カメラ → キーフレームのループで、以下手順でカメラ画像からDepth-Anything-3による深度推定を実施
    - 画像を入力してDepth-Anything-3で単眼深度推定推論
    - sky_mask機能により空領域を保持
    - 内部パラメータを深度画像のサイズに合わせて補正
    - 補正後内部パラメータを用いて、深度画像をメートル単位に補正
- キーフレームのループで、以下手順でカメラ間同一インスタンス結合・3Dバウンディングボックスのフィッティングを実施
    - キーフレーム内の全インスタンスマスクを走査し、以下手順でインスタンスごとDepth点群を得る
        - インスタンスマスクにクロージング処理を適用。クロージング処理のカーネルサイズは以下のように決める
            - 収縮: Dilationパラメータを使用
            - 膨張: 基本はErosionパラメータを使用するが、マスクの短辺がSmall Mask Short Side以下なら`round(Dilation+(Erosion-Dilation)*Small Mask Erosion Ratio)`を使用する
        - 深度画像からsky_maskを用いて空領域を削除したのちインスタンスマスクに投影し、補正後内部パラメータと外部パラメータを用いて点群に変換し、インスタンスごとDepth点群を得る
        - インスタンスごとDepth点群にRORとDBSCANによるノイズ除去を順番に適用する。各種パラメータは以下の値を使用する
            - RORのnb_points: Depth ROR nb_pointsパラメータにラベルごとに異なる`settings.NB_POINTS_RATIO`を掛け、マスクの短辺がSmall Mask Short Side以下ならさらに掛ける
            - RORのradius: Depth ROR nb_pointsパラメータをそのまま用いる
            - DBSCANのmin_samples: Depth DBSCAN min_samplesパラメータを用い、マスクの短辺がSmall Mask Short Side以下ならさらに掛ける
            - DBSCANのeps: Depth DBSCAN epsパラメータをそのまま用いる
    - LiDARを使用する場合、以下手順でインスタンスごとLiDAR点群を作成してDepth点群と混合し、最終的なインスタンスごと点群を得る
        - キーフレームから直近LiDAR Sweepsスイープ分のLiDARデータを読み込み、スイープごとにPatchwork++による地面除去を実行したのち、キーフレームのLiDAR座標に合わせて合体する
        - 合体したLiDAR点群のうち自車周辺の直方体領域（`settings.EGO_BOX_X_RANGE`,`settings.EGO_BOX_Y_RANGE`,`settings.EGO_BOX_Z_RANGE`で指定）の点を自車反射とみなして削除
        - キーフレーム内の全インスタンスマスクを走査し、LiDAR点群をインスタンスマスクに投影し、インスタンスごとLiDAR点群を得る
        - インスタンスごとLiDAR点群にRORとDBSCANによるノイズ除去を順番に適用する。各種パラメータは以下の値を使用する
            - RORのnb_points: LiDAR ROR nb_pointsパラメータにラベルごとに異なる`settings.NB_POINTS_RATIO`を掛ける
            - RORのradius: LiDAR ROR nb_pointsパラメータをそのまま用いる
            - DBSCANのmin_samples: LiDAR DBSCAN min_samplesパラメータをそのまま用いる
            - DBSCANのeps: LiDAR DBSCAN epxパラメータをそのまま用いる
        - インスタンスごとLiDAR点群の点数が閾値未満なら、LiDAR点群は使用せずにDepth点群のみをインスタンスごと点群として使用する
        - インスタンスごとLiDAR点群の点数が閾値以上なら、インスタンスごとDepth点群のz座標に`median(z_lidar / z_depth)`を掛けて（z_lidar、z_depthは対応するLiDAR点が存在するインスタンスマスクの点から得た組み合わせ）深さ方向位置を補正したのち、LiDAR点群と混合してインスタンスごと点群とする
        - インスタンスごと点群の点数が0なら、そのインスタンスは削除する
    - 以下手順で複数カメラ間での同一インスタンス結合を実施
        - 別カメラのインスタンスごと点群の組み合わせのうち、上から見たXY座標での凸包のIoS（小さい方に対する重なり率）が閾値（Overlap Threshold）以上かつ中心距離がMax Centroid Distance以下の組み合わせをHungarian algorithmで結合し、同一のインスタンスID（global_track_id）を割り振る（ただし、同一カメラかつ同一フレームを含む、またはフレームがMax Same-Camera Gap以上離れているトラック同士は結合されないようにする）
        - 結合されなかったインスタンスには個別のインスタンスIDを割り振る
    - global_track_idごとに結合した点群に対して、以下の方法で3Dバウンディングボックスをフィッティングする
        - global_track_idあたりの点数がMin Points to fit Box未満なら、バウンディングボックスをフィッティングしない
        - [こちらの手法](https://arxiv.org/abs/2302.01034)で3Dバウンディングボックスの平面方向の大きさと角度（yaw角）を推定する
        - 3Dバウンディングボックスの高さは、global_track_idごと点群の分位点に基づき推定する

### 推論に使用するパラメータ

Depth Boxfittingでは、以下のパラメータを指定できます

#### UIで指定可能なパラメータ

UI（Depth Boxfittingページの画面上部のエクスパンダー）からは以下パラメータを指定できます

|パラメータ名|型|内容|デフォルト値|
|---|---|---|---|
|Instance Tracking|-|入力とするInstance Trackingの結果を選択（`instance_tracking_2d_params`テーブルのレコードをラジオボタン付きでリスト表示）|`started_at`が最新のレコード|
|Use LiDAR|bool|LiDAR点群を使用するかどうかを指定|`settings.BOXFIT_USE_LIDAR_DEFAULT`で指定|
|Dilation|int|インスタンスマスクのクロージング処理の膨張カーネルサイズ||
|Erosion|int|インスタンスマスクのクロージング処理の収縮カーネルサイズ（）||
|Small Mask Short Side|int|各種小マスク向け倍率を適用するためのマスクの短辺サイズ閾値（これより小さければ収縮カーネルサイズにはSmall Mask Erosion Ratioを、Depth ROR nb_points, Depth DBSCANにはSmall Mask nb_points / min_samples Ratioを適用）|`settings.SMALL_MASK_SHORT_SIDE_DEFAULT`|
|Small Mask Erosion Ratio|float|短辺がSmall Mask Short Side以下のマスクの収縮カーネルサイズに適用する倍率。適用後カーネルサイズは`round(Dilation+(Erosion-Dilation)*Small Mask Erosion Ratio)`|`settings.SMALL_MASK_EROSION_RATIO_DEFAULT`|
|Depth ROR nb_points|int|インスタンスごとDepth点群に適用するRORのnb_pointsパラメータ||
|Depth ROR radius|float|インスタンスごとDepth点群に適用するRORのradiusパラメータ||
|Depth DBSCAN eps|float|インスタンスごとDepth点群に適用するDBSCANのepsパラメータ||
|Depth DBSCAN min_samples|int|インスタンスごとDepth点群に適用するDBSCANのmin_samplesパラメータ||
|Small Mask nb_points / min_samples Ratio|float|短辺がSmall Mask Short Side以下のマスクのSmall Mask nb_points / min_samples Ratioに適用する倍率|`settings.SMALL_MASK_NB_POINTS_RATIO_DEFAULT`|
|LiDAR Sweeps|int|キーフレームあたりで結合するLiDAR点群のsweep数（1ならキーフレームのみを使用）|`settings.DEFAULT_TRACKING_NUM_SWEEPS`で指定|
|Min LiDAR Points to use LiDAR|int|インスタンスごとLiDAR点群をインスタンス点群として使用するための点数の下限しきい値（これを下回ったインスタンスはDepth点群のみ使用する）||
|Max LiDAR Points to use Depth|int|Depth点群を使用するためのLiDAR点数の上限しきい値（LiDAR点群数がこれを上回ったインスタンスはLiDAR点群のみ使用する）||
|LiDAR ROR nb_points|int|インスタンスごとLiDAR点群に適用するRORのnb_pointsパラメータ||
|LiDAR ROR radius|float|インスタンスごとLiDAR点群に適用するRORのradiusパラメータ||
|LiDAR DBSCAN eps|float|インスタンスごとLiDAR点群に適用するDBSCANのepsパラメータ||
|LiDAR DBSCAN min_samples|int|インスタンスごとLiDAR点群に適用するDBSCANのmin_samplesパラメータ||
|Inter-cam Merge Enabled|bool|カメラ間の同一インスタンス結合を実施するかどうかを指定||
|Inter-cam Merge Match Method|"BEV convex-hull"|カメラ間の同一インスタンス結合に使用する手法を選択|"BEV convex-hull"ならXY平面での凸包の重なりがしきい値以上＆重心距離がしきい値以下のフレーム数がMin Match Frames以上なら同一インスタンスと判定して結合。|`settings.DEFAULT_MERGE_METHOD`で指定|
|Inter-cam Merge Overlap Threshold|float|Match Method="BEV convex-hull"のとき使用。XY平面での凸包の重なりのしきい値||
|Inter-cam Merge Max Centroid Distance|float|Match Method="BEV convex-hull"のとき使用。インスタンス間の重心距離のしきい値||
|Inter-cam Merge Min Match Frames|float|Match Method="BEV convex-hull"のとき使用。同一判定されたフレーム数がこのしきい値以上なら同一インスタンスと判定して結合する||
|Inter-cam Merge Max Same-Camera Gap|int|同一カメラのトラック同士は、このキーフレーム数を超えて離れている場合は結合されない（同一キーフレームを含む場合も結合されない）|`settings.DEFAULT_MERGE_MAX_SAME_CAMERA_GAP`で指定|
|Inter-cam Merge Label Match|"Label", "Category Group", or None|カメラ間同一インスタンス結合時にラベルまたはカテゴリグループの一致も考慮するかを指定|`settings.DEFAULT_TRACKING_IOU_LABEL_MATCH`で指定|
|Min Points to fit Box|int|Box Fittingを行うためのglobal_track点数の下限しきい値（これを下回ったglobal_trackは3Dバウンディングボックスを推定しない）|`settings.BOXFIT_MIN_POINTS_DEFAULT`で指定|
|Boxfitting Method|"convex_hull_moa"|3Dバウンディングボックスのフィッティングに使用するアルゴリズム。"convex_hull_moa"なら[こちらの論文](https://arxiv.org/abs/2302.01034)の手法を使用||
|convex_hull_moa angle_step_deg|float|Boxfitting Method="convex_hull_moa"のとき使用。最もフィットするyaw角度を探索するステップ||
|convex_hull_moa z_percentile|float|Boxfitting Method="convex_hull_moa"のとき使用。高さの下限と上限として採用するパーセンタイル||

#### 設定ファイルから指定するパラメータ

設定ファイル`webapp/app/core/config.py`からは、以下のパラメータを設定できます

|設定名|型|内容|
|---|---|---|
|`settings.BOXFIT_USE_LIDAR_DEFAULT`|bool|Use LiDARパラメータのデフォルト値|
|`settings.MASK_DILATION_DEFAULT`|int|Dilationパラメータのデフォルト値|
|`settings.MASK_DILATION_MAX`|int|Dilationパラメータの最大値|
|`settings.MASK_EROSION_DEFAULT`|int|Erosionパラメータのデフォルト値|
|`settings.MASK_EROSION_MAX`|int|Erosionパラメータの最大値|
|`settings.SMALL_MASK_SHORT_SIDE_DEFAULT`|int|Small Mask Short Sideパラメータのデフォルト値|
|`settings.SMALL_MASK_SHORT_SIDE_MAX`|int|Small Mask Short Sideパラメータの最大値|
|`settings.SMALL_MASK_EROSION_RATIO_DEFAULT`|float|Small Mask Erosion Ratioパラメータのデフォルト値|
|`settings.NB_POINTS_RATIO`|float|ROR nb_pointsにかけるラベルごとの倍率|
|`settings.DEPTH_ROR_NB_POINTS_DEFAULT`|int|Depth ROR nb_pointsパラメータのデフォルト値|
|`settings.DEPTH_ROR_NB_POINTS_MAX`|int|Depth ROR nb_pointsパラメータの最大値|
|`settings.DEPTH_ROR_RADIUS_DEFAULT`|float|Depth ROR radiusパラメータのデフォルト値|
|`settings.DEPTH_ROR_RADIUS_MAX`|float|Depth ROR radiusパラメータの最大値|
|`settings.DEPTH_DBSCAN_EPS_DEFAULT`|float|Depth DBSCAN epsパラメータのデフォルト値|
|`settings.DEPTH_DBSCAN_EPS_MAX`|float|Depth DBSCAN epsパラメータの最大値|
|`settings.DEPTH_DBSCAN_MIN_SAMPLES_DEFAULT`|int|Depth DBSCAN min_samplesパラメータのデフォルト値|
|`settings.DEPTH_DBSCAN_MIN_SAMPLES_MAX`|int|Depth DBSCAN min_samplesパラメータの最大値|
|`settings.SMALL_MASK_NB_POINTS_RATIO_DEFAULT`|float|Small Mask nb_points / min_samples Ratioパラメータのデフォルト値|
|`settings.LIDAR_NUM_SWEEPS_DEFAULT`|int|LiDAR Sweepsパラメータのデフォルト値|
|`settings.LIDAR_MIN_POINTS_DEFAULT`|int|Min LiDAR Points to Use LiDARパラメータのデフォルト値|
|`settings.LIDAR_MIN_POINTS_MAX`|int|Min LiDAR Points to Use LiDARパラメータの最大値|
|`settings.LIDAR_MAX_POINTS_FOR_DEPTH_DEFAULT`|int|Max LiDAR Points to Use Depthパラメータのデフォルト値|
|`settings.LIDAR_MAX_POINTS_FOR_DEPTH_MAX`|int|Max LiDAR Points to Use Depthパラメータの最大値|
|`settings.LIDAR_ROR_NB_POINTS_DEFAULT`|int|LiDAR ROR nb_pointsパラメータのデフォルト値|
|`settings.LIDAR_ROR_NB_POINTS_MAX`|int|LiDAR ROR nb_pointsパラメータの最大値|
|`settings.LIDAR_ROR_RADIUS_DEFAULT`|float|LiDAR ROR radiusパラメータのデフォルト値|
|`settings.LIDAR_ROR_RADIUS_MAX`|float|LiDAR ROR radiusパラメータの最大値|
|`settings.LIDAR_DBSCAN_EPS_DEFAULT`|float|LiDAR DBSCAN epsパラメータのデフォルト値|
|`settings.LIDAR_DBSCAN_EPS_MAX`|float|LiDAR DBSCAN epsパラメータの最大値|
|`settings.LIDAR_DBSCAN_MIN_SAMPLES_DEFAULT`|int|LiDAR DBSCAN min_samplesパラメータのデフォルト値|
|`settings.LIDAR_DBSCAN_MIN_SAMPLES_MAX`|int|LiDAR DBSCAN min_samplesパラメータの最大値|
|`settings.EGO_BOX_X_RANGE`|list[int]|LiDAR点群を削除する自車周辺の直方体領域のx座標。`[xmin, xmax]`の形式|
|`settings.EGO_BOX_Y_RANGE`|list[int]|LiDAR点群を削除する自車周辺の直方体領域のy座標。`[ymin, ymax]`の形式|
|`settings.EGO_BOX_Z_RANGE`|list[int]|LiDAR点群を削除する自車周辺の直方体領域のz座標。`[zmin, zmax]`の形式|
|`settings.DEPTH_CORRECTION_METHODS`|list[str]|LiDAR を基準に深度点群を補正する方式一覧（現状`["scale", "shift", "affine"]`の3種類）|
|`settings.DEFAULT_DEPTH_CORRECTION_METHOD`|list[str]|LiDAR を基準に深度点群を補正する方式。"scale"ならインスタンスごとDepth点群のz座標に`median(z_lidar / z_depth)`を掛けて補正。|
|`settings.EGO_REFERENCE_CHANNEL`|str|Depth点群の座標を揃えるための基準センサー名（各カメラごとに取得時のego_poseが異なるため、ここで指定したセンサのego_poseに揃えることで同一のグローバル座標に点群を移せる）|
|`settings.MERGE_METHODS`|list[str]|Inter-cam Merge Match Methodパラメータ（カメラ間同一インスタンス結合の判定方法）の選択肢|
|`settings.DEFAULT_MERGE_METHOD`|str|Inter-cam Merge Match Methodパラメータのデフォルト値|
|`settings.DEFAULT_MERGE_OVERLAP_THRESHOLD`|float|Inter-cam Merge Overlap Thresholdパラメータのデフォルト値|
|`settings.MERGE_OVERLAP_THRESHOLD_MAX`|float|Inter-cam Merge Overlap Thresholdパラメータの最大値|
|`settings.DEFAULT_MERGE_MAX_CENTROID_DISTANCE`|float|Inter-cam Merge Max Centroid Distanceパラメータのデフォルト値|
|`settings.MERGE_MAX_CENTROID_DISTANCE_MAX`|float|Inter-cam Merge Max Centroid Distanceパラメータの最大値|
|`settings.DEFAULT_MERGE_MIN_MATCH_FRAMES`|int|Inter-cam Merge Min Match Framesパラメータのデフォルト値|
|`settings.MERGE_MIN_MATCH_FRAMES_MAX`|int|Inter-cam Merge Min Match Framesパラメータの最大値|
|`settings.DEFAULT_MERGE_MAX_SAME_CAMERA_GAP`|int|Inter-cam Merge Max Same-Camera Gapパラメータのデフォルト値|
|`settings.MERGE_MAX_SAME_CAMERA_GAP_MAX`|int|Inter-cam Merge Max Same-Camera Gapパラメータの最大値|
|`settings.DEFAULT_MERGE_LABEL_MATCH`|str|Inter-cam Merge Label Matchパラメータ（カメラ間同一インスタンス結合時にラベルまたはカテゴリグループの一致も考慮するか）のデフォルト値|
|`settings.BOXFIT_MIN_POINTS_DEFAULT`|int|Min Points to fit Boxパラメータのデフォルト値|
|`settings.BOXFIT_MIN_POINTS_MAX`|int|Min Points to fit Boxパラメータの最大値|
|`settings.BOXFIT_METHODS`|list[str]|Boxfitting Methodパラメータ（3Dバウンディングボックスのフィッティングに使用するアルゴリズム）の選択肢|
|`settings.BOXFIT_METHOD_DEFAULT`|str|Boxfitting Methodパラメータのデフォルト値|
|`settings.BOXFIT_ANGLE_STEP_DEG_DEFAULT`|float|convex_hull_moa angle_step_degパラメータのデフォルト値|
|`settings.BOXFIT_ANGLE_STEP_DEG_MIN`|float|convex_hull_moa angle_step_degパラメータの最小値|
|`settings.BOXFIT_ANGLE_STEP_DEG_MAX`|float|convex_hull_moa angle_step_degパラメータの最大値|
|`settings.BOXFIT_Z_PERCENTILE_LOW_DEFAULT`|float|convex_hull_moa z_percentileの下側値のデフォルト値|
|`settings.BOXFIT_Z_PERCENTILE_HIGH_DEFAULT`|float|convex_hull_moa z_percentileパラメータの上側値のデフォルト値|
|`settings.DEPTH_MAX_RUNS_PER_SCENE`|int|1シーンあたり保持するrunの上限|
|`settings.DEFAULT_GLOBAL_POINTCLOUD_EYE`|[float,float,float]|表示のみ使用。PointCloudタブでGlobal Viewボタンを押したときの視点方向|
|`settings.DEFAULT_GLOBAL_POINTCLOUD_UP`|[float,float,float]|表示のみ使用。PointCloudタブでGlobal Viewボタンを押したときの視点方向|
|`settings.POINTCLOUD_AXIS_LENGTH_M`|float|表示のみ使用。PointCloudタブの点群ビューに描く自車姿勢の軸の長さ|
|`settings.SHOW_RAW_LIDAR`|bool|表示のみ使用。PointCloudタブのRaw LiDARチェックボックスのデフォルト値|
|`settings.SHOW_RAW_LIDAR_GROUND`|bool|表示のみ使用。PointCloudタブのLiDAR Groundチェックボックスのデフォルト値|
|`settings.SHOW_RAW_DEPTH_POINTCLOUD`|bool|表示のみ使用。PointCloudタブのRaw Depthチェックボックスのデフォルト値|
|`settings.DEPTH_MAP_DOWNSCALE`|float|表示のみ使用。深度マップの保存解像度倍率。1/2にすると保存容量が1/4に削減できる|
|`settings.BOXFIT_STORED_POINTS_MAX`|float|表示のみ使用。DBに保存するインスタンス点群の上限（ボクセル間引き後）|
|`settings.POINTCLOUD_DISPLAY_MAX_POINTS`|float|表示のみ使用。Plotlyへ渡す総点群数の上限。これ以上の数の点群は間引かれる|
|`settings.DEPTH_MODEL_NAME`|str|runの記録に残す深度推定のモデル名（重み名を指定。推論サーバー側の実体と合わせる必要がある）|
|`settings.BOXFIT_STUB_DELAY_SEC`|float|モデル使用時には使用しない（スタブでの待ち時間を指定する）|

### 推論の実行単位

Depth Boxfittingは、キーフレームごとに実施する。

ただし前述のように、LiDAR点群はキーフレームのsweepに加え、LiDAR Sweepsパラメータで指定したsweep数分だけ結合した点群を使用することに注意（キーフレームから直近LiDAR Sweepsで指定した数だけのsweepを結合）。

### 推論モデルの実装

深度推定のDepth-Anything-3モデルは、HuggingFace版ではなくGitHubの公式リポジトリ版を使用するが、重みはHuggingFaceから自動ダウンロードされる（ホストとコンテナ両方で~/.cache/huggingfaceフォルダに格納される）。

Depth-Anything-3のモデルインスタンスは以下のように作成する

```python
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.alignment import compute_sky_mask

DA3_MODEL_NAME = "DA3METRIC-LARGE"
device = "cuda" if torch.cuda.is_available() else "cpu"
da3_model = DepthAnything3.from_pretrained(f"depth-anything/{DA3_MODEL_NAME}").to(device=device)
```

推論時は以下のようにカメラ・サンプルのループでDA3推論を実施して結果を保持し（get_metric_depth関数は推論された深度マップをメートル単位に修正し、内部パラメータのスケールを深度マップの解像度に合わせる。詳細は`webapp/inference/app/models_impl/da3_depth.py`参照）

```python
depth_est_results = {}
# Camera channel loop
for camera_channel in CAMERA_CHANNELS:
    max_track_id = -1  # Initialize max_track_id for each camera channel
    depth_est_results[camera_channel] = {}
    # Sample loop
    for sample_index in range(len(samples)):
        calibrated_sensor_cam = # そのサンプル・カメラのcalibrated_sensor
        ego_pose_cam = # そのサンプル・カメラのego_pose
        image = # そのサンプル・カメラのPIL.Image.Image形式のカメラ画像
        # Inference depth using DepthAnything3 (without pose conditioning)
        prediction = da3_model.inference([image])
        metric_depths, scaled_intrinsics = get_metric_depth(
            prediction,
            model_name=DA3_MODEL_NAME,
            camera_intrinsics=[calibrated_sensor_cam["camera_intrinsic"]],
            original_image_width=image.width,
            original_image_height=image.height
        )
        # Store the depth estimation results
        depth_est_results[camera_channel][sample_index] = {
            "metric_depth": metric_depths[0],
            "scaled_intrinsics": scaled_intrinsics[0],
            "non_sky_mask": compute_sky_mask(prediction.sky[0]) if prediction.sky is not None else None,
            "original_image_width": image.width,
            "original_image_height": image.height,
            "depth_image_width": metric_depths[0].shape[1],
            "depth_image_height": metric_depths[0].shape[0],
            "camera_translation": calibrated_sensor_cam["translation"],
            "camera_quaternion": calibrated_sensor_cam["rotation"],
            "ego_translation": ego_pose_cam["translation"],
            "ego_quaternion": ego_pose_cam["rotation"],
        }
```

上記に続く処理

- LiDAR点群の読込と地面除去
- インスタンスマスクのクロージング処理適用
- 推論で得られたdepth画像のマスク投影点群化とsky_mask、ROR・DBSCAN適用
- LiDAR点群のマスク投影とROR・DBSCANの適用
- インスタンスごとのDepth点群とLiDAR点群の混合
- カメラ間同一インスタンス結合
- Box Fitting

は、以下のように実施します

```python
import pypatchworkpp

# Frame loop
for sample_index in range(len(samples)):
    pointclouds_per_instance[sample_index] = {}

    # Read the LiDAR point cloud (keyframe LiDAR coordinates)
    lidar_data = get_lidar_pointcloud_in_sample(
        sample_index=sample_index,
        samples_in_scene=samples,
        sample_data_in_scene=sample_data,
        ego_poses_in_scene=ego_poses,
        calibrated_sensors_in_scene=calibrated_sensors,
        lidar_sensor_token=sensor_lookup["LIDAR_TOP"],
        nuscenes_root=NUSCENES_ROOT,
        nsweeps=NUM_LIDAR_SWEEPS,
        stack_result=False,
    )
    lidar_translation=lidar_data[-1]["lidar_translation"]
    lidar_quaternion=lidar_data[-1]["lidar_quaternion"]
    ego_translation=lidar_data[-1]["ego_translation"]
    ego_quaternion=lidar_data[-1]["ego_quaternion"]
    # Remove the ground points by Patchwork++
    nonground = []
    for sweep_lidar in lidar_data:
        pointcloud = np.hstack((sweep_lidar["points"], sweep_lidar["intensity"].reshape(-1, 1)))  # Combine points and intensity into a single array
        PatchworkPLUSPLUS.estimateGround(pointcloud)
        sweep_ground = PatchworkPLUSPLUS.getGround()
        sweep_nonground = PatchworkPLUSPLUS.getNonground()
        nonground.append(sweep_nonground)
    lidar_points = np.vstack(nonground)
    # Convert LiDAR coordinates to ego
    lidar_points_ego = transform_lidar_to_ego(lidar_points, lidar_translation, lidar_quaternion)

    # Camera channel loop
    for i_cam, camera_channel in enumerate(CAMERA_CHANNELS):
        ###### Depth map projection ######
        pointclouds_per_instance[sample_index][camera_channel] = {}
        depth_est_result = depth_est_results[camera_channel][sample_index]
        depth_image = depth_est_result["metric_depth"]
        # Resize the instance masks to the depth image size
        instance_masks = instance_tracking_results[camera_channel][sample_index]
        depth_instance_masks = [instance.convert_to_original_coordinates(
            original_width=depth_est_result["depth_image_width"],
            original_height=depth_est_result["depth_image_height"],
            input_width=depth_est_result["original_image_width"],
            input_height=depth_est_result["original_image_height"]
        ) for instance in instance_masks]
        # Instance mask closing operation to fill holes and remove noise
        closed_instance_masks = [mask_morphology(mask, kernel_sizes=CLOSING_KERNEL_SIZES, ratio_morphology=RATIO_MORPHOLOGY)
                                for mask in depth_instance_masks]
        # Get the pseudo-LiDAR point clouds from the depth image and instance masks
        depth_points_per_instance, colors = depth_map_to_point_cloud_per_instance(
            metric_depth=depth_image,
            camera_intrinsics=depth_est_result["scaled_intrinsics"],
            instances=closed_instance_masks,
            common_mask=depth_est_result["non_sky_mask"],
            color=category_color_dict,
            color_attr="label"
        )
        # Per instance processes
        for instance, inst_depth_points in zip(closed_instance_masks, depth_points_per_instance):
            inst_depth_points_ego = transform_cam_to_ego(
                inst_depth_points,
                camera_translation=depth_est_result["camera_translation"],
                camera_quaternion=depth_est_result["camera_quaternion"]
            )
            inst_depth_points_global = transform_ego_to_global(
                inst_depth_points_ego,
                ego_translation=depth_est_result["ego_translation"],
                ego_quaternion=depth_est_result["ego_quaternion"]
            )
            # Apply ROR and DBSCAN noise removal
            inst_depth_points_global = apply_ror_dbscan(inst_depth_points_global,
                                                        depth_ror_nb_points, depth_ror_radius,
                                                        depth_dbscan_eps, depth_dbscan_min_samples)
            
            ###### Project LiDAR points to instance mask ######
            inst_lider_points = instance_lidar_points(
                instance,
                lidar_points_ego,
                camera_translation=depth_est_result["camera_translation"],
                camera_quaternion=depth_est_result["camera_quaternion"]
            )
            inst_lider_points_global = transform_ego_to_global(
                inst_lider_points,
                ego_translation=depth_est_result["ego_translation"],
                ego_quaternion=depth_est_result["ego_quaternion"]
            )
            # Apply ROR and DBSCAN noise removal
            inst_lidar_points_global = apply_ror_dbscan(inst_lidar_points_global,
                                                        lidar_ror_nb_points, lidar_ror_radius,
                                                        lidar_dbscan_eps, lidar_dbscan_min_samples)
            # Use Depth only if the number of LiDAR points is lower than threshold
            if len(inst_lidar_points_global) < min_lidar_points_to_use:
                inst_points = inst_depth_points_global
            # Mix Depth and LiDAR instande points
            elif len(inst_lidar_points_global) < max_lidar_points_to_use_depth:
                inst_points = mix_depth_with_lidar(inst_depth_points_global, inst_lider_points_global)
            # Use LiDAR only if the number of LiDAR points is higher than threshold
            else:
                inst_points = inst_lider_points_global
            pointclouds_per_instance[sample_index][camera_channel][instance.box.track_id] = inst_points
        
    ##### Inter-cam instance points merge ######
    pointclouds_per_global_track = inter_cam_instance_merge(
        method="bev_convex_hull",
        label_match="category_group",
        overlap_threshold=0.15,
        max_centroid_distance=3.0,
        min_match_frames=1
    )

    for global_track_points in pointclouds_per_global_track:
        ##### Box fitting ######
        fitted_box = fit_convex_hull_moa(
            global_track_points,
            angle_step_deg=0.5,
            z_percentiles=(1, 99),
        )
```
