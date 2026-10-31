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

UI（Detection2Dページの画面上部の「Inference Parameters」エクスパンダー）からは以下パラメータを指定できます。エクスパンダー内は処理の適用順に「Detection2D」「Re-Classification」「Duplicate Suppression」「Whole-Image Fallback」の4タブに分かれています（括弧内は現在の設定値）

|パラメータ名|タブ|型|内容|デフォルト値|範囲|
|---|---|---|---|---|---|
|Sample Interval|Detection2D|int|推論を実施するsample（キーフレーム）の間隔。最後のsampleは間隔に関わらず必ず含める|`settings.DET2D_DEFAULT_SAMPLE_INTERVAL`（3）|1〜10|
|Score Threshold|Detection2D|float|検出したバウンディングボックスのscore閾値にかける倍率。実際に適用する閾値は、カテゴリグループごとに異なる`settings.DET2D_DEFAULT_SCORE_THRESHOLDS`にここで選択した倍率を掛けたものとなる（1.0で頭打ち）|1.0|0.0〜2.0|
|NMS Threshold|Detection2D|float|検出したバウンディングボックスでNMSを実施する際に、IoUの閾値にかける倍率。実際に適用する閾値は、同ラベル間のbox結合はカテゴリグループごとに異なる`settings.DET2D_NMS_SAME_CLASS_IOUS`にここで選択した倍率を掛けたもの、別ラベル間のbox結合は`settings.DET2D_NMS_CROSS_CLASS_IOU`にここで選択した倍率を掛けたものとなる（いずれも1.0で頭打ち）|1.0|0.0〜2.0|
|Re-Classification Crop Margin|Re-Classification|float|SigLIP2の再判定用切り出し画像をGroudingDINOのバウンディングボックスから拡張する割合（ボックスの幅・高さに対する比）|`settings.DEFAULT_RECLASSIFICATION_CROP_MARGIN_RATIO`（0.1）|0.0〜`settings.RECLASSIFICATION_CROP_MARGIN_RATIO_MAX`|
|IoS Delete Threshold|Duplicate Suppression|float|ボックス同士を総当たりでIoSで結合判定する際の、IoSの閾値。この閾値以上のボックス同士が結合され、`settings.DET2D_IOS_DELETE_DIRECTION`でラベルごとに指定した大小どちらかのボックスのみが保持される。0で無効|`settings.DET2D_IOS_DELETE_THRESHOLD`（0.8）|0.0〜`settings.DET2D_IOS_DELETE_THRESHOLD_MAX`|
|Whole-Image Score Threshold|Whole-Image Fallback|float|Whole-Image再判定時のSigLIP2スコアの閾値。この閾値未満の結果は採用しない（論理削除として残るので、表示オプションのShow deletedで確認できる）|`settings.DEFAULT_WHOLE_IMAGE_SCORE_THRESHOLD`（0.04）|0.0〜`settings.WHOLE_IMAGE_SCORE_THRESHOLD_MAX`|
|Whole-Image Resize Ratio|Whole-Image Fallback|float|Whole-Image再判定時の画像リサイズ率|`settings.DEFAULT_WHOLE_IMAGE_RESIZE_RATIO`（0.5）|0.1〜`settings.WHOLE_IMAGE_RESIZE_RATIO_MAX`|

#### 設定ファイルから指定するパラメータ

設定ファイル`webapp/app/core/config.py`からは、以下のパラメータを設定できます

|設定名|型|内容|公式nuScenesでの適正値|適正値の根拠|
|---|---|---|---|---|
|`settings.NUSC_CATEGORY_TO_LABEL`|dict[str, str]|nuScenes形式データセットのcategoryをDetection2Dのラベルに変換するdict。keyがnuScenesのcagtegory名、valueがDetection2Dのラベル|[config.py参照](../webapp/app/core/config.py)|[MMDetection3Dのラベル](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0rc6/mmdet3d/datasets/nuscenes_dataset.py#L56)に準拠|
|`settings.LABEL_TO_NUSC_CATEGORY`|dict[str, str]|Detection2DのラベルをnuScenes形式データセットのcategoryに変換するdict。keyがDetection2Dのラベル、valueがnuScenesのcagtegory名|[config.py参照](../webapp/app/core/config.py)|[MMDetection3Dのラベル](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0rc6/mmdet3d/datasets/nuscenes_dataset.py#L56)に準拠|
|`settings.LABEL_TO_SUBLABEL`|dict[str, list[str]]|Detection2DのラベルをGroundingDINO推論で使用するサブラベルに変換するdict。keyがDetection2Dのラベル、valueがサブラベル|[config.py参照](../webapp/app/core/config.py)|公式データセットで見逃しを減らすよう調整（後でSigLIP2で再判定するため、過検出気味セッティングにしている）|
|`settings.LABEL_TO_CATEGORY_GROUP`|dict[str, str]|Detection2DのラベルをGrounidingDINO推論の実行単位であるカテゴリグループに変換するdict。keyがDetection2Dのラベル、valueがカテゴリグループ|[config.py参照](../webapp/app/core/config.py)|サイズや機能が似たオブジェクト同士をグルーピングした（vehicle系、road_object系、two_wheeler系、pedestrian系）|
|`settings.DET2D_DEFAULT_SAMPLE_INTERVAL`|int|Sample Intervalパラメータのデフォルト値|3|推論時間（小さなintervalはこれが増える）、短期間のみ出現するオブジェクトの見逃し防止（大きなintervalはこれが増える）、Tracking伝播（大きすぎるとオブジェクトを見失い、小さすぎると伝播の意味がなくなる）のバランスを取った|
|`settings.DET2D_DEFAULT_SCORE_THRESHOLDS`|dict[str, float]|GroundingDINOのスコア閾値のカテゴリグループごとのデフォルト値。keyがカテゴリグループ、valueがスコア閾値のデフォルト値。実際の推論時はこの値に前述のScore Thresholdパラメータを掛けた値未満のバウンディングボックスを削除する|[config.py参照](../webapp/app/core/config.py)|実データで見逃しを減らすよう調整|
|`settings.DET2D_NMS_SAME_CLASS_IOUS`|dict[str, float]|GroundingDINOの同ラベルNMSで使用する閾値のカテゴリグループごとのデフォルト値。keyがカテゴリグループ、valueがデフォルト値。実際の推論時はこの値に前述のNMS Thresholdパラメータを掛けた値以上の同ラベルバウンディングボックス同士をNMS結合する（実際にはしきい値を超えたら自動結合されるわけではなく、貪欲マッチングで結合対象を選ぶことに注意。またサブラベルではなくラベルで判定することにも注意）|[config.py参照](../webapp/app/core/config.py)|実データで誤結合とダブルカウントのバランスを取るよう調整|
|`settings.DET2D_NMS_CROSS_CLASS_IOU`|float|GroundingDINOの別ラベルNMSで使用する閾値のデフォルト値。実際の推論時はこの値に前述のNMS Thresholdパラメータを掛けた値以上の別ラベルバウンディングボックス同士をNMS結合する（実際にはしきい値を超えたら自動結合されるわけではなく、貪欲マッチングで結合対象を選ぶことに注意）|0.85|実データで誤結合とダブルカウントのバランスを取るよう調整|
|`settings.RE_CLASSIFICATION_CANDIDATES`|dict[str, dict[str,str]]|Detection2Dのラベルごとに、SigLIP2によるラベル再判定のラベル候補と最終ラベルを指定するdict。keyがDetection2Dが付けたラベル、valueのkeyがラベル候補、valueのvalueが最終的に割り当てるラベルとなる。最終的に割り当てるラベルがNoneの場合、そのボックスは削除する|[config.py参照](../webapp/app/core/config.py)|Detectionの間違いを、ラベル違いと過検出に分けて判断。ラベル違いは正しいラベルをkey・label共に指定し、過検出はその物体の真の種類（判定対象のラベルには含まれない）をkeyに、Noneをvalueに指定する|
|`settings.DEFAULT_RECLASSIFICATION_CROP_MARGIN_RATIO`|float|前述のRe-Classification Crop Marginパラメータのデフォルト値|0.1|実データで調整。多少ずれてもSigLIPの分類能力が高いので割とうまくいきそうな印象|
|`settings.RECLASSIFICATION_CROP_MARGIN_RATIO_MAX`|float|前述のRe-Classification Crop Marginパラメータの最大値|1.0|原理上の最大値|
|`settings.WHOLE_IMAGE_CANDIDATES`|dict[str, dict[str,str]]|Whole-Image再判定の、SigLIP2推論のラベル候補と最終ラベルを指定するdict。辞書のkeyがラベル候補、valueがそのラベル候補と判定されたときに最終的に割り当てるラベルを表す|[config.py参照](../webapp/app/core/config.py)|①truckやbus等の画面いっぱいに映ってDetection2Dで見逃されやすいオブジェクト、②road,building,vegetation等のありがちな背景クラスをkeyに加える。①はvalueを正解ラベル名、②はvalueをNoneとする|
|`settings.DEFAULT_WHOLE_IMAGE_SCORE_THRESHOLD`|float|Whole-Image Score Thresholdパラメータのデフォルト値|0.04|実データで調整。SigLIP2のスコアは確率ではないシグモイド値なので、1よりかなり小さな値が適正値になる|
|`settings.WHOLE_IMAGE_SCORE_THRESHOLD_MAX`|float|Whole-Image Score Thresholdパラメータの最大値|1.0|原理上の最大値|
|`settings.DEFAULT_WHOLE_IMAGE_RESIZE_RATIO`|float|Whole-Image Resize Ratioパラメータのデフォルト値|0.5|処理時間と物体の写りの綺麗さのバランス。画像サイズが500x300pixになるくらいが、SigLIP2 NaFlexでmax_num_patches=576程度となってちょうど良さそう|
|`settings.WHOLE_IMAGE_RESIZE_RATIO_MAX`|float|Whole-Image Resize Ratioパラメータの最大値|1.0|原理上の最大値|
|`settings.DET2D_IOS_DELETE_THRESHOLD`|float|IoS Delete Thresholdパラメータのデフォルト値|0.8|実データで誤結合とダブルカウントのバランスを取るよう調整|
|`settings.DET2D_IOS_DELETE_THRESHOLD_MAX`|float|IoS Delete Thresholdパラメータの最大値|1.0|原理上の最大値|
|`settings.DET2D_IOS_DELETE_DIRECTION`|dict[str, str]|ボックス同士を総当たりでIoSで結合判定する際の、ラベルごとに大小どちらのボックスを削除するかを指定する辞書。keyがラベル名を表し、valueが"small"なら小さい方の、"big"なら大きい方のボックスを削除|[config.py参照](../webapp/app/core/config.py)|同じ物体で複数box検出されやすいラベル（車両系）は"small"で小さい方を削除し、複数物体がまとめて検出されやすいラベル（barrier,traffic_cone等の小物体系）は"big"で大きい方を削除|
|`settings.DET2D_MAX_RUNS_PER_SCENE`|int|1シーンあたり保持するrunの上限|10|DB容量と利便性のバランスを取った|
|`settings.DET2D_MODEL_NAME`|str|runの記録に残すモデル名|"IDEA-Research/grounding-dino-base"|HuggingFaceのモデルIDを指定。推論サーバー側の`settings.GROUNDINGDINO_MODEL`と合わせる必要がある|
|`settings.DET2D_MANUAL_REPLACE_IOU`|float|再実行時、このIoU以上で重なる手修正ボックスがあれば、推論ボックスを手修正ボックスで置き換える（手修正ボックスは推論サーバーへは送らない）|0.5|手修正を再推論で失わないようにするため|
|`settings.DET2D_STUB_DELAY_SEC`|float \| None|スタブ（`USE_STUB_MODELS=true`）使用時の1推論あたりの待ち時間。本実装では使用しない|0.05|-|
|`settings.CAM_DISPLAY_ORDER`|list[str]|表示のみ使用。カメラ画像の表示順（2列グリッドの先頭2つが1行目）。ここに無いchannelは末尾にchannel名順で並ぶ|`["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_BACK"]`|確認頻度の高いCAM_FRONTを先頭にする|

なお`settings.LABEL_TO_SUBLABEL`では、同じサブラベルを複数のラベルに割り当てないこと（逆引きできなくなる。`label_service.validate_label_config`で検出される）

#### 推論サーバー側の設定

推論サーバーの設定ファイル[`webapp/inference/app/core/config.py`](app/core/config.py)（環境変数で上書き可）からは、以下を設定できます

|設定名|型|内容|値|
|---|---|---|---|
|`settings.GROUNDINGDINO_MODEL`|str|GroundingDINO（Transformers版）のHuggingFaceモデルID|"IDEA-Research/grounding-dino-base"|
|`settings.SIGLIP2_MODEL`|str|ラベル再判定・Whole-Image再判定に使うSigLIP2のHuggingFaceモデルID|"google/siglip2-so400m-patch16-naflex"|
|`settings.MAX_RESIDENT_MODELS`|int|同時にGPUへ載せるモデル数の上限。1のときGroundingDINOとSigLIP2は同時に載らないため、SigLIP2による再判定・Whole-Image再判定は全フレームの検出が終わった後に別フェーズとして実行される|1|
|`settings.USE_STUB_MODELS`|bool|trueならGPU不要のスタブ（`models_impl/*_stub.py`）で推論する（UIとジョブの流れの確認用）|False|

### 推論の実行単位

Detection2Dは、パラメータで指定したSample Intervalごとに実施する（間引き後のサンプル数N' × カメラ数C × カテゴリグループ数G回推論が実施される）。
例えばSample Interval=4、カテゴリグループ数3のとき、以下のように推論が行われる

- 1回目の推論: Sample0の画像を入力データとして与え、カテゴリグループ1に含まれる複数ラベルをプロンプトとして与えて推論を実施
- 2回目の推論: Sample0の画像を入力データとして与え、カテゴリグループ2に含まれる複数ラベルをプロンプトとして与えて推論を実施
- 3回目の推論: Sample0の画像を入力データとして与え、カテゴリグループ3に含まれる複数ラベルをプロンプトとして与えて推論を実施
- 4回目の推論: Sample4の画像を入力データとして与え、カテゴリグループ1に含まれる複数ラベルをプロンプトとして与えて推論を実施
- 5回目の推論（以下略）

### 推論モデルの実装

GroundingDINOのモデルは、Transformers版（`IDEA-Research/grounding-dino-base`。公式の重みgroundingdino_swinb_cogcoor.pthに相当）を使用する。重みはHuggingFaceから自動ダウンロードされ、HF_HOME（composeでホストの~/.cache/huggingfaceをマウント）に保存されるため、手動ダウンロードは不要。モデルIDは推論サーバーの`settings.GROUNDINGDINO_MODEL`で指定する。

GroundingDINOのモデルインスタンスは以下のように作成する

```python
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

GROUNDINGDINO_MODEL = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
# Build the GroundingDINO model and processor
groundingdino_processor = AutoProcessor.from_pretrained(GROUNDINGDINO_MODEL)
groundingdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDINGDINO_MODEL).to(device).eval()
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
LABEL_TO_SUBLABEL: dict[str, list[str]] = {
    "car": ["car", "van"],
    "truck": ["truck", "tank_truck", "truck_cab"],
    "construction_vehicle": ["construction_vehicle"],
    "bus": ["bus"],
    "trailer": ["trailer"],
    "barrier": ["road_barrier"],
    "traffic_cone": ["traffic_cone"],
    "motorcycle": ["motorcycle"],
    "bicycle": ["bicycle"],
    "pedestrian": ["pedestrian"],
}
###### データ読込処理（省略） ######
# Frame loop
for sample_index in proc_sample_indices:
    # Camera channel loop
    for camera_channel in CAMERA_CHANNELS:
        raw_boxes = []
        # Category group loop
        for category_group in category_groups:
            image = images[sample_index][camera_channel]
            # UIのScore Threshold / NMS Thresholdの倍率を掛ける（1.0で頭打ち）
            box_threshold = min(DET2D_DEFAULT_SCORE_THRESHOLDS[category_group] * score_threshold_ratio, 1.0)
            same_class_nms_iou = min(DET2D_NMS_SAME_CLASS_IOUS[category_group] * nms_threshold_ratio, 1.0)
            cross_class_nms_iou = min(DET2D_NMS_CROSS_CLASS_IOU * nms_threshold_ratio, 1.0)
            labels = [k for k, v in LABEL_TO_CATEGORY_GROUP.items() if v == category_group]
            sublabels = [sublabel for label in labels for sublabel in LABEL_TO_SUBLABEL[label]]
            # Infer the image with GroundingDINO model
            predicted_boxes, caption = predict_multi_labels(
                model=groundingdino_model,
                processor=groundingdino_processor,
                image=image,
                labels=sublabels, # 実際にプロンプトとして使用するラベル=サブラベルを渡す
                box_threshold=box_threshold,
                same_class_nms_iou=same_class_nms_iou,
                cross_class_nms_iou=cross_class_nms_iou,  # グループ内のクラス跨ぎNMS
                sublabel_to_label=sublabel_to_label # Reverse mapping of LABEL_TO_SUBLABEL
            )
            raw_boxes.extend(predicted_boxes)
        # グループ間の重複（同じ物体が別グループで二重検出）を、全グループが揃ってから潰す
        kept_boxes = cross_class_nms(raw_boxes, cross_class_nms_iou)
# 以降、GroundingDINOを解放してからSigLIP2を載せ、全フレームの
# ラベル再判定 → Whole-Image再判定 → IoSによる重複削除 を別フェーズで行う
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

UI（Instance Trackingページの画面上部の「Inference Parameters」エクスパンダー）からは以下パラメータを指定できます。Box Promptはタブの外（全処理に共通の入力）にあり、それ以外は「Instance Tracking」「Track ID Inheritance」の2タブに分かれています（括弧内は現在の設定値）

|パラメータ名|タブ|型|内容|デフォルト値|範囲|
|---|---|---|---|---|---|
|Box Prompt|（タブ外）|-|入力とするDetection2Dの結果を選択（`detection_2d_params`テーブルのレコードを、開始時刻・ボックス数・Sample Intervalとともにラジオボタン付きでリスト表示）|`started_at`が最新のレコード|-|
|Sweeps per Sample|Instance Tracking|int|トラッキングに使用する画像のキーフレームあたりフレーム数（1ならキーフレームのみを使用。非キーフレームは伝播にのみ使い、結果は保存しない）|`settings.DEFAULT_TRACKING_NUM_SWEEPS`（2）|1〜min(`settings.SWEEPS_PER_SAMPLE`, `MAX_SWEEPS_PER_SAMPLE`=10)|
|IoU Threshold|Track ID Inheritance|float|track_idのブロック間引き継ぎに使用するインスタンス同士のHungarian algorithmによるIoUマッチング後のIoU閾値|`settings.DEFAULT_TRACKING_IOU_THRESHOLD`（0.4）|0.0〜1.0|
|IoU Method|Track ID Inheritance|"box" or "mask"|上記IoUマッチングで使用するIoUの計算方法を、マスクの外接矩形同士のIoU（"box"）にするか、Mask IoU（"mask"）にするかを選択|`settings.DEFAULT_TRACKING_IOU_METHOD`（"box"）|-|
|IoU Label Match|Track ID Inheritance|"label", "category_group", or "none"|上記IoUマッチング時にラベル（"label"）またはカテゴリグループ（"category_group"）の一致も要求するか、ラベルを問わないか（"none"）を指定|`settings.DEFAULT_TRACKING_IOU_LABEL_MATCH`（"category_group"）|-|
|Track ID Inheritance|Track ID Inheritance|"continuous_id" or "forward_backward_matching"|IoUマッチングに使用する手法を指定する。両手法の詳細は後述の「推論モデルの実装」参照|`settings.DEFAULT_TRACK_ID_INHERITANCE`（"forward_backward_matching"）|-|

#### 設定ファイルから指定するパラメータ

設定ファイル`webapp/app/core/config.py`からは、以下のパラメータを設定できます

|設定名|型|内容|公式nuScenesでの適正値|適正値の根拠|
|---|---|---|---|---|
|`settings.SWEEPS_PER_SAMPLE`|int|nuScenesのキーフレームあたりのsweep数（カメラ12Hz÷キーフレーム2Hz）。Sweeps per Sampleパラメータと、Depth BoxfittingのLiDAR Sweepsパラメータの最大値に使う|6|nuScenesのセンサー仕様（Sweeps per Sampleはさらに`sweep_service.MAX_SWEEPS_PER_SAMPLE`=10で頭打ち）|
|`settings.DEFAULT_TRACKING_NUM_SWEEPS`|int|Sweeps per Sampleパラメータのデフォルト値|2|推論時間と伝播ミスのバランスをとるよう、実データで調整|
|`settings.DEFAULT_TRACKING_IOU_THRESHOLD`|float|IoU Thresholdパラメータのデフォルト値|0.4|結合見逃しと誤結合のバランスをとるよう、実データで調整|
|`settings.DEFAULT_TRACKING_IOU_METHOD`|"box" or "mask"|IoU Methodパラメータのデフォルト値|"box"|"mask"だと計算時間がネックとなるため|
|`settings.DEFAULT_TRACKING_IOU_LABEL_MATCH`|"label", "category_group" or None|IoU Label Matchパラメータのデフォルト値|"category_group"|"label"だとDetection2Dのラベルミスの影響を強く受け、Noneだと制約が弱く誤結合されやすいため、バランスを取って"category_group"|
|`settings.DEFAULT_TRACK_ID_INHERITANCE`|"continuous_id" or "forward_backward_matching"|Track ID Inheritanceパラメータのデフォルト値|"forward_backward_matching"|"continuous_id"だとIntervalの途中から現れた物体が見逃されるため|
|`settings.TRACKING_MAX_RUNS_PER_SCENE`|int|1シーンあたり保持するrunの上限|10|DB容量と利便性のバランスを取った|
|`settings.TRACKING_MODEL_NAME`|str|runの記録に残すモデル名（HuggingFaceのモデルIDを指定。推論サーバー側の`settings.SAM2_MODEL`と合わせる必要がある）|"facebook/sam2.1-hiera-large"|HuggingFaceのモデルIDを指定。推論サーバー側の実体と合わせる必要がある|
|`settings.DEFAULT_TRACKING_MASK_SCORE_THRESHOLD`|float|推論サーバーへ`mask_score_threshold`として送るが、現状のSAM2実装では使用していない（将来的にマスクにスコアを付けるトラッキングアルゴリズムを使用する場合のために準備）|0.5|-|
|`settings.TRACKING_STUB_DELAY_SEC`|float \| None|モデル使用時には使用しない（スタブでの待ち時間を指定する）|0.05|-|

#### 推論サーバー側の設定

推論サーバーの設定ファイル[`webapp/inference/app/core/config.py`](app/core/config.py)からは、以下を設定できます

|設定名|型|内容|値|
|---|---|---|---|
|`settings.SAM2_MODEL`|str|SAM2 Video（Transformers版）のHuggingFaceモデルID。webapp側の`settings.TRACKING_MODEL_NAME`と合わせる|"facebook/sam2.1-hiera-large"|

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

SAM2のモデルは、Transformers版の`Sam2VideoModel`（`facebook/sam2.1-hiera-large`）を使用する。重みはHuggingFaceから自動ダウンロードされ、HF_HOME（composeでホストの~/.cache/huggingfaceをマウント）に保存されるため、手動ダウンロードは不要。モデルIDは推論サーバーの`settings.SAM2_MODEL`で指定する。

SAM2のモデルインスタンスは以下のように作成する（CUDAではbfloat16で推論する）

```python
from transformers import Sam2VideoModel, Sam2VideoProcessor

SAM2_MODEL = "facebook/sam2.1-hiera-large"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
sam2_model = Sam2VideoModel.from_pretrained(SAM2_MODEL).to(device, dtype=dtype).eval()
sam2_processor = Sam2VideoProcessor.from_pretrained(SAM2_MODEL)
```

推論はブロックごとに`sam2_processor.init_video_session`で新しい推論セッション（inference_state）を作り、ボックスプロンプトを`add_inputs_to_inference_session`で追加したのち`sam2_model.propagate_in_video_iterator`で伝播させる（実装は`webapp/inference/app/models_impl/sam2_video.py`参照）。以下の実装例は公式リポジトリ版の書き方で処理の流れを示したもので、関数の引数は実装と異なる。

インスタンスセグメンテーション・トラッキングの推論はこのモデルを用いて実施しますが、`track_id_inheritance`パラメータに"continuous_id"と"forward_backward_matching"どちらを指定するかで、各推論間のtrack_idの引き継ぎ方法を以下のように変えます（どちらも今回のSample Intervalから次のSample Intervalまでの画像フレームをまとめて推論に使用する）

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
- カメラ → キーフレームのループで、以下手順でカメラ画像からDepth-Anything-3による深度推定を実施（MapAnythingは未実装）
    - 画像を入力してDepth-Anything-3で単眼深度推定推論
    - Mask機能により無効領域（空領域等）を保持
    - 内部パラメータを深度画像のサイズに合わせて補正
    - 補正後内部パラメータを用いて、深度画像をメートル単位に補正
    - 深度画像を`settings.DEPTH_MAP_DOWNSCALE`倍に縮小し（内部パラメータも同じ倍率で補正）、深度・内部パラメータ・無効領域マスクを`.npz`として`DERIVED_ROOT`に保存
- キーフレームのループで、LiDARの読込を実施（Use LiDARに関わらず常に実施。UIでの比較表示に使うため）
    - キーフレームから直近LiDAR Sweepsスイープ分のLiDARデータを読み込み、スイープごとにPatchwork++による地面判定を実行したのち、キーフレームのLiDAR座標に合わせて全sweep点群を合体する（Patchwork++のsensor_heightにはLiDARの実際の取り付け高さを渡す）
    - 合体したLiDAR点群をego座標に変換し、自車周辺の直方体領域（`settings.EGO_BOX_X_RANGE`,`settings.EGO_BOX_Y_RANGE`,`settings.EGO_BOX_Z_RANGE`で指定）の点を自車反射とみなして削除
    - 点群と地面判定結果を`.npz`として`DERIVED_ROOT`に保存
- キーフレーム → カメラのループ（パス1）で、以下手順でインスタンスごと点群を作成する（3Dバウンディングボックスのフィッティングは、全キーフレームの点群が揃ってからパス2で実施）
    - キーフレーム内の全インスタンスマスクを走査し、以下手順でインスタンスごとDepth点群を得る
        - インスタンスマスクにクロージング処理（膨張→収縮）を適用。クロージング処理のカーネルサイズは以下のように決める（いずれも奇数。偶数だとマスクが右下へずれるため）
            - 膨張: Dilationパラメータを使用
            - 収縮: 基本はErosionパラメータを使用するが、マスクの外接矩形の短辺（元画像のpx）がSmall Mask Short Side以下なら`Dilation + 2 × round((Erosion − Dilation) / 2 × Small Mask Erosion Ratio)`を使用する（片側で削る画素数に倍率を掛ける。Erosion ≤ DilationならErosionのまま）
        - 深度画像から無効領域マスクを用いて空領域を削除したのちインスタンスマスクに投影し、補正後内部パラメータを用いて点群に変換し、インスタンスごとDepth点群を得る（推論サーバーの`settings.DEPTH_MAX_DISTANCE`より遠い点は点群にしない）
    - Use LiDARの場合、キーフレームのLiDAR点群（地面点を除いたもの）をインスタンスマスク（元画像解像度に戻したもの）に投影し、インスタンスごとLiDAR点群を得る
    - Depth点群・LiDAR点群は、外部パラメータとego_poseを用いて、そのsampleの基準ego座標（`settings.EGO_REFERENCE_CHANNEL`のego_pose）に揃える（カメラごとにsample_dataの時刻が異なり、揃えないとカメラ間で最大0.6m程度ずれるため）
    - 以下のフィルタ処理をインスタンスごとDepth点群、LiDAR点群それぞれに加え、ノイズを除去する
        - フィルタ1: ROR/DBSCAN。RORとDBSCANによるノイズ除去を順番に適用する（カメラごと・結合前に適用し、結合後の再クラスタリングは行わない）。各種パラメータは以下の値を使用する
            - Depth点群
                - RORのnb_points: Depth ROR nb_pointsパラメータにラベルごとに異なる`settings.NB_POINTS_RATIO`を掛け、マスクの短辺がSmall Mask Short Side以下ならさらにSmall Mask nb_points / min_samples Ratioを掛ける（1を下回らない）
                - RORのradius: Depth ROR radiusパラメータをそのまま用いる
                - DBSCANのmin_samples: Depth DBSCAN min_samplesパラメータを用い、マスクの短辺がSmall Mask Short Side以下ならさらにSmall Mask nb_points / min_samples Ratioを掛ける
                - DBSCANのeps: Depth DBSCAN epsパラメータをそのまま用いる
            - LiDAR点群
                - RORのnb_points: LiDAR ROR nb_pointsパラメータにラベルごとに異なる`settings.NB_POINTS_RATIO`を掛ける
                - RORのradius: LiDAR ROR radiusパラメータをそのまま用いる
                - DBSCANのmin_samples: LiDAR DBSCAN min_samplesパラメータをそのまま用いる
                - DBSCANのeps: LiDAR DBSCAN epsパラメータをそのまま用いる
            - ROR/DBSCAN後のLiDAR点群の点数がMin LiDAR Points to use LiDAR未満なら、そのインスタンスのLiDAR点群は空にする（少なすぎる点は補正に使えないため。表示もしない）
        - インスタンスごとDepth点群の補正式選択: インスタンスごとLiDAR点群が存在する場合、LiDAR PointcloudタブのDepth Correctionパラメータに基づき、Depth点群の深さ方向座標zの補正式を選択します（zが補正前、z'が補正後）。この補正式は直後の仮補正と、Depth点群とLiDAR点群結合時の補正の両方に使用します
            - Depth Correction=scaleのとき:	z’ = s·z（s = median(z_lidar,i / z_pred,i)）
            - Depth Correction=shiftのとき: z’ = z + o（o = median(z_lidar,i − z_pred,i)）
            - Depth Correction=affineのとき: z’ = a·z + b（最小二乗。対応点が5点未満なら scale に切り替える）
        - インスタンスごとDepth点群の仮補正: インスタンスごとLiDAR点群が存在する場合、マスク内でLiDAR点が当たった画素の予測深度（Depth点群のz座標）z_predと、そのLiDAR点の深度z_lidarから、直前で選択した補正式に基づきインスタンスごとDepth点群を仮補正（フィルタ2と3を精度良く適用できるようにすることが目的。L 字フィルタのMarginや距離フィルタの長辺などは、補正前の深度だとずれるためですマスクの縁で背景を拾った画素も対応に入るので、係数としては信頼できません）
        - フィルタ2: L-shape Filter。GeneralタブのEnable L-shape Filterがオンかつインスタンスのラベルが`settings.LSHAPE_FILTER_LABELS`に含まれる場合、以下の「L-shape Filter」を実施し、フィッティングしたL字平面から自車側に外れた点を削除する。この処理はXY座標（BEV座標）のみを用いて実施し、Z座標は用いない。センサー位置はDepth点群ならそのカメラ、LiDAR点群ならLIDAR_TOP（キーフレーム時）の基準ego座標でのXYを用いる（センサーごとに手前の面が変わるため、カメラ間結合の前に適用する）
            - 手順1: ビンごとに輪郭点検出と重み付け（Support BandパラメータはInference ParametersのGeneralタブで、Azimuth Bin、Near QuantileはDepth点群に適用するものはDepth Estimationタブで、LiDAR点群で適用するものはLiDAR Pointcloudタブで指定）
                - 自車から見た方位角を、Azimuth Bin幅のビンに分ける
                - 各ビンで、自車からの距離のNear Quantileパーセンタイルの位置にある点（横方向座標はビンの中心）を輪郭点とする
                - 各ビンで、自車から輪郭点までの距離から ±Support Band以内にある点の数で、輪郭点に重み付けする
                - 点が3点未満、または手前輪郭のビンが3つ未満、または見えている面が1つも無い場合は、手順2以降を実施せず全点を残して終了
            - 手順2: 以下手順で直交L字平面のフィッティング（[こちらの論文](https://ieeexplore.ieee.org/document/7995698)がベース。Angle Stepは`settings.LSHAPE_ANGLE_STEP_DEG`、L-shape Violation WeightパラメータはGeneralタブで指定。Support Bandパラメータは手順1と同じ値を使用）
                - 向きθを0〜90°の範囲でAngle Step刻みにループし、その向きの軸と直交する軸の2軸に対し、以下の評価値が最大のθを採用する
                    - センサ位置と全輪郭点を2軸それぞれに射影し、センサの射影座標が輪郭点の`settings.LSHAPE_BOUNDARY_QUANTILE`よりも外側にあれば、その軸に直交する面はセンサから見えるとみなす。そうでなければその軸はセンサから見えないとみなして使用しない（例えば車を真横から見る場合、車の進行方向の軸に射影したセンサ座標は全輪郭点の中央付近にあるため、この軸に直交する車の前面や後面は見えないと考えると理解しやすい）。
                    - 「センサから見える」判定された軸に関して、軸方向のオフセット座標t（これがL字平面の位置を決める）を変化させ、t ± Support Bandの軸座標範囲にある輪郭点を「面に乗る点」、tよりSupport Band 以上手前にある輪郭点を「手前に残る点」とする
                    - 点あたりの評価値を`max(0, 1 − |輪郭点の射影値 − t| / Support Band)`とし、面に乗る点の評価値の合計 - (L shape Violation Weight × 手前に残る点の評価値の合計)」を軸評価値とする（L shape Violation Weightが大きいほどノイズを取り逃がし、小さいと真の点を消しやすい）
                    - 2軸それぞれについて軸評価値が最大となるtを採用
                    - 2軸にtを代入して得られる2平面（BEV座標上では2直線）に対し、以下手順で求めた2軸全体の評価値が最大となるθを最終的に採用
                        - 面に乗る点：各輪郭点を近いほうの平面だけに割り当て、その平面との距離で数える
                        - 手前に残る点：どれか1つの平面よりSupport Band以上手前にある輪郭点を、1回だけ数える
                - このθとtで決まる2つの平面（`settings.LSHAPE_BOUNDARY_QUANTILE`の条件を満たさない場合1平面または0平面）を、直交L字平面とする
            - 手順3: 角度の微調整（直交L字平面の角度を一定範囲内でずらし、最もフィットする角度を探す）
                - 手順2で決めた直交L字平面をBEV座標上で直交する2直線とみなし、線を1本ずつ、もう一方を固定したまま、角度を±`settings.LSHAPE_ADJUST_RANGE_DEG`の範囲で`settings.LSHAPE_ADJUST_STEP_DEG`刻みに振る
                - 各角度でオフセットtを手順2と同じ評価値・方法で決め直し、L字全体の評価値が最大の角度を採用する
                - 担当する輪郭点が3点未満の線は調整しない。センサーが面の内側に入ってしまう角度は除外する
                - 1本目を確定させてから2本目を調整する（順番に1回ずつ）
            - 手順4: L字平面手前の点の除去
                - 点群の各点に関して、以下手順でL字平面でより一定以上手前にあるかを判定し、条件を満たす場合インスタンスごと点群から削除する
                    - センサ位置から各点へ向かう視線が、各L字平面を越える距離を求める。視線と交わらない平面は判定対象外とし、どの平面とも交わらない点は以降の判定をしない（残す）
                    - 各L字平面を越える距離の最大値を基準距離とする（視線がL字平面で囲われた領域に入る距離を意味する）
                    - 以下のようにMarginを求める
                        - Depth点群の場合: Depth EstimationタブにあるMargin Min, Margin Ratioパラメータを用い、Margin = max(Margin Min, Margin Ratio × 基準距離)
                        - LiDAR点群の場合: LiDAR PointcloudタブにあるMarginをそのまま用いる（LiDARは距離測定精度が高いため、Depth点群のときのように距離に応じて動的に閾値は変えない。なお、LiDAR L-shape Margin < L-shape Support Bandなら、LiDAR L-shape Marginは判定に寄与しない）
                    - 次の2条件を両方満たす点を消す（基準より奥の点は消さない）
                        - 視線方向の不足量（基準距離 − 点の距離）＞ Margin
                        - 基準距離を決めた線から、垂直方向にも Support Band を超えて手前に出ている（浅い角度で見えている面の点は、視線方向の不足量だけが大きく出るので、この条件で守る）
        - フィルタ3: Distance Filter。ラベルごとに指定した`settings.DISTANCE_FILTER_METHODS`辞書（keyがラベル名、valueがフィルタ方法）に基づき、以下の「Distance Filter」を実施し、ラベルの規定サイズから大きく前後方に外れた点を削除する。L-shape Filterの直後に実施し、L-shape Filterと同じく仮補正後のDepth点群で判定する。この処理はXY座標（BEV座標）のみを用いて実施し、Z座標は用いない。点数が5点（`distance_filter.MIN_POINTS`）未満なら何もしない
            - 方法がNone（辞書にラベルが無い場合を含む）のとき: 何も実施しない（点群をそのまま残す）
            - 方法が"long_side"のとき: 以下で求めた前側距離から後側距離の間にある対象インスタンスの点群のみを残し、それ以外を削除（motorcycle, pedestrian, traffic_coneを想定）
                - 対象インスタンス点群のセンサからのBEV距離の分位点（GeneralタブのDistance Quantileパラメータで指定）を基準距離とする
                - 前側距離: 基準距離 - Distance Marginパラメータ
                - 後側距離: 基準距離 + 長辺（`settings.DEFAULT_BOX_SIZES`から取得。幅と長さのうち大きい方）+ Distance Marginパラメータ
                - `settings.DEFAULT_BOX_SIZES`にラベルが無い場合は何もしない
            - 方法が"dbscan"のとき: インスタンス点群にBEV座標でDBSCAN（GeneralタブのDistance DBSCAN eps / Distance DBSCAN min_samplesパラメータを使用。Depth・LiDAR共通）を適用し、点数上位2クラスタのうち手前にある方（各クラスタのセンサからの距離のDistance Quantile分位点が小さい方）のみを残す（bicycleで物体点群と背景点群に大きく別れるケースで、背景点群を削除することを意図）。全点がノイズ判定された場合は何もしない
    - 以下手順で、インスタンスごとLiDAR点群とDepth点群を混合（あるいはどちらかを選択）し、インスタンスごと点群を確定する
        - Use LiDARがオフ、またはインスタンスごとLiDAR点群が空（Min LiDAR Points to use LiDAR未満で空にした場合を含む）なら、Depth点群のみをインスタンスごと点群として使用する
        - インスタンスごとLiDAR点群が存在し、点数がMax LiDAR Points to use Depth未満（Max LiDAR Points to use Depth=0なら常に）なら、以下の手順でDepth点群を補正したのちLiDAR点群と混合する
            - インスタンスごとLiDAR点群（各点の深さ方向座標をz_lidarとする）がカメラに写っている画素を求め、周囲 (2k+1)×(2k+1) 画素の窓（kは`settings.DEPTH_CORRECTION_NEIGHBOR_RADIUS_PX`で指定）にあるDepth点群（ROR/DBSCAN, L-shape Filter, Distance Filter適用後）の予測深度の中央値をz_predとする。この範囲内にDepth点群が存在しない場合は対応点なしと判定
            - 対応点の数に応じて以下のようにDepth点群の深さ方向座標を補正する（補正前がz、補正後をz'とする。zは先ほどの仮補正前の座標を用いることに注意）
                - 対応点の数がMin Correspondences for Pointwise Correction未満のとき: z’ = z * 残った LiDAR 点の距離の中央値 ÷ 残った深度点の予測深度の中央値
                - 対応点の数がMin Correspondences for Pointwise Correction以上のとき: 直前で求めたz_predとz_lidarを用いて、先ほど選択した補正式に基づきz’を求める
            - 深さ方向座標補正後のDepth点群とLiDAR点群を合体させたものをインスタンスごと点群とする
            - 補正係数が求まらなかった場合（対応点が無い等）は、補正前のDepth点群のみをインスタンスごと点群とする（LiDAR点群とは混合しない）
        - インスタンスごとLiDAR点群の点数がMax LiDAR Points to use Depth以上（Max LiDAR Points to use Depth>0のとき）、またはフィルタ後のDepth点群が0点なら、Depth点群は使用せずにLiDAR点群のみをインスタンスごと点群として使用する
        - Depth点群・LiDAR点群がともに0点なら、そのインスタンスは`status="no_points"`として以降の処理を行わない
- パス1が全キーフレームで終わったら（パス2）、以下手順でシーン全体を通して複数カメラ間での同一インスタンス結合を実施してglobal_track_idを割り当てる。なお結合を実施しない場合（Inter-cam Mergeタブの「カメラ間の結合を行う」がオフ）は照合を行わず、（カメラ, track_id）ごとに global_track_idを割り当てる（結合の有無に関わらず、以降のフィッティング・寸法補正はglobal_track_id単位で同じ流れで行う）
    - 各トラック（カメラ, track_id）のラベルを、トラック内の多数決（同数ならscore合計が大きい方）で決める
    - キーフレームごとに、別カメラのインスタンスごと点群の組み合わせについて、以下の条件で照合する（同一カメラ同士は照合しない）
        - 上から見たXY座標での凸包同士の隙間がMax Hull Gap以下（足切り。重なっていれば隙間は0）
        - Label Matchパラメータで指定した属性（トラックの多数決ラベル、またはそのカテゴリグループ）が一致する（"none"なら問わない）
        - 凸包のIoS（小さい方に対する重なり率）がOverlap Threshold以上
        - 条件を満たす組み合わせを、カメラ対ごとにHungarian algorithmで1対1に対応付ける
            - 例外処理として、Label Match=category_groupの場合に結合候補が2つ以上存在する場合、IoSが閾値を超えている限りは、IoSの大小よりもlabelの一致を優先する（labelが異なる結合をされにくくするため）
    - トラック対ごとに、対応付けられたキーフレーム数と最大IoSを集計し、Min Match Frames以上のキーフレームで対応付けられたトラック対を、ラベルが同じ対（Label Match=category_groupのとき）→ 最大IoSが大きい対の順に結合して、同一のglobal_track_idを割り振る。ただし、結合後のグループ内に以下の同一カメラのトラックが含まれることになる結合は却下する
        - 同一カメラで、存在するキーフレームが重なるトラック（同時に存在するなら別物体）
        - 同一カメラで、存在するキーフレームの隔たりがMax Same-Camera Gapを超えるトラック（Max Same-Camera Gap=0なら同一カメラのトラックは一切同居させない）
    - 結合されなかったトラックには個別のglobal_track_idを割り振る
    - global_track_idごとに全トラックのラベルを多数決し、それを全メンバーの最終ラベルとする
    - 結合しなかった対も含め、却下理由などの診断情報を`merge_diagnostics`としてrunに残す
- キーフレーム → global_track_idのループで、結合した点群に対して以下のようにバウンディングボックスをフィッティングする
    - 同一キーフレーム・同一global_track_idのメンバー（複数カメラのインスタンス）の点群を結合する。代表行（点数が最も多いメンバー）にのみボックスを持たせ、他のメンバーは`status="merged"`とする
    - 結合後の点数がMin Points to fit Box未満なら、そのキーフレームではバウンディングボックスをフィッティングしない（`status="too_few_points"`。点群は保持し、トラックも残る）
    - ラベルごとに指定した`settings.BOXFIT_METHODS`辞書（keyがラベル名、valueが推定方法。辞書に無いラベルは"convex_hull_moa"）に基づき、以下のようにバウンディングボックスの平面方向の大きさと角度（yaw角）を推定する
        - `settings.BOXFIT_METHODS="convex_hull_moa"`のとき（現状は全ラベルこの手法）
            - [こちらの手法](https://arxiv.org/abs/2302.01034)で3Dバウンディングボックスの平面方向の大きさとyaw角を推定（BEVの凸包に対し、yaw角をBox FittingタブのAngle Step刻みに振り、センサーから見たオクルージョン面積が最小になる向きを選ぶ）。センサー原点には、グループ内で点数が最も多いカメラの位置を使う
            - 長辺をlength、短辺をwidthとする
            - 凸包の面積 ÷ 矩形の面積（0〜1、大きいほど良い）を`fitting_score`とする
        - 他の手法（PartHOE等の向き推定と組み合わせるもの）は未実装
    - ラベルごとに指定した`settings.BOX_BOTTOM_ESTIMATIONS`辞書（keyがラベル名、valueが推定方法。辞書に無いラベルは"percentile"）に基づき、3Dバウンディングボックスの底面を推定する
        - `settings.BOX_BOTTOM_ESTIMATIONS="percentile"`のとき: global_track点群のz座標のパーセンタイル（Bottom Estimationのz_percentileパラメータで指定）に基づき底面高さを決める
        - `settings.BOX_BOTTOM_ESTIMATIONS="ground"`のとき: Patchwork++で判定した地面位置に基づき底面高さを決める（Use LiDAR時のみ適用可能。Use LiDARがオフならpercentileと同じ方法）。地面推定時には全ての地面点群を使うのではなく、Box外形からのXY平面上距離がGround Estimation Radius以内の地面点群を用い、Ground Height Methodパラメータに応じて以下のようにボックス底面高さを推定する。なお地面点群の数がMin Ground Points以下なら、`settings.BOX_BOTTOM_ESTIMATIONS="percentile"`のときの方法で底面高さを決める
            - Ground Height Method=percentileのとき: 地面点群のz座標のパーセンタイル（Ground Percentileパラメータで指定）をボックス底面高さとする
            - Ground Height Method=least_squaresのとき: 地面点群に最小二乗法で平面を当てはめ、この平面とボックスのXY方向中心が交わる位置の高さをボックス底面高さとする
        - `settings.BOX_BOTTOM_ESTIMATIONS="ego"`のとき: 基準ego座標系で高さ=0を底面高さとする（自車両と底面を揃える）
    - global_track点群のz座標の分位点（Top Estimationのz_percentileパラメータで指定）に基づき上面高さを決める（後で複数フレームを考慮して補正）
- global_track_idごとに以下手順でAIモデルによるボックス角度の再推定を実施（**未実装**。`settings.BOX_ORIENTATION_ESTIMATIONS`も未追加で、実装時に追加する。後述の疑似コードは実装方針を示すもの）
    - 当該global_trackに複数のカメラ画像が含まれる（複数カメラ間で同一インスタンスが結合された）場合、使用している点群の点数（Max LiDAR Points to use DepthパラメータでDepth点群が削除されている場合、削除された点数もカウントする）が最大のカメラを選択する
    - ラベルごとに指定した`settings.BOX_ORIENTATION_ESTIMATIONS`辞書（keyがラベル名、valueが補正方法）に基づき、以下の角度推定を実施。補正時はボックスのXY方向中心を固定し、yaw角のみを動かす
        - `settings.BOX_ORIENTATION_ESTIMATIONS="part_hoe"`のとき: PartHOEで補正
        - `settings.BOX_ORIENTATION_ESTIMATIONS="orient_anything_2"`のとき: Orient-Anything-2で補正
        - `settings.BOX_ORIENTATION_ESTIMATIONS=None`のとき: 補正しない（フィッティングした元々のボックスの方向をそのまま使用）
- 全キーフレームのボックスが揃ったら、global_track_idごとに所属する全てのフレームを考慮し、以下のようにバウンディングボックスの方向、高さ、幅、長さ補正を実施する（寸法補正は`webapp/inference/app/services/box_size.py`）
    - ラベルごとに指定した`settings.BOX_ORIENTATION_CORRECTIONS`辞書（keyがラベル名、valueが補正方法）に基づき、以下の角度補正を実施（**未実装**。`settings.BOX_ORIENTATION_CORRECTIONS`も未追加）
        - `settings.BOX_ORIENTATION_CORRECTIONS="part_hoe"`のとき: PartHOEで補正
        - `settings.BOX_ORIENTATION_CORRECTIONS=None`のとき: 補正しない
    - 現状は向きを揃える処理が無いため、各フレームのMOAの「長辺=length、短辺=width」をそのまま集計する（後ろ姿だけのフレームでは長辺と短辺が入れ替わり、向きも90°ずれることがある）
    - ラベルごとに指定した`settings.BOX_HEIGHT_CORRECTIONS`辞書（keyがラベル名、valueが補正方法）に基づき、以下の高さ補正を実施する。底面を固定して上面の位置を動かす
        - `settings.BOX_HEIGHT_CORRECTIONS="percentile"`のとき: 全フレームのボックス高さの分位点（Height Percentileパラメータで指定）をそのglobal_track共通のボックス高さとする。フレーム数がMin Frames for Size Correction未満なら補正しない
        - `settings.BOX_HEIGHT_CORRECTIONS="default"`のとき: そのラベルの`settings.DEFAULT_BOX_SIZES`の高さを、そのglobal_track共通のボックス高さとする
        - `settings.BOX_HEIGHT_CORRECTIONS=None`（辞書に無い場合を含む）のとき: 補正しない
    - 以下手順で、長さ方向（向きyawの軸）と幅方向（それに直交する軸）それぞれの面が見えているかを判定（センサー位置はフィッティング時のセンサー原点を使う）
        - 箱の中心からセンサーへのベクトルをd、その軸の単位ベクトルをu、軸の半分の長さをeとする
        - a = d·u（センサーがその軸の方向に中心からどれだけ離れているか）を求める
        - 以下のように面が見えているかを判定する
            - |a| > e：その面（a > 0 なら +u 側、a < 0 なら −u 側）は見えている（センサーは、その軸の片側の面より外側にいる）
            - |a| ≤ e：その面は見えていない（センサーはその軸の2面の間、すなわち面を真横から見る位置にいる）。この軸の面はどちらも見えていないので、中心を固定して左右対称に伸ばします。
    - ラベルごとに指定した`settings.BOX_LENGTH_CORRECTIONS`辞書（keyがラベル名、valueが補正方法）に基づき、以下の長さ補正を実施する。長さ方向の面（車両の場合前面または後面のうちセンサに近い方）が見えている場合、その面を固定して反対側の面の位置を動かす。見えていない場合は前面と後面の中心を固定して対称に両面を動かす
        - `settings.BOX_LENGTH_CORRECTIONS="percentile"`のとき: 全フレームのボックス長さの分位点（Length Percentileパラメータで指定）をそのglobal_track共通のボックス長さとする。フレーム数がMin Frames for Size Correction未満なら補正しない
        - `settings.BOX_LENGTH_CORRECTIONS="default"`のとき: そのラベルの`settings.DEFAULT_BOX_SIZES`の長さを、そのglobal_track共通のボックス長さとする
        - `settings.BOX_LENGTH_CORRECTIONS=None`（辞書に無い場合を含む）のとき: 補正しない
    - ラベルごとに指定した`settings.BOX_WIDTH_CORRECTIONS`辞書（keyがラベル名、valueが補正方法）に基づき、以下の幅補正を実施する（長さ補正で動かした後の中心を基準に判定する）。幅方向の面（車両の場合両側面のうちセンサに近い方）が見えている場合、その面を固定して反対側の面の位置を動かす。見えていない場合は両側面の中心を固定して対称に両面を動かす
        - `settings.BOX_WIDTH_CORRECTIONS="percentile"`のとき: 全フレームのボックス幅の分位点（Width Percentileパラメータで指定）をそのglobal_track共通のボックス幅とする。フレーム数がMin Frames for Size Correction未満なら補正しない
        - `settings.BOX_WIDTH_CORRECTIONS="default"`のとき: そのラベルの`settings.DEFAULT_BOX_SIZES`の幅を、そのglobal_track共通のボックス幅とする
        - `settings.BOX_WIDTH_CORRECTIONS=None`（辞書に無い場合を含む）のとき: 補正しない
    - 補正前の寸法と固定の仕方は`fit_metrics["size_correction"]`に記録する

### 推論に使用するパラメータ

Depth Boxfittingでは、以下のパラメータを指定できます

#### UIで指定可能なパラメータ

UI（Depth Boxfittingページの画面上部の「Inference Parameters」エクスパンダー）からは以下パラメータを指定できます。Instance Trackingはタブの外にあり、それ以外は処理の適用順に「General」「Depth Estimation」「LiDAR Pointcloud」「Inter-cam Merge」「Box Fitting」の5タブに分かれています（括弧内は現在の設定値）

|パラメータ名|タブ|型|内容|デフォルト値|範囲|
|---|---|---|---|---|---|
|Instance Tracking|（タブ外）|-|入力とするInstance Trackingの結果を選択（`instance_tracking_2d_params`テーブルのレコードをラジオボタン付きでリスト表示）|`started_at`が最新のレコード|-|
|Use LiDAR|General|bool|インスタンスごとLiDAR点群を作成し、Depth点群の補正・混合に使うかどうかを指定（LiDARの読込自体は比較表示のため常に行う）|`settings.BOXFIT_USE_LIDAR_DEFAULT`（True）|-|
|Dilation|General（Mask Closing）|int（奇数）|インスタンスマスクのクロージング処理の膨張カーネルサイズ|`settings.MASK_DILATION_DEFAULT`（3）|1〜`settings.MASK_DILATION_MAX`（2刻み）|
|Erosion|General（Mask Closing）|int（奇数）|インスタンスマスクのクロージング処理の収縮カーネルサイズ|`settings.MASK_EROSION_DEFAULT`（5）|1〜`settings.MASK_EROSION_MAX`（2刻み）|
|Small Mask Short Side [px]|General（Small Mask）|int|各種小マスク向け倍率を適用するためのマスクの外接矩形の短辺サイズ閾値（元画像のpx）。これ以下なら収縮カーネルサイズにはSmall Mask Erosion Ratioを、Depth ROR nb_points, Depth DBSCAN min_samplesにはSmall Mask nb_points / min_samples Ratioを適用。0で無効|`settings.SMALL_MASK_SHORT_SIDE_DEFAULT`（60）|0〜`settings.SMALL_MASK_SHORT_SIDE_MAX`|
|Small Mask Erosion Ratio|General（Small Mask）|float|短辺がSmall Mask Short Side以下のマスクの収縮量に適用する倍率。適用後カーネルサイズは`Dilation + 2 × round((Erosion − Dilation) / 2 × Small Mask Erosion Ratio)`（Erosion ≤ Dilationなら適用しない）|`settings.SMALL_MASK_EROSION_RATIO_DEFAULT`（0.2）|0.0〜1.0|
|Enable L-shape Filter|General（L-shape Filter）|bool|L-shape Filterを実施するかどうか（対象ラベルは`settings.LSHAPE_FILTER_LABELS`）|`settings.LSHAPE_FILTER_ENABLED_DEFAULT`（True）|-|
|L-shape Support Band [m]|General（L-shape Filter）|float|L-shape Filterで輪郭点が面に「乗っている」とみなす幅。輪郭点の重み、L字当てはめの評価（面に乗る点・手前に残る点の判定）、除去条件（面から垂直方向にこれを超えて手前の点だけ消す）の4役を兼ねる|`settings.LSHAPE_SUPPORT_BAND_DEFAULT`（0.3）|0.05〜`settings.LSHAPE_SUPPORT_BAND_MAX`|
|L-shape Violation Weight|General（L-shape Filter）|float|L字当てはめで、面より手前に残る輪郭点1つあたりの減点（面に乗る点1つに対する比）。大きいとL字が外側（センサー側）に寄り、小さいと点の多い内側に寄る|`settings.LSHAPE_VIOLATION_WEIGHT_DEFAULT`（5.0）|0.0〜`settings.LSHAPE_VIOLATION_WEIGHT_MAX`|
|Distance Quantile [%]|General（Distance Filter）|float|Distance Filterで使う、センサからのBEV距離の分位点（long_sideの基準距離、dbscanの「手前」の比較に使う）|`settings.DISTANCE_FILTER_QUANTILE_DEFAULT`（5.0）|0.0〜`settings.DISTANCE_FILTER_QUANTILE_MAX`|
|Distance Margin [m]|General（Distance Filter）|float|Distance Filter（long_side）で前側・後側の両方に取る余裕|`settings.DISTANCE_FILTER_MARGIN_DEFAULT`（0.3）|0.0〜`settings.DISTANCE_FILTER_MARGIN_MAX`|
|Distance DBSCAN eps [m]|General（Distance Filter）|float|Distance Filter（dbscan）のBEV DBSCANのeps（Depth・LiDAR共通）|`settings.DISTANCE_FILTER_DBSCAN_EPS_DEFAULT`（0.5）|0.05〜`settings.DISTANCE_FILTER_DBSCAN_EPS_MAX`|
|Distance DBSCAN min_samples|General（Distance Filter）|int|Distance Filter（dbscan）のBEV DBSCANのmin_samples（Depth・LiDAR共通）|`settings.DISTANCE_FILTER_DBSCAN_MIN_SAMPLES_DEFAULT`（3）|1〜`settings.DISTANCE_FILTER_DBSCAN_MIN_SAMPLES_MAX`|
|nb_points|Depth Estimation（ROR）|int|インスタンスごとDepth点群に適用するRORのnb_pointsパラメータ（Depth ROR nb_points）|`settings.DEPTH_ROR_NB_POINTS_DEFAULT`（5）|1〜`settings.DEPTH_ROR_NB_POINTS_MAX`|
|Radius|Depth Estimation（ROR）|float|インスタンスごとDepth点群に適用するRORのradiusパラメータ（Depth ROR radius）|`settings.DEPTH_ROR_RADIUS_DEFAULT`（0.8）|0.1〜`settings.DEPTH_ROR_RADIUS_MAX`|
|eps|Depth Estimation（DBSCAN）|float|インスタンスごとDepth点群に適用するDBSCANのepsパラメータ（Depth DBSCAN eps）|`settings.DEPTH_DBSCAN_EPS_DEFAULT`（1.0）|0.1〜`settings.DEPTH_DBSCAN_EPS_MAX`|
|min_samples|Depth Estimation（DBSCAN）|int|インスタンスごとDepth点群に適用するDBSCANのmin_samplesパラメータ（Depth DBSCAN min_samples）|`settings.DEPTH_DBSCAN_MIN_SAMPLES_DEFAULT`（10）|2〜`settings.DEPTH_DBSCAN_MIN_SAMPLES_MAX`|
|Small Mask nb_points / min_samples Ratio|Depth Estimation（Small Mask）|float|短辺がSmall Mask Short Side以下のマスクのDepth ROR nb_points（`settings.NB_POINTS_RATIO`を掛けた後）とDepth DBSCAN min_samplesに適用する倍率|`settings.SMALL_MASK_NB_POINTS_RATIO_DEFAULT`（0.6）|0.1〜1.0|
|Azimuth Bin [deg]|Depth Estimation（L-shape Filter）|float|Depth点群のL-shape Filterで手前輪郭を作る方位ビンの幅|`settings.DEPTH_LSHAPE_BIN_DEG_DEFAULT`（0.5）|0.1〜5.0|
|Near Quantile [%]|Depth Estimation（L-shape Filter）|float|Depth点群のL-shape Filterで、各ビンの輪郭点とする距離の分位点|`settings.DEPTH_LSHAPE_QUANTILE_DEFAULT`（10.0）|0.0〜50.0|
|Margin Min [m]|Depth Estimation（L-shape Filter）|float|Depth点群のL-shape Filterの除去閾値`max(Margin Min, Margin Ratio × 基準距離)`の最小値|`settings.DEPTH_LSHAPE_MARGIN_MIN_DEFAULT`（0.3）|0.05〜2.0|
|Margin Ratio|Depth Estimation（L-shape Filter）|float|Depth点群のL-shape Filterの除去閾値の、基準距離に対する比（単眼深度の誤差は距離とともに大きくなるため）|`settings.DEPTH_LSHAPE_MARGIN_RATIO_DEFAULT`（0.03）|0.0〜0.2|
|Depth Correction|LiDAR Pointcloud|"scale", "shift", or "affine"|LiDARを基準にDepth点群の深さ方向座標を補正する方式（Use LiDARのときだけ有効）。scale: z' = s·z / shift: z' = z + o / affine: z' = a·z + b（対応点が5点未満ならscaleに切り替える）|`settings.DEFAULT_DEPTH_CORRECTION_METHOD`（"scale"）|`settings.DEPTH_CORRECTION_METHODS`|
|LiDAR Sweeps|LiDAR Pointcloud|int|キーフレームあたりで結合するLiDAR点群のsweep数（1ならキーフレームのみを使用）|`settings.LIDAR_NUM_SWEEPS_DEFAULT`（5）|1〜`settings.SWEEPS_PER_SAMPLE`|
|Min LiDAR Points to use LiDAR|LiDAR Pointcloud|int|インスタンスごとLiDAR点群（ROR/DBSCAN後）を使用するための点数の下限しきい値（これを下回ったインスタンスはLiDAR点群を持たず、Depth点群のみ使用する）|`settings.LIDAR_MIN_POINTS_DEFAULT`（6）|1〜`settings.LIDAR_MIN_POINTS_MAX`|
|Max LiDAR Points to use Depth|LiDAR Pointcloud|int|Depth点群を使用するためのLiDAR点数の上限しきい値（LiDAR点数がこれ以上のインスタンスはLiDAR点群のみ使用する）。0で無効（常にDepth点群も使う）|`settings.LIDAR_MAX_POINTS_FOR_DEPTH_DEFAULT`（100）|0〜`settings.LIDAR_MAX_POINTS_FOR_DEPTH_MAX`|
|Min Correspondences for Pointwise Correction|LiDAR Pointcloud|int|Depth点群の最終補正で、点ごとの対応（LiDAR点とその周囲の残った深度点の予測深度の中央値）から補正係数を求めるのに必要な対応点の数。届かなければ「残ったLiDAR点の深度の中央値 ÷ 残った深度点の予測深度の中央値」で求める|`settings.LIDAR_MIN_CORRESPONDENCES_DEFAULT`（3）|1〜`settings.LIDAR_MIN_CORRESPONDENCES_MAX`|
|nb_points|LiDAR Pointcloud（ROR）|int|インスタンスごとLiDAR点群に適用するRORのnb_pointsパラメータ（LiDAR ROR nb_points）|`settings.LIDAR_ROR_NB_POINTS_DEFAULT`（3）|1〜`settings.LIDAR_ROR_NB_POINTS_MAX`|
|Radius|LiDAR Pointcloud（ROR）|float|インスタンスごとLiDAR点群に適用するRORのradiusパラメータ（LiDAR ROR radius）|`settings.LIDAR_ROR_RADIUS_DEFAULT`（0.8）|0.1〜`settings.LIDAR_ROR_RADIUS_MAX`|
|eps|LiDAR Pointcloud（DBSCAN）|float|インスタンスごとLiDAR点群に適用するDBSCANのepsパラメータ（LiDAR DBSCAN eps）|`settings.LIDAR_DBSCAN_EPS_DEFAULT`（1.2）|0.1〜`settings.LIDAR_DBSCAN_EPS_MAX`|
|min_samples|LiDAR Pointcloud（DBSCAN）|int|インスタンスごとLiDAR点群に適用するDBSCANのmin_samplesパラメータ（LiDAR DBSCAN min_samples）|`settings.LIDAR_DBSCAN_MIN_SAMPLES_DEFAULT`（5）|2〜`settings.LIDAR_DBSCAN_MIN_SAMPLES_MAX`|
|Azimuth Bin [deg]|LiDAR Pointcloud（L-shape Filter）|float|LiDAR点群のL-shape Filterで手前輪郭を作る方位ビンの幅（LiDARは点が疎なのでDepthより広く）|`settings.LIDAR_LSHAPE_BIN_DEG_DEFAULT`（1.0）|0.1〜5.0|
|Near Quantile [%]|LiDAR Pointcloud（L-shape Filter）|float|LiDAR点群のL-shape Filterで、各ビンの輪郭点とする距離の分位点|`settings.LIDAR_LSHAPE_QUANTILE_DEFAULT`（10.0）|0.0〜50.0|
|Margin [m]|LiDAR Pointcloud（L-shape Filter）|float|LiDAR点群のL-shape Filterの除去閾値（LiDARは距離の誤差が小さいので固定値。Support Band未満なら判定に寄与しない）|`settings.LIDAR_LSHAPE_MARGIN_DEFAULT`（0.2）|0.05〜2.0|
|カメラ間の結合を行う|Inter-cam Merge|bool|カメラ間の同一インスタンス結合を実施するかどうかを指定。オフなら（カメラ, track_id）ごとにglobal_track_idを割り当てる|True|-|
|Match Method|Inter-cam Merge|str|カメラ間の同一インスタンス結合に使用する手法を選択。"BEV convex-hull"なら、XY平面での凸包の隙間がMax Hull Gap以下かつIoSがOverlap Threshold以上の対応がMin Match Frames以上のキーフレームで得られたトラック同士を、同一インスタンスと判定して結合|`settings.DEFAULT_MERGE_METHOD`（"BEV convex-hull"）|`settings.MERGE_METHODS`|
|Label Match|Inter-cam Merge|"label", "category_group", or "none"|カメラ間同一インスタンス結合時に、トラックの多数決ラベル（"label"）またはそのカテゴリグループ（"category_group"）の一致も要求するか、問わないか（"none"）を指定|`settings.DEFAULT_MERGE_LABEL_MATCH`（"category_group"）|-|
|Overlap Threshold|Inter-cam Merge|float|XY平面での凸包の重なり率（IoS: 小さい方の面積に対する重なり率）のしきい値|`settings.DEFAULT_MERGE_OVERLAP_THRESHOLD`（0.15）|0.0〜`settings.MERGE_OVERLAP_THRESHOLD_MAX`|
|Max Hull Gap [m]|Inter-cam Merge|float|凸包同士の隙間の上限（足切り）。これを超える組は重なりを計算しない（重なっていれば隙間は0なので、結合結果には影響せず、計算量と診断に残す範囲に効く）|`settings.DEFAULT_MERGE_MAX_HULL_GAP`（0.5）|0.0〜`settings.MERGE_MAX_HULL_GAP_MAX`|
|Max Same-Camera Gap|Inter-cam Merge|int|同一カメラの2トラックを同じグループに入れてよい時間的な隔たり（キーフレーム数）。これを超えて離れている場合、または同一キーフレームに同時に存在する場合は結合されない。0なら同一カメラのトラックは一切同居させない|`settings.DEFAULT_MERGE_MAX_SAME_CAMERA_GAP`（20）|0〜`settings.MERGE_MAX_SAME_CAMERA_GAP_MAX`|
|Min Match Frames|Inter-cam Merge|int|トラック対を結合するのに必要な、対応付けられたキーフレーム数|`settings.DEFAULT_MERGE_MIN_MATCH_FRAMES`（1）|1〜`settings.MERGE_MIN_MATCH_FRAMES_MAX`|
|Min Points to fit Box|Box Fitting（Common）|int|Box Fittingを行うための点数の下限しきい値。カメラ間結合後に、global_track_id単位・キーフレームごとに、実際にフィッティングへ渡す点群で判定する（下回ったキーフレームは3Dバウンディングボックスを推定しない）|`settings.BOXFIT_MIN_POINTS_DEFAULT`（10）|1〜`settings.BOXFIT_MIN_POINTS_MAX`|
|Angle Step [deg]|Box Fitting（convex_hull_moa）|float|convex_hull_moaで最もフィットするyaw角度を探索するステップ|`settings.BOXFIT_ANGLE_STEP_DEG_DEFAULT`（0.5）|`settings.BOXFIT_ANGLE_STEP_DEG_MIN`〜`settings.BOXFIT_ANGLE_STEP_DEG_MAX`|
|z_percentile|Box Fitting（Bottom Estimation）|float|底面の高さとして採用する、点群のz座標のパーセンタイル（percentile方式と、ground方式の切り替え先で使用）|`settings.BOXFIT_Z_PERCENTILE_LOW_DEFAULT`（1.0）|0.0〜50.0|
|Ground Estimation Radius [m]|Box Fitting（Bottom Estimation）|float|ground方式で使う地面点の、ボックス外形からのXY距離の上限|`settings.BOX_BOTTOM_GROUND_RADIUS_DEFAULT`（2.0）|0.5〜`settings.BOX_BOTTOM_GROUND_RADIUS_MAX`|
|Min Ground Points|Box Fitting（Bottom Estimation）|int|ground方式に必要な地面点の数。これ以下ならpercentile方式に切り替える|`settings.BOX_BOTTOM_MIN_GROUND_POINTS_DEFAULT`（10）|1〜`settings.BOX_BOTTOM_MIN_GROUND_POINTS_MAX`|
|Ground Height Method|Box Fitting（Bottom Estimation）|"percentile" or "least_squares"|ground方式で地面点から底面の高さを求める方法|`settings.BOX_BOTTOM_GROUND_HEIGHT_METHOD_DEFAULT`（"percentile"）|`settings.BOX_BOTTOM_GROUND_HEIGHT_METHODS`|
|Ground Percentile [%]|Box Fitting（Bottom Estimation）|float|Ground Height Method=percentileのとき、底面の高さとする地面点のz座標の分位点（歩道や縁石の点が混ざると路面より高く出るため、中央値より低め）|`settings.BOX_BOTTOM_GROUND_PERCENTILE_DEFAULT`（30.0）|0.0〜100.0|
|z_percentile|Box Fitting（Top Estimation）|float|上面の高さとして採用する、点群のz座標のパーセンタイル|`settings.BOXFIT_Z_PERCENTILE_HIGH_DEFAULT`（99.0）|50.0〜100.0|
|Min Frames for Size Correction|Box Fitting|int|percentile方式の寸法補正（高さ・幅・長さ）に必要なフレーム数。これ未満のglobal_trackは補正しない|`settings.BOX_SIZE_CORRECTION_MIN_FRAMES_DEFAULT`（3）|1〜`settings.BOX_SIZE_CORRECTION_MIN_FRAMES_MAX`|
|Height Percentile [%]|Box Fitting（Height Estimation）|float|percentile方式の高さ補正で、全フレームのボックス高さから取る分位点|`settings.BOX_HEIGHT_CORRECTION_PERCENTILE_DEFAULT`（80.0）|0.0〜100.0|
|Width Percentile [%]|Box Fitting（Width Estimation）|float|percentile方式の幅補正で、全フレームのボックス幅から取る分位点|`settings.BOX_WIDTH_CORRECTION_PERCENTILE_DEFAULT`（70.0）|0.0〜100.0|
|Length Percentile [%]|Box Fitting（Length Estimation）|float|percentile方式の長さ補正で、全フレームのボックス長さから取る分位点|`settings.BOX_LENGTH_CORRECTION_PERCENTILE_DEFAULT`（80.0）|0.0〜100.0|

なお、ボックスのフィッティング手法（`settings.BOXFIT_METHODS`）、底面の推定方式（`settings.BOX_BOTTOM_ESTIMATIONS`）、寸法補正の方式（`settings.BOX_HEIGHT_CORRECTIONS`等）、Distance Filterの方式（`settings.DISTANCE_FILTER_METHODS`）はラベルごとに設定ファイルで指定し、UIでは選択できない（Box Fittingタブ・Generalタブに現在の割り当てが表示される）

#### 設定ファイルから指定するパラメータ

設定ファイル`webapp/app/core/config.py`からは、以下のパラメータを設定できます。UIパラメータのデフォルト値・範囲として使う`*_DEFAULT` / `*_MAX` / `*_MIN`系の設定は、前述のUIパラメータの表に記載したとおり（例: `settings.DEPTH_ROR_NB_POINTS_DEFAULT`はDepth Estimationタブのnb_pointsのデフォルト値、`settings.DEPTH_ROR_NB_POINTS_MAX`はその最大値）のため、ここではそれ以外の設定を記載する

|設定名|型|内容|値|根拠|
|---|---|---|---|---|
|`settings.NB_POINTS_RATIO`|dict[str, float]|ROR nb_pointsにかけるラベルごとの倍率（Depth・LiDAR共通。四捨五入し、1を下回らない）|traffic_cone: 0.5, pedestrian: 0.8, 他: 1.0|小さい物体は点群が疎で、全ラベル共通のnb_pointsだと点が丸ごと消えるため|
|`settings.DEPTH_CORRECTION_METHODS`|list[str]|Depth Correctionパラメータの選択肢|`["scale", "shift", "affine"]`|-|
|`settings.DEPTH_CORRECTION_NEIGHBOR_RADIUS_PX`|int|Depth点群の最終補正で、LiDAR点に対応させる深度点を探す近傍の半径k（周囲(2k+1)×(2k+1)画素。深度マップの画素単位）|2|深度マップは元画像の`settings.DEPTH_MAP_DOWNSCALE`倍なので、元画像の約±4画素に相当|
|`settings.EGO_BOX_X_RANGE`|tuple[float, float]|LiDAR点群を自車反射として削除する自車周辺の直方体領域（ego座標）のx座標。`(xmin, xmax)`の形式|(-1.5, 4.2)|nuScenesのego車両（4.084 x 1.730 x 1.562 m）に余裕を持たせた。半径ではなく直方体で切るのは、自車のすぐ横の歩行者や前方1.5mの車を消さないため|
|`settings.EGO_BOX_Y_RANGE`|tuple[float, float]|同上のy座標。`(ymin, ymax)`の形式|(-1.3, 1.3)|同上|
|`settings.EGO_BOX_Z_RANGE`|tuple[float, float]|同上のz座標。`(zmin, zmax)`の形式|(-0.5, 2.3)|同上|
|`settings.EGO_REFERENCE_CHANNEL`|str|Depth点群・LiDAR点群の座標を揃えるための基準センサー名（各カメラごとに取得時のego_poseが異なるため、ここで指定したセンサのego_poseに揃えることで同一のego座標に点群を移せる）|"LIDAR_TOP"|LiDARを使わない構成では常に存在するカメラ（CAM_FRONT等）を指定すれば同じ仕組みで動く|
|`settings.LSHAPE_FILTER_LABELS`|list[str]|L-shape Filterを適用するラベル|`["car", "truck", "bus", "trailer", "construction_vehicle"]`|L字の形がはっきりしている車両系|
|`settings.LSHAPE_ANGLE_STEP_DEG`|float|L-shape FilterでL字の向きθを総当たりする刻み [度]|1.0|-|
|`settings.LSHAPE_ADJUST_RANGE_DEG`|float|L-shape Filterの手順3（角度の微調整）で各線の角度を振る幅（±）[度]。0で微調整しない|10.0|単眼深度では2面が直交からずれることがあるため|
|`settings.LSHAPE_ADJUST_STEP_DEG`|float|L-shape Filterの手順3（角度の微調整）の刻み [度]|1.0|-|
|`settings.LSHAPE_BOUNDARY_QUANTILE`|float|L-shape Filterで、その向きの面がセンサから見えているか（I字かL字か）の判定に使う輪郭点の分位点 [%]|5.0|-|
|`settings.DISTANCE_FILTER_METHODS`|dict[str, str \| None]|Distance Filterのラベルごとの方式（"long_side" / "dbscan"）。辞書に無いラベル・Noneは何もしない|bicycle: "dbscan", motorcycle / pedestrian / traffic_cone: "long_side"|bicycleは壁のすぐ手前を走るケースなどで背景の塊を落とすため|
|`settings.DEFAULT_BOX_SIZES`|dict[str, tuple[float, float, float]]|ラベルごとの想定サイズ（length, width, height）[m]。Distance Filter（long_side）の長辺と、寸法補正（default方式）に使う|[config.py参照](../webapp/app/core/config.py)（例: car (4.63, 1.97, 1.74)）|仮の値（nuScenesの各カテゴリの平均的な寸法の目安）。統計を取って差し替える予定|
|`settings.BOXFIT_METHODS`|dict[str, str]|ラベルごとのBox Fittingの手法。辞書に無いラベルは"convex_hull_moa"|全ラベル"convex_hull_moa"|向き推定（PartHOE / Orient Anything等）を足すときに、ラベルごとに手法を切り替える差し込み口として辞書にしている|
|`settings.BOX_BOTTOM_ESTIMATIONS`|dict[str, str]|ラベルごとの底面の推定方式（"percentile" / "ground" / "ego"）。辞書に無いラベルは"percentile"|全ラベル"ground"|LiDARの上下方向のビームは疎なので、分位点だと底面が高く（箱が低く）なりやすいため|
|`settings.BOX_BOTTOM_GROUND_HEIGHT_METHODS`|list[str]|Ground Height Methodパラメータの選択肢|`["percentile", "least_squares"]`|-|
|`settings.BOX_HEIGHT_CORRECTIONS`|dict[str, str \| None]|ラベルごとの高さ補正の方式（"percentile" / "default" / None）|全ラベル"percentile"|各フレームの箱は隠れや疎なビームで小さめに出るため、全フレームから共通の寸法を決める|
|`settings.BOX_WIDTH_CORRECTIONS`|dict[str, str \| None]|ラベルごとの幅補正の方式（"percentile" / "default" / None）|車両系（car, truck, bus, trailer, construction_vehicle）: "default", 他: "percentile"|幅は車種でそれほど変わらないので、車両系はdefaultに寄せる|
|`settings.BOX_LENGTH_CORRECTIONS`|dict[str, str \| None]|ラベルごとの長さ補正の方式（"percentile" / "default" / None）|全ラベル"percentile"|長さは車種によってまちまちなのでpercentileに寄せる|
|`settings.DEPTH_MAP_DOWNSCALE`|float|深度マップの保存解像度倍率（内部パラメータも同じ倍率で補正して保存する）。1/2にすると保存容量が1/4に削減できる|0.5|1 runあたり約565MB → 141MB|
|`settings.DEPTH_MAX_RUNS_PER_SCENE`|int|1シーンあたり保持するrunの上限。超えたら古いものから削除する|3|派生ファイルが1 runあたり約280MBあるため少なめにする|
|`settings.BOXFIT_STORED_POINTS_MAX`|int|DBに保存するインスタンス点群の上限点数（ボクセル間引き後）。フィッティングには間引き前の点群を使う|500|-|
|`settings.DEPTH_MODEL_NAME`|str|runの記録に残す深度推定のモデル名（HuggingFaceのモデルID。推論サーバー側の`settings.DEPTH_ANYTHING_MODEL`と合わせる必要がある）|"depth-anything/DA3METRIC-LARGE"|-|
|`settings.BOXFIT_STUB_DELAY_SEC`|float \| None|モデル使用時には使用しない（スタブでの待ち時間を指定する）|0.02|-|
|`settings.POINTCLOUD_DISPLAY_MAX_POINTS`|int|表示のみ使用。Plotlyへ渡す総点群数の上限。これ以上の数の点群は間引かれる|50000|超えると転送量と描画が重くなる|
|`settings.DEFAULT_GLOBAL_POINTCLOUD_EYE`|list[float]|表示のみ使用。PointCloudタブでGlobal Viewボタンを押したときの視線位置（global座標。ego_poseの回転を打ち消して適用）|[-1.0, -0.5, 1.5]|-|
|`settings.DEFAULT_GLOBAL_POINTCLOUD_UP`|list[float]|表示のみ使用。PointCloudタブでGlobal Viewボタンを押したときの上方向|[0.0, 0.0, 1.0]|-|
|`settings.POINTCLOUD_AXIS_LENGTH_M`|float|表示のみ使用。PointCloudタブの点群ビューに描く自車姿勢の軸の長さ [m]|5.0|-|
|`settings.SHOW_RAW_LIDAR`|bool|表示のみ使用。PointCloudタブのRaw LiDARチェックボックスのデフォルト値|False|-|
|`settings.SHOW_RAW_LIDAR_GROUND`|bool|表示のみ使用。PointCloudタブのLiDAR Groundチェックボックスのデフォルト値|False|-|
|`settings.SHOW_RAW_DEPTH_POINTCLOUD`|bool|表示のみ使用。PointCloudタブのRaw Depthチェックボックスのデフォルト値|False|-|

#### 推論サーバー側の設定

推論サーバーの設定ファイル[`webapp/inference/app/core/config.py`](app/core/config.py)からは、以下を設定できます

|設定名|型|内容|値|
|---|---|---|---|
|`settings.DEPTH_ANYTHING_MODEL`|str|Depth-Anything-3のモデル名（重みは`depth-anything/<モデル名>`からHF_HOMEへ自動ダウンロード）。DA3METRIC-LARGE / DA3NESTED-GIANT-LARGE(-1.1)に対応|"DA3METRIC-LARGE"|
|`settings.DEPTH_MAX_DISTANCE`|float|これより遠い深度の点は点群にしない [m]（遠方は誤差が大きく、入れるとクラスタリングが引きずられるため）|60.0|

### 推論の実行単位

Depth Boxfittingは、以下の段階ごとに実施する（進捗のtotalは3段階の合計）

1. 深度推定: カメラを外側、キーフレームを内側にループして、カメラフレームごとに実施（対象キーフレーム数 × カメラ数C回推論が実施される）
2. LiDAR統合: キーフレームごとに実施（Use LiDARに関わらず常に実施）。LiDAR点群はキーフレームのsweepに加え、LiDAR Sweepsパラメータで指定したsweep数分だけ結合した点群を使用することに注意（キーフレームから直近LiDAR Sweepsで指定した数だけのsweepを結合）
3. インスタンス点群作成（パス1）: カメラフレームごとに、そのフレームの全インスタンスをまとめて処理（深度マップを1フレームにつき1回だけ読むため）
4. カメラ間結合・Box Fitting・寸法補正（パス2）: シーン全体（入力したInstance Trackingの全キーフレーム）をまとめて実施

### 推論モデルの実装

深度推定のDepth-Anything-3モデルは、GitHubの公式リポジトリ版（コミットを固定してインストール）を使用するが、重みはHuggingFaceから自動ダウンロードされる（HF_HOMEにホストの~/.cache/huggingfaceフォルダがマウントされ、ホストと共有される）。モデル名は推論サーバーの`settings.DEPTH_ANYTHING_MODEL`で指定する。

Depth-Anything-3のモデルインスタンスは以下のように作成する

```python
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.alignment import compute_sky_mask

DA3_MODEL_NAME = "DA3METRIC-LARGE"
device = "cuda" if torch.cuda.is_available() else "cpu"
da3_model = DepthAnything3.from_pretrained(f"depth-anything/{DA3_MODEL_NAME}").to(device=device)
```

推論時は以下のようにカメラ・サンプルのループでDA3推論を実施して結果を保持する（predict_metric_depth関数はDA3で推論し、内部で呼ぶget_metric_depth関数で深度マップをメートル単位に修正して内部パラメータのスケールを深度マップの解像度に合わせ、空マスクも返す。詳細は`webapp/inference/app/models_impl/da3_depth.py`参照。姿勢付き推論のpredict_pose_conditioned_depth関数は今後同じ形で追加する）

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
        metric_depths, scaled_intrinsics, non_sky_masks = predict_metric_depth(
            da3_model,
            DA3_MODEL_NAME,
            [image],
            camera_intrinsics=[calibrated_sensor_cam["camera_intrinsic"]],
        )
        # Store the depth estimation results
        depth_est_results[camera_channel][sample_index] = {
            "metric_depth": metric_depths[0],
            "scaled_intrinsics": scaled_intrinsics[0],
            "non_sky_mask": non_sky_masks[0] if non_sky_masks is not None else None,
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

- LiDAR点群の読込と地面判定・自車反射の削除
- インスタンスマスクのクロージング処理適用
- 推論で得られたdepth画像のマスク投影点群化と無効領域マスク適用でインスタンスごとdepth点群を得る
- LiDAR点群のマスク投影でインスタンスごとLiDAR点群を得る
- インスタンスごとdepth点群、LiDAR点群にROR・DBSCAN、L-shape Filter、Distance Filter適用
- インスタンスごとのdepth点群とLiDAR点群の混合
- カメラ間同一インスタンス結合
- Box Fitting（底面推定を含む）
- ボックス角度再推定（未実装）
- global_track_idごとに所属する全てのフレームを考慮してボックスの方向（未実装）、高さ、幅、長さ補正

は、以下のパイプラインで実装しています（実装は`webapp/inference/app/routers/depth_boxfitting.py`の`_run_boxfitting`、`webapp/inference/app/models_impl/boxfitting_pipeline.py`、`webapp/inference/app/services/`以下。関数の引数は実装より簡略化している）。点群はすべて、そのsampleの基準ego座標（`settings.EGO_REFERENCE_CHANNEL`のego_pose）で扱います。

#### パス1: インスタンスごと点群の作成

```python
# Frame loop
for sample_index in range(len(samples)):
    ###### LiDAR（Use LiDARに関わらず常に読み込む。UIの比較表示に使うため） ######
    # 直近NUM_LIDAR_SWEEPS個のsweepを読み、sweepごとにPatchwork++で地面判定してから
    # キーフレームのLiDAR座標に合わせて合体する（sensor_heightにはLiDARの取り付け高さを渡す）
    lidar_points, ground_mask = merge_sweeps_with_ground(lidar_sweeps[sample_index], sensor_height)
    lidar_points_ego = lidar_to_ego(lidar_points, lidar_calib)
    # 自車の反射を落とす（インスタンス抽出より前に落とさないと、DBSCANの最大クラスタとして勝ってしまう）
    lidar_points_ego, keep = remove_ego_points(lidar_points_ego, EGO_BOX_X_RANGE, EGO_BOX_Y_RANGE, EGO_BOX_Z_RANGE)
    ground_points = lidar_points_ego[ground_mask[keep]]       # 底面推定（ground方式）用
    nonground_points = lidar_points_ego[~ground_mask[keep]]   # インスタンス抽出用
    lidar_origin_xy = sensor_origin_xy(lidar_calib, lidar_ego_pose, reference_ego_pose)

    # Camera channel loop
    for camera_channel in CAMERA_CHANNELS:
        depth_est_result = depth_est_results[camera_channel][sample_index]  # .npzから読む
        camera_origin_xy = sensor_origin_xy(camera_calib, camera_ego_pose, reference_ego_pose)
        # 全画素の点群は1回だけ作り、マスクで切り出す（DEPTH_MAX_DISTANCEより遠い点は捨てる）
        all_depth_points = depth_map_to_point_cloud(depth_est_result["metric_depth"],
                                                    depth_est_result["scaled_intrinsics"],
                                                    depth_threshold=DEPTH_MAX_DISTANCE)

        # Per instance processes
        for instance in instance_tracking_results[camera_channel][sample_index]:
            label = instance.label
            full_mask = decode_rle(instance.mask_rle)
            small = is_small_mask(full_mask, SMALL_MASK_SHORT_SIDE)  # 外接矩形の短辺で判定（元画像のpx）
            mask = resize_mask_nearest(full_mask, depth_height, depth_width)
            # Instance mask closing（小物体は収縮を弱める）
            erosion = small_mask_erosion(DILATION, EROSION, SMALL_MASK_EROSION_RATIO) if small else EROSION
            closed = close_mask(mask, DILATION, erosion)

            ###### Depth point cloud + ROR/DBSCAN ######
            small_ratio = SMALL_MASK_NB_POINTS_RATIO if small else 1.0
            depth_points_cam = instance_points_from_depth(
                all_depth_points, closed, common_mask=depth_est_result["non_sky_mask"],
                ror_nb_points=DEPTH_ROR_NB_POINTS * NB_POINTS_RATIO[label] * small_ratio,
                ror_radius=DEPTH_ROR_RADIUS,
                dbscan_eps=DEPTH_DBSCAN_EPS,
                dbscan_min_samples=DEPTH_DBSCAN_MIN_SAMPLES * small_ratio,
            )

            ###### LiDAR point cloud + ROR/DBSCAN ######
            lidar_points_inst = empty()
            if USE_LIDAR:
                lidar_points_inst = instance_lidar_points(
                    nonground_points, resize_mask_nearest(closed, image_height, image_width), camera_calib)
                lidar_points_inst = filter_outliers(
                    lidar_points_inst,
                    ror_nb_points=LIDAR_ROR_NB_POINTS * NB_POINTS_RATIO[label],
                    ror_radius=LIDAR_ROR_RADIUS,
                    dbscan_eps=LIDAR_DBSCAN_EPS,
                    dbscan_min_samples=LIDAR_DBSCAN_MIN_SAMPLES,
                )
                if len(lidar_points_inst) < MIN_LIDAR_POINTS:
                    lidar_points_inst = empty()  # 少なすぎる点は補正に使えない

            ###### L-shape Filter / Distance Filter（LiDAR） ######
            lshape_enabled = LSHAPE_FILTER_ENABLED and label in LSHAPE_FILTER_LABELS
            if lshape_enabled and len(lidar_points_inst):
                keep, _ = lshape_filter(lidar_points_inst, lidar_origin_xy,
                                        bin_deg=LIDAR_LSHAPE_BIN_DEG, quantile=LIDAR_LSHAPE_QUANTILE,
                                        margin_min=LIDAR_LSHAPE_MARGIN, margin_ratio=0.0,  # LiDARは固定値
                                        **lshape_common_params)
                lidar_points_inst = lidar_points_inst[keep]
            distance_method = DISTANCE_FILTER_METHODS.get(label)
            if distance_method and len(lidar_points_inst):
                keep, _ = distance_filter(lidar_points_inst, lidar_origin_xy, method=distance_method,
                                          **distance_params(label))
                lidar_points_inst = lidar_points_inst[keep]

            if len(depth_points_cam) == 0 and len(lidar_points_inst) == 0:
                status = "no_points"
                continue
            depth_points = camera_to_reference(depth_points_cam, camera_calib, camera_ego_pose, reference_ego_pose)

            # Depth点群を使うか（LiDARが十分ならLiDARのみ）
            lidar_only = (len(depth_points) == 0
                          or (MAX_LIDAR_POINTS_FOR_DEPTH > 0
                              and len(lidar_points_inst) >= MAX_LIDAR_POINTS_FOR_DEPTH))

            ###### Depth点群の仮補正（フィルタの判定用） ######
            # クロージング後マスク全体の画素で係数を求める。L-shape / Distance Filterの
            # 距離[m]の閾値を正しく効かせるためだけに使う
            tentative = None
            use_correction = not lidar_only and USE_LIDAR and len(lidar_points_inst) > 0
            if use_correction:
                lidar_cam = ego_to_camera(lidar_points_inst, camera_calib)
                z_lidar, z_pred = sample_depth_at(lidar_cam, depth_image, intrinsics, mask=closed)
                initial_correction = fit_correction(z_lidar, z_pred, DEPTH_CORRECTION_METHOD)
                if initial_correction is not None:
                    tentative = camera_to_reference(apply_correction(depth_points_cam, initial_correction), ...)

            ###### L-shape Filter / Distance Filter（Depth） ######
            # 仮補正後の点群で判定し、同じマスクを補正前の点群にも当てる（補正は視線方向に動かすだけ）
            judge = tentative if tentative is not None else depth_points
            if lshape_enabled and len(depth_points):
                keep, _ = lshape_filter(judge, camera_origin_xy,
                                        bin_deg=DEPTH_LSHAPE_BIN_DEG, quantile=DEPTH_LSHAPE_QUANTILE,
                                        margin_min=DEPTH_LSHAPE_MARGIN_MIN,
                                        margin_ratio=DEPTH_LSHAPE_MARGIN_RATIO,
                                        **lshape_common_params)
                depth_points, depth_points_cam, judge = depth_points[keep], depth_points_cam[keep], judge[keep]
            if distance_method and len(depth_points):
                keep, _ = distance_filter(judge, camera_origin_xy, method=distance_method,
                                          **distance_params(label))
                depth_points, depth_points_cam = depth_points[keep], depth_points_cam[keep]

            ###### Depth点群の最終補正（フィルタ後に残った点だけで係数を求め直す） ######
            corrected = None
            if use_correction:  # 仮補正が求まらなかった場合も最終補正は試みる
                correction = refine_correction(
                    lidar_cam, depth_image, intrinsics, depth_points_cam, DEPTH_CORRECTION_METHOD,
                    radius=DEPTH_CORRECTION_NEIGHBOR_RADIUS_PX,      # 周囲(2k+1)×(2k+1)画素
                    min_correspondences=LIDAR_MIN_CORRESPONDENCES,  # 届かなければ中央値の比で求める
                )
                if correction is not None:
                    corrected = camera_to_reference(apply_correction(depth_points_cam, correction), ...)

            ###### Depth and LiDAR mixing ######
            if lidar_only:
                inst_points = lidar_points_inst
            elif corrected is not None:
                inst_points = np.vstack([corrected, lidar_points_inst])
            else:
                inst_points = depth_points  # 補正できなければDepthのみ（LiDARとは混ぜない）

            # 結合してから当てはめるので、ここではBEV凸包とz範囲の要約だけ持ち越す
            pointclouds_per_instance[sample_index][(camera_channel, instance.track_id)] = summarize_instance(
                inst_points, z_percentiles=(Z_PERCENTILE_LOW, Z_PERCENTILE_HIGH), sensor_origin_xy=camera_origin_xy)
```

#### パス2: カメラ間結合・Box Fitting・寸法補正

```python
###### Inter-cam instance points merge（シーン全体） ######
# 1) トラック（カメラ, track_id）ごとのラベルを多数決（同数ならscore合計）
track_label = {track: majority_label(labels, scores) for track, (labels, scores) in labels_by_track.items()}
# 2) キーフレームごと・カメラ対ごとに、BEV凸包で1対1の対応を求める（同一カメラ同士は照合しない）
frame_matches = []
for sample_index in range(len(samples)) if MERGE_ENABLED else []:
    edges = match_frame(
        pointclouds_per_instance[sample_index], track_label,
        overlap_threshold=MERGE_OVERLAP_THRESHOLD,   # 凸包のIoSの下限
        max_hull_gap=MERGE_MAX_HULL_GAP,             # 凸包の隙間による足切り
        label_match=MERGE_LABEL_MATCH,               # category_groupなら同ラベルを優先
    )                                                # カメラ対ごとにHungarianで1対1
    frame_matches.append(edges)
# 3) Min Match Frames以上対応したトラック対を、（同ラベル優先 →）最大IoSの大きい順に結合。
#    同時に存在する / 隔たりがMax Same-Camera Gapを超える同一カメラのトラックを含む結合は却下
global_ids = assign_global_track_ids(tracks, frame_matches,
                                     min_match_frames=MERGE_MIN_MATCH_FRAMES,
                                     max_same_camera_gap=MERGE_MAX_SAME_CAMERA_GAP,
                                     prefer_same_label=(MERGE_LABEL_MATCH == "category_group"))
# 4) global_track_idごとにラベルを多数決し、全メンバーの最終ラベルとする
global_label = {gid: majority_label(...) for gid in set(global_ids.values())}

###### Box fitting（キーフレーム × global_track_id） ######
fitted_rows = []
for sample_index in range(len(samples)):
    for global_id, members in group_by_global_id(pointclouds_per_instance[sample_index], global_ids).items():
        combined = combine_summaries(members)       # 凸包の和集合 = 結合後の凸包
        primary = max(members, key=num_points)      # 代表行（ボックスはここだけに持たせ、他はstatus="merged"）
        if combined.num_points < BOXFIT_MIN_POINTS:
            status = "too_few_points"               # このキーフレームだけボックスが欠ける（トラックは残る）
            continue
        label = global_label[global_id]
        origin_xy = dominant_sensor_origin(members)  # 点数が最も多いカメラの位置をMOAのセンサー原点にする
        fitted_box = fit_box(combined.points, BOXFIT_METHODS.get(label, "convex_hull_moa"),
                             angle_step_deg=BOXFIT_ANGLE_STEP_DEG,
                             z_range=combined.z_range,             # 上面: Z_PERCENTILE_HIGH分位点
                             sensor_origin_xy=origin_xy)
        if fitted_box is None:
            status = "not_fitted"
            continue
        # 底面をラベルごとの方式で決め直す（percentile / ground / ego）
        fitted_box = apply_bottom(fitted_box, BOX_BOTTOM_ESTIMATIONS.get(label, "percentile"),
                                  ground_points=ground_points[sample_index], use_lidar=USE_LIDAR)
        fitted_rows.append((global_id, label, fitted_box, origin_xy))

###### Box orientation re-estimation（未実装。実装方針） ######
for global_id, rows in group_by(fitted_rows, key=global_id).items():
    method = BOX_ORIENTATION_ESTIMATIONS.get(label)   # 設定は未追加
    if method is None:
        continue                                      # フィッティングした向きをそのまま使う
    for row in rows:
        ...                                           # 後述の疑似コード参照

###### Size correction per global track（全フレームのボックスが揃ってから） ######
correct_box_sizes(fitted_rows, box_fitting_params)   # 後述の疑似コード参照
```

#### L-shape Filter

実装は`webapp/inference/app/services/lshape_filter.py`。XY座標（BEV）のみを使い、物体は各線の内側（`u · x >= t`、uは内向き法線）にあるものとして扱う

```python
def lshape_filter(points, origin_xy, *, bin_deg, quantile, margin_min, margin_ratio,
                  angle_step_deg, adjust_range_deg, adjust_step_deg,
                  boundary_quantile, support_band, violation_weight):
    keep = np.ones(len(points), dtype=bool)
    if len(points) < 3:
        return keep                                   # too_few_points: 全点を残す
    xy = points[:, :2]

    ###### 手順1: 方位ビンごとの手前輪郭点と重み ######
    rel = xy - origin_xy
    ranges, azimuth = np.hypot(rel[:, 0], rel[:, 1]), np.arctan2(rel[:, 1], rel[:, 0])
    reference = circular_mean(azimuth)                # ±πをまたぐ物体でビンが分断されないように
    bins = floor(wrap_pi(azimuth - reference) / deg2rad(bin_deg))
    contour, weights = [], []
    for b in unique(bins):
        r = np.percentile(ranges[bins == b], quantile)                    # 距離のNear Quantile分位点
        a = reference + np.median(wrap_pi(azimuth[bins == b] - reference))
        contour.append(origin_xy + r * [cos(a), sin(a)])
        weights.append(np.sum(abs(ranges[bins == b] - r) <= support_band))  # 輪郭付近の点数で重み付け
    if len(contour) < 3:
        return keep                                   # too_few_bins

    ###### 手順2: 直交L字の当てはめ ######
    def best_offset(proj):
        # 物体は proj >= t 側。評価値 = 面に乗る点の多さ − violation_weight × 面より手前に残る点の数
        candidates = unique(proj)
        support = [sum(clip(1 - abs(proj - t) / support_band, 0, None) * weights) for t in candidates]
        in_front = [sum((proj < t - support_band) * weights) for t in candidates]
        score = support - violation_weight * in_front
        return candidates[argmax(score)]

    def visible_faces(axes):
        lines = []
        for axis in axes:
            proj = contour @ axis
            u = -axis if origin_xy @ axis >= np.median(proj) else axis   # センサー側を内向き法線の始点に
            if origin_xy @ u >= np.percentile(contour @ u, boundary_quantile):
                continue                              # センサーが物体の範囲内 → この向きの面は見えない
            lines.append((u, best_offset(contour @ u)))
        return lines

    def evaluate(lines):
        # 各輪郭点を近い方の線に割り当てて「乗る点」を数え、どれか1本よりsupport_band以上手前の点を1回だけ減点
        signed = stack([contour @ u - t for u, t in lines], axis=1)
        d = min(abs(signed), axis=1)
        return (sum(clip(1 - d / support_band, 0, None) * weights)
                - violation_weight * sum((signed.min(axis=1) < -support_band) * weights))

    best_lines, best_score = [], -inf
    for theta in np.arange(0.0, 90.0, angle_step_deg):
        axes = [[cos(theta), sin(theta)], [-sin(theta), cos(theta)]]
        lines = visible_faces(axes)
        if lines and evaluate(lines) > best_score:
            best_lines, best_score = lines, evaluate(lines)
    if not best_lines:
        return keep                                   # no_visible_face

    ###### 手順3: 角度の微調整（1本ずつ、もう一方を固定して） ######
    for i in range(len(best_lines)):
        if num_contour_points_assigned_to(best_lines, i) < 3:
            continue
        base = angle_of(best_lines[i][0])
        candidates = []
        for offset in np.arange(-adjust_range_deg, adjust_range_deg + 1e-9, adjust_step_deg):
            v = [cos(base + offset), sin(base + offset)]
            t = best_offset(contour @ v)
            if origin_xy @ v >= t:
                continue                              # センサーが面の内側に入る向きは除外
            trial = best_lines[:i] + [(v, t)] + best_lines[i + 1:]
            candidates.append((evaluate(trial), (v, t)))
        best_lines[i] = max(candidates)[1]

    ###### 手順4: L字の手前の点を除去 ######
    direction = rel / ranges[:, None]
    ref = perp = np.full(len(xy), np.nan)
    for u, t in best_lines:
        hit = (t - origin_xy @ u) / (direction @ u)  # 視線がこの面を越える距離（交わらなければnan）
        better = isnan(ref) | (hit > ref)            # 全ての面を越えた地点（最大値）が基準距離
        ref = where(better, hit, ref)
        perp = where(better, t - xy @ u, perp)       # 基準距離を決めた面から垂直方向の不足量
    deficit = ref - ranges                           # 視線方向の不足量
    margin = np.maximum(margin_min, margin_ratio * ref)
    front = (deficit > margin) & (perp > support_band)  # どの面とも交わらない点（nan）は残す
    return ~front
```

#### Distance Filter

実装は`webapp/inference/app/services/distance_filter.py`。XY座標（BEV）のみを使う

```python
def distance_filter(points, origin_xy, *, method, quantile, margin, long_side,
                    dbscan_eps, dbscan_min_samples):
    keep = np.ones(len(points), dtype=bool)
    if len(points) < 5:
        return keep                                   # too_few_points
    ranges = np.hypot(*(points[:, :2] - origin_xy).T)  # センサからのBEV距離

    if method == "long_side":
        if long_side is None:                         # DEFAULT_BOX_SIZESにラベルが無い
            return keep
        ref = np.percentile(ranges, quantile)         # Distance Quantile
        lower = ref - margin                          # 前側距離
        upper = ref + long_side + margin              # 後側距離（long_side = max(length, width)）
        return (ranges >= lower) & (ranges <= upper)

    if method == "dbscan":
        labels = dbscan_bev(points, eps=dbscan_eps, min_samples=dbscan_min_samples)
        if all(labels < 0):
            return keep                               # 全点ノイズなら絞り込まない
        top2 = two_largest_clusters(labels)
        # クラスタごとの距離のDistance Quantile分位点が小さい方（手前）を残す
        kept = min(top2, key=lambda c: np.percentile(ranges[labels == c], quantile))
        return labels == kept
```

#### ボックス角度再推定（未実装）

実装時の方針を示す疑似コード。`settings.BOX_ORIENTATION_ESTIMATIONS`（ラベル → "part_hoe" / "orient_anything_2" / None）は未追加で、実装時に追加する。グローバルトラックの全フレームを考慮した向きの補正（`settings.BOX_ORIENTATION_CORRECTIONS`）も同じく未実装

```python
def reestimate_orientation(fitted_rows, members_by_row, method_by_label):
    for global_id, label, box, origin_xy in fitted_rows:
        method = method_by_label.get(label)
        if method is None:
            continue                                  # フィッティングした向きをそのまま使う
        # 複数カメラで結合されている場合は、使用点数が最大のカメラの画像を使う
        # （Max LiDAR Points to use DepthでDepth点群を使わなかった場合も、その点数を数える）
        member = max(members_by_row[global_id, sample_index],
                     key=lambda m: m.num_points_used + m.num_points_depth_dropped)
        crop = crop_box(image_of(member), member.box_2d, margin=...)
        if method == "part_hoe":
            yaw_cam = part_hoe_predict(crop)
        elif method == "orient_anything_2":
            yaw_cam = orient_anything_2_predict(crop)
        # カメラ座標系の向きを基準ego座標系のyawへ変換し、XY中心を固定してyawのみ差し替える
        box["yaw_ego"] = camera_yaw_to_ego(yaw_cam, member.camera_calib, member.ego_pose, reference_ego_pose)
        # 長さ・幅は新しい向きの軸に合わせて点群から測り直す（向きが90°変わると長辺と短辺が入れ替わるため）
```

#### global_track_id単位の寸法補正

実装は`webapp/inference/app/services/box_size.py`

```python
def correct_box_sizes(fitted_rows, box_fitting_params):
    for global_id, rows in group_by(fitted_rows, key=global_id).items():
        label = rows[0].label
        targets = {}
        for dim in ("height", "width", "length"):
            method = {"height": BOX_HEIGHT_CORRECTIONS, "width": BOX_WIDTH_CORRECTIONS,
                      "length": BOX_LENGTH_CORRECTIONS}[dim].get(label)
            values = [row.box.size[dim] for row in rows]
            if method == "default":
                targets[dim] = DEFAULT_BOX_SIZES[label][dim]                       # (length, width, height)
            elif method == "percentile" and len(values) >= BOX_SIZE_CORRECTION_MIN_FRAMES:
                targets[dim] = np.percentile(values, PERCENTILE[dim])              # Height / Width / Length Percentile
            else:
                targets[dim] = None                                                # 補正しない
        if not any(targets.values()):
            continue

        for row in rows:                       # 各フレームの箱に当て直す
            box, origin_xy = row.box, row.origin_xy
            center = box.center_xy
            u = [cos(box.yaw), sin(box.yaw)]   # 長さ方向
            v = [-sin(box.yaw), cos(box.yaw)]  # 幅方向
            if targets["length"]:
                center = resize_axis(center, u, box.length / 2, targets["length"] / 2, origin_xy)
            if targets["width"]:
                # 長さ方向で動かした後の中心で判定する
                center = resize_axis(center, v, box.width / 2, targets["width"] / 2, origin_xy)
            if targets["height"]:
                bottom = box.center_z - box.height / 2                             # 底面を固定して上面を動かす
                box.center_z = bottom + targets["height"] / 2
            box.center_xy = center
            box.size = (targets["width"] or box.width, targets["length"] or box.length,
                        targets["height"] or box.height)


def resize_axis(center, axis, half, new_half, origin_xy):
    """見えている面を固定して反対側の面を動かす。見えていなければ中心を固定して対称に伸ばす"""
    a = (origin_xy - center) @ axis    # センサーがその軸方向に中心からどれだけ離れているか
    if abs(a) > half:                  # センサーはその軸の片側の面より外側 → その面は見えている
        side = sign(a)
        near_face = center + side * half * axis
        return near_face - side * new_half * axis
    return center                      # 2面の間（真横から見ている）→ 中心固定
```
