# 推論のフロー

## Detection2D

GroundingDINOによるDetection2Dは、パラメータで指定したSample Intervalごとに実施する（間引き後のサンプル数N' × カメラ数C × カテゴリグループ数G回推論が実施される）。
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

SAM2によるInstance Trackingは、`detection_2d_params`（Detection2Dでの推論実行単位でパラメータを保持するテーブル）のSample Intervalごとに実施する。このSample Interval間のフレームをまとめてトラッキングに利用する（キーフレームだけでなく、Sweeps per Sampleパラメータに応じて非キーフレームをほぼ等間隔となるよう選択して使用する。このトラッキングに利用するフレーム群をブロックと呼ぶこととする）。
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
