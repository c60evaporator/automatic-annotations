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
###### データ読込処理（省略） ######
# Frame loop
for sample_index in proc_sample_indices:
    # Camera channel loop
    for camera_channel in CAMERA_CHANNELS:
        # Category group loop
        for category_group in category_groups:
            image = images[sample_index][camera_channel]
            box_threshold = DET2D_DEFAULT_SCORE_THRESHOLDS[category_group]
            labels = label_groups[category_group]
            # Infer the image with GroundingDINO model
            predicted_boxes, caption = predict_multi_labels(
                model=groundingdino_model,
                image=image,
                labels=category_names_in_group,
                box_threshold=box_threshold,
            )
            # convert box coordinates from normalized to pixel coordinates
            for box in predicted_boxes:
                box.xyxy = box.xyxy * np.array([image.width, image.height, image.width, image.height])
```

## Instance Tracking

SAM2によるInstance Trackingは、`detection_2d_params`（Detection2Dでの推論実行単位でパラメータを保持するテーブル）のSample Intervalごとに実施する（間引き後のサンプル数N' × カメラ数C回推論が実施される）。
例えばSample Interval=4のとき、以下のように推論が行われる

- 1回目の推論: Sample0からSample4までの画像を入力データとして与え、Sample0のボックス（Detection2Dの結果）をプロンプトとして与えて推論（トラッキングpropagation）を実施
- 2回目の推論: Sample4からSample8までの画像を入力データとして与え、Sample4のボックス（Detection2Dの結果）をプロンプトとして与えて推論を実施。前回（1回目の推論）のtrack_idをどのように引き継ぐかは、後述の`track_id_inheritance`パラメータにより変わる
- 3回目の推論（以下略）

なお、入力データとして与えるのはキーフレームだけでなく、Sweeps per Sampleパラメータに応じて非キーフレームをほぼ等間隔となるよう選択して使用する。具体的なフレームは、以下の参考コードのように、キーフレームと前のキーフレームとの間をSweeps per Sampleパラメータ（参考コード中の`nsweeps`）で等分するするように選択される

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

インスタンスセグメンテーション・トラッキングの推論はこの`sam2_predictor`インスタンスを用いて実施するが、`track_id_inheritance`パラメータに"continuous_id"と"reverse_matching"どちらを指定するかで、各推論間のトラッキングIDの引き継ぎ方法を以下のように変える（どちらも今回のSample Intervalから次のSample Intervalまでの画像フレームをまとめて推論に使用する）

- `track_id_inheritance="continuous_id"`: 最初のフレームにDetection2Dのボックスプロンプトを与えた後ろ向きトラッキング結果のインスタンスと、最後のフレーム（次のSample Intervalの最初のフレームに相当）にDetection2Dのボックスプロンプトを与えて推論したインスタンスをIoUマッチングしてtrack_idを引き継ぐ（詳細は後述の例を参照）
- `track_id_inheritance="reverse_matching"`: 最初のフレームにボックスプロンプトを与えた後ろ向きトラッキング推論と、最後のフレームにボックスプロンプトを与えた前向きトラッキング推論を実施し、全フレームに対してIoUマッチングを実施してtrack_idを引き継ぐ（詳細は後述の例を参照）

#### track_id_inheritance="continuous_id"

例えばSample Interval=4のとき、以下のように推論が行われる

- 1回目の推論: Sample0からSample4までの画像を入力データとして与え、Sample0のボックス（Detection2Dの結果）をプロンプトとして与えて推論（トラッキングpropagation）を実施
- 2回目の推論: Sample4からSample8までの画像を入力データとして与え、Sample4のボックス（Detection2Dの結果）をプロンプトとして与えて推論を実施。このとき、前回の推論（1回目の推論）で伝播により得られたSample4のインスタンスと、今回の推論（2回目の推論）で得られたSample4のインスタンス同士で貪欲マッチングを実施し、IoUが閾値（後述）以上のインスタンスが存在すれば1回目の推論のtrack_idを引き継ぎ、存在しなければ新たにtrack_idを割り振ります。
- 3回目の推論（以下略）

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

#### track_id_inheritance="reverse_matching"
