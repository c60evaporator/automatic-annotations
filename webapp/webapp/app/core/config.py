"""環境変数・設定（Pydantic Settings）"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """アプリケーション設定.

    値は .env または環境変数から読み込む。
    ホスト側のデータフォルダ（HOST_DATA_ROOT）は docker-compose 側で
    DATA_ROOT にマウントされる想定で、アプリからは DATA_ROOT のみを参照する。
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    ###### 共通設定 ######

    LOG_LEVEL: str = "INFO"

    # --- データ配置 ---
    # コンテナ内でのデータセットルート（ホスト側 HOST_DATA_ROOT のマウント先）
    DATA_ROOT: Path = Path("/data")
    # 推論の派生成果物（深度マップ .npz 等）の保存先
    DERIVED_ROOT: Path = Path("/data/_derived")

    # --- データベース ---
    SQLITE_PATH: Path = Path("/db/app.db")
    SQL_ECHO:    bool = False
    # 書き込みロック待ちの上限（ミリ秒）。推論結果の一括書き込みと
    # UI の読み取りが競合したときの "database is locked" を防ぐ
    SQLITE_BUSY_TIMEOUT_MS: int = 10_000

    # --- 推論サーバー ---
    # タイムアウトは用途ごとに差が大きいので、ここではなく
    # services/inference_client.py の定数で持つ
    # （投入・ポーリング・再フィルタでそれぞれ別の値）
    INFERENCE_BASE_URL: str = "http://inference:8000"

    # --- 起動環境 ---
    # このコンテナを起動した compose ファイル名（compose 側の environment で渡す）。
    # 画面に出す参考コマンド（データインポート等）の -f 指定に使う
    COMPOSE_FILE_NAME: str = "docker-compose.yml"

    # --- ラベル変換 ---
    NUSC_CATEGORY_TO_LABEL: dict[str, str] = {  # nuScenes category -> 検出ラベル
        "vehicle.car": "car",
        "vehicle.truck": "truck",
        "vehicle.construction": "construction_vehicle",
        "vehicle.bus.bendy": "bus",
        "vehicle.bus.rigid": "bus",
        "vehicle.trailer": "trailer",
        "movable_object.barrier": "barrier",
        "vehicle.motorcycle": "motorcycle",
        "vehicle.bicycle": "bicycle",
        "human.pedestrian.adult": "pedestrian",
        "human.pedestrian.child": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "movable_object.trafficcone": "traffic_cone",
    }
    LABEL_TO_NUSC_CATEGORY: dict[str, str] = {  # 検出ラベル -> nuScenes category
        "car": "vehicle.car",
        "truck": "vehicle.truck",
        "construction_vehicle": "vehicle.construction",
        "bus": "vehicle.bus.rigid",
        "trailer": "vehicle.trailer",
        "barrier": "movable_object.barrier",
        "motorcycle": "vehicle.motorcycle",
        "bicycle": "vehicle.bicycle",
        "pedestrian": "human.pedestrian.adult",
        "traffic_cone": "movable_object.trafficcone",
    }

    # --- 表示 ---
    # カメラ画像の表示順（channel 名）。
    # DB から取る順序は channel 名の昇順（CAM_BACK が先頭）になるため、
    # 確認頻度の高い CAM_FRONT を先頭に置く。
    # 画像は 2 列グリッドに並ぶので、先頭 2 つが 1 行目になる。
    # ここに無い channel は末尾へ回る（channel 名順）
    CAM_DISPLAY_ORDER: list[str] = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
    ]

    ###### Detection2D関係の設定 ######
    # ------ Detection2D (GroundingDINO推論) ------
    # 検出ラベル -> カテゴリグループ
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

    # ラベル -> GroundingDINO のプロンプトに使うサブラベル。
    #
    # ラベル名をそのまま投げると、語彙のずれで見逃しや取り違えが起きる
    # （タンクローリーが truck として拾われない、van が car と truck の
    #   どちらに寄るか安定しない、など）。呼び方の違いを列挙して
    # まとめて投げ、検出後にラベルへ畳み込む。
    #
    # NOTE: 同じサブラベルを複数のラベルに割り当てないこと。
    # どのラベルへ畳み込むか決まらなくなる（validate_label_config で検出する）。
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

    # 推論を実施するsample（キーフレーム）の間隔（大きすぎると自車や対象の速度が速い場面で取りこぼす可能性がある）
    DET2D_DEFAULT_SAMPLE_INTERVAL: int = 2
    # スコア閾値（Category Groupごとに指定。適用時はこれにUIのScore Thresholdパラメータで倍率をかける）
    DET2D_DEFAULT_SCORE_THRESHOLDS: dict[str, float] = {
        "vehicle": 0.23,
        "road_object": 0.25,
        "two_wheeler": 0.3,
        "pedestrian": 0.25,
    }
    # 同一label 内での NMS の閾値（Category Groupごとに指定。適用時はこれにUIのNMS Thresholdパラメータで倍率をかける）
    DET2D_NMS_SAME_CLASS_IOUS: dict[str, float] = {
        "vehicle": 0.7,
        "road_object": 0.6,
        "two_wheeler": 0.6,
        "pedestrian": 0.6,
    }
    # 異なるラベル間での NMS の閾値（全体で共通。適用時はこれにUIのNMS Thresholdパラメータで倍率をかける）
    DET2D_NMS_CROSS_CLASS_IOU: float = 0.6

    # ------ Re-classification (SigLIP2 によるラベル再判定) ------
    # GroundingDINO はプロンプト由来の取り違えが多いため、
    # ボックス周辺を切り出して zero-shot 分類で確認する。
    #   key   … Detection2D が付けたラベル（再判定の対象）
    #   value … {zero-shot のラベル候補: 最終的に割り当てるラベル}
    #           None は「このボックスを使わない」（論理削除）
    # ここに無いラベルは再判定しない
    RE_CLASSIFICATION_CANDIDATES: dict[str, dict[str, str | None]] = {
        "barrier": {"barrier": "barrier",
                    "fence": None,
                    "sidewalk": None,
                    "sign": None,
                    "toll payment machine": None,
                    "boom gate": None,
                    "bollard": None},
        "traffic_cone": {"traffic cone": "traffic_cone",
                         "bollard": None,
                         "road marking": None},
        "car": {"car": "car",
                "a photo of a car parked on the street": "car",
                "a close-up side view of a car": "car",
                "truck cab": "truck",
                "reflection of a car in a window": None,
                "painting": None},
        "construction_vehicle": {"construction vehicle": "construction_vehicle",
                                 "truck": "truck"},
        "bus": {"bus": "bus",
                "truck": "truck",
                "building": None},
        "motorcycle": {"motorcycle": "motorcycle",
                       "a photo of a car at night": "car",
                       "human legs": None,
                       "mural": None},
        "bicycle": {"bicycle": "bicycle",
                    "human legs": None},
        "pedestrian": {"pedestrian": "pedestrian",
                       "mural": None,
                       "rider": None},
    }

    # 切り出し時にボックスを広げる比率。
    # 文脈が写らないと zero-shot 分類が当たらないため、少し広めに取る
    DEFAULT_RECLASSIFICATION_CROP_MARGIN_RATIO: float = 0.1
    RECLASSIFICATION_CROP_MARGIN_RATIO_MAX: float = 1.0
    # 「使わない」候補（RE_CLASSIFICATION_CANDIDATES の値が None）の最大スコアが、残す候補の
    # 最大スコアをこの値より大きく上回ったときだけ削除する（SigLIP2 の sigmoid スコアの差）。
    # 0 ならスコア最大の候補で決める。正しい検出を消すと見逃しになるため、確信が高いときだけ消す
    DEFAULT_RECLASSIFICATION_DELETE_MARGIN: float = 0.002
    RECLASSIFICATION_DELETE_MARGIN_MAX: float = 0.05

    # ------ Duplicate Suppression (IoSによる重複検出の抑制) ------
    # 同じラベルのボックスがほぼ内包関係にあるとき、IoU の NMS では
    # 合体できない（大小差があると IoU が小さくなる）。
    # IoS（小さい方の面積に対する重なり率）で判定して片方を落とす。
    #
    # 実質的な重複検出が残ると、トラッキングで同じ物体に 2 つの
    # track_id が付くなど後段に響く
    DET2D_IOS_DELETE_THRESHOLD: float = 0.8
    DET2D_IOS_DELETE_THRESHOLD_MAX: float = 1.0
    # ラベルごとに、どちらを残すか。
    #   "small" … 小さい方を削除（物体の一部を拾った検出を落とす。車両向け）
    #   "big"   … 大きい方を削除（複数の物体をまとめて囲った検出を落とす。
    #             並ぶことが多い barrier / traffic_cone / 歩行者向け）
    #   "multi_big" … 子（内包される小さいボックス）が 2 個以上あり、子の上端から下端
    #             までが親の高さの DET2D_IOS_MULTI_BIG_VERTICAL_COVERAGE 倍を超えるときだけ、
    #             親を削除する。子は削除しない（複数台のまとめ検出は落としたいが、
    #             手前の車の背後に見える車は別物体として残したい car 向け）
    # ここに無いラベルは判定しない
    DET2D_IOS_DELETE_DIRECTION: dict[str, str] = {
        "car": "multi_big",
        "truck": "small",
        "construction_vehicle": "small",
        "bus": "small",
        "trailer": "small",
        "barrier": "multi_big",
        "traffic_cone": "big",
        "motorcycle": "big",
        "bicycle": "big",
        "pedestrian": "big",
    }
    DET2D_IOS_MULTI_BIG_VERTICAL_COVERAGE: float = 0.8

    # ------ Whole-Image Fallback (検出 0 件のフレームの救済) ------
    # 視野いっぱいに車両が写る構図では、輪郭が画面に収まらず
    # GroundingDINO が何も検出しないことがある。
    # そのフレームに限り、画像全体を 1 枚の切り出しとみなして分類する。
    #
    # 値が None の候補は **除外候補**（判定されても検出としない）。
    # 相対比較で落とせるので、閾値の較正に依存しないで済む
    WHOLE_IMAGE_CANDIDATES: dict[str, str | None] = {
        "bus": "bus",
        "truck cargo bed": "truck",
        "road": None,
        "building": None,
        "vegetation": None,
    }
    # 採用候補が勝った場合でも、このスコア未満なら採用しない。
    # 画像全体を覆うボックスになるので、誤ると後段への影響が大きい。
    #
    # NOTE: SigLIP のスコアは **確率ではない**（候補ごとに独立した sigmoid）。
    # 合計 1 になる制約がないため、全体的に低い値が出る。
    # 実データでの較正の結果この値にしている
    DEFAULT_WHOLE_IMAGE_SCORE_THRESHOLD: float = 0.04
    WHOLE_IMAGE_SCORE_THRESHOLD_MAX: float = 1.0
    # 分類へ渡す前の縮小率。元解像度のままでは前処理が重い
    DEFAULT_WHOLE_IMAGE_RESIZE_RATIO: float = 0.5
    WHOLE_IMAGE_RESIZE_RATIO_MAX: float = 1.0

    # ------ Human-in-the-Loop ------
    # 再実行時、この IoU 以上で重なる手修正ボックスがあれば、
    # 推論ボックスを手修正ボックスで置き換える（推論サーバーへは送らない）
    DET2D_MANUAL_REPLACE_IOU: float = 0.5

    # --- 記録・管理関係 ---
    # 1シーンあたり保持する run の上限。超えたら古いものから削除する。
    # ただし Instance Tracking から参照されている run は削除しない
    # （消すとトラッキング結果と 3D ボックスまで CASCADE で消えるため）
    DET2D_MAX_RUNS_PER_SCENE: int = 10

    # run の記録に残すモデル名（推論サーバー側の GROUNDINGDINO_MODEL と合わせる）
    DET2D_MODEL_NAME: str = "IDEA-Research/grounding-dino-base"

    # スタブ推論の1回あたりの待ち時間（本実装に差し替えたら None にする）
    DET2D_STUB_DELAY_SEC: float | None = 0.05

    ###### Instance Tracking関係の設定 ######
    # ------ SAM2 Tracking ------
    # キーフレームあたりのカメラフレーム数（カメラ12Hz÷キーフレーム2Hz、Sweeps per Sample パラメータの最大値）
    CAM_FRAMES_PER_SAMPLE: int = 6
    # Sweeps per Sample の既定値
    DEFAULT_TRACKING_NUM_SWEEPS: int = 2

    # ------ Track ID Inheritance ------
    DEFAULT_TRACKING_IOU_THRESHOLD: float = 0.4
    DEFAULT_TRACKING_IOU_METHOD: str = "box"
    DEFAULT_TRACKING_IOU_LABEL_MATCH: str = "category_group"
    DEFAULT_TRACKING_MASK_SCORE_THRESHOLD: float = 0.5
    # track_id の引き継ぎ方式。
    #   continuous_id            … Forward のみ。区間境界で次のプロンプトと照合
    #   forward_backward_matching … Forward と Backward を走らせ、区間内で照合。
    #                               区間途中に現れたインスタンスを拾える
    DEFAULT_TRACK_ID_INHERITANCE: str = "forward_backward_matching"

    # ------ 記録・管理関係 ------
    TRACKING_MAX_RUNS_PER_SCENE: int = 10
    TRACKING_MODEL_NAME: str = "facebook/sam2.1-hiera-large"  # 推論サーバー側の SAM2_MODEL と合わせる
    TRACKING_STUB_DELAY_SEC: float | None = 0.05

    ###### Depth Estimation & Box Fitting ######
    # --- ラベルごとの物体の想定サイズ ---
    # ラベルごとの (length, width, height) [m]。距離フィルタ（long_side）で
    # 長辺（length と width の大きい方）を奥行きの上限に使う。
    # run には box_fitting_params["box_sizes"] として残る。
    # TODO: 仮の値（nuScenes の各カテゴリの平均的な寸法の目安）。
    #       統計を取って差し替える
    DEFAULT_BOX_SIZES: dict[str, tuple[float, float, float]] = {
        "car": (4.63, 1.97, 1.74),
        "truck": (6.93, 2.51, 2.84),
        "bus": (11.10, 2.95, 3.47),
        "trailer": (12.29, 2.90, 3.87),
        "construction_vehicle": (6.37, 2.85, 3.19),
        "bicycle": (1.76, 0.60, 1.44),
        "motorcycle": (2.11, 0.77, 1.47),
        "pedestrian": (0.73, 0.66, 1.77),
        "traffic_cone": (0.41, 0.41, 1.07),
        "barrier": (2.50, 0.50, 0.98),
    }

    # ------ Generalタブ (Depth点群とLiDAR点群共通で適用するパラメータ) ------
    # Use LiDAR チェックボックスの既定値
    BOXFIT_USE_LIDAR_DEFAULT: bool = True

    # --- マスクのクロージング (General タブの Mask Closingセクション) ---
    MASK_DILATION_DEFAULT: int = 3
    MASK_DILATION_MAX: int = 31
    MASK_EROSION_DEFAULT: int = 5
    MASK_EROSION_MAX: int = 31
    # --- 小物体（遠方・小型の物体）用の調整 (General タブの Small Maskセクション) ---
    # マスクの外接矩形の短辺がこの値以下 [元画像の px] なら、収縮と ROR / DBSCAN を緩める。
    # 同じ条件で処理すると、収縮でマスクの大半が消えたり、点が疎で
    # ROR に丸ごと消されたりする。0 で無効
    SMALL_MASK_SHORT_SIDE_DEFAULT: int = 60
    SMALL_MASK_SHORT_SIDE_MAX: int = 300
    # 小物体の収縮: Erosion = Dilation + (Erosion - Dilation) * この倍率
    # （Erosion <= Dilation なら適用しない）
    SMALL_MASK_EROSION_RATIO_DEFAULT: float = 0.2

    # --- L-shape Filter (General タブの L-shape Filterセクション) ---
    # 車両の手前 2 面（L 字）を検出し、それより手前へ突き出た点を落とす。
    LSHAPE_FILTER_ENABLED_DEFAULT: bool = True
    # 適用するラベル（L 字の形がはっきりしている車両系）
    LSHAPE_FILTER_LABELS: list[str] = [
        "car", "truck", "bus", "trailer", "construction_vehicle",
    ]
    # 輪郭点が面に「乗っている」とみなす幅 [m]（General タブ / Filter Params）。
    # 1 つで 4 役を兼ねる重要なパラメータ:
    #   (1) 輪郭点の重み（輪郭から ±この幅の点の数）
    #   (2) 面に乗っているとみなす幅（L 字の当てはめの評価）
    #   (3) 面より手前に残っているとみなす許容幅（同上の減点）
    #   (4) 除去の条件（面から垂直方向にこれを超えて手前の点だけ消す。
    #       浅い角度で見えている面の点を守るため）
    LSHAPE_SUPPORT_BAND_DEFAULT: float = 0.3
    LSHAPE_SUPPORT_BAND_MAX: float = 1.0
    # L 字の向きを総当たりする刻み [度]
    LSHAPE_ANGLE_STEP_DEG: float = 1.0
    # 直交から各線の角度を個別に調整する幅（±）と刻み [度]。
    # 単眼深度では 2 面が直交からずれることがあるため。0 で調整しない
    LSHAPE_ADJUST_RANGE_DEG: float = 10.0
    LSHAPE_ADJUST_STEP_DEG: float = 1.0
    # 面が見えているか（I 字か L 字か）の判定に使う分位点 [%]
    LSHAPE_BOUNDARY_QUANTILE: float = 5.0
    # 面より手前に残る点 1 つあたりの減点（面に乗る点 1 つに対する比）。
    # 大きいと L 字が外側（センサー側）に寄り、小さいと点の多い内側に寄る。
    # 大きすぎると手前のノイズの塊に面が乗ってノイズを取り逃がし、
    # 小さすぎると短辺（前後面）がボンネット等の内側の点に引きずられてめり込み、手前の真の点を消す
    LSHAPE_VIOLATION_WEIGHT_DEFAULT: float = 5.0
    LSHAPE_VIOLATION_WEIGHT_MAX: float = 20.0

    # --- Distance Filter (General タブの Distance Filterセクション) ---
    # 物体の前後方向に大きく外れた点を落とす。XY（BEV）のみで判定。
    # L 字フィルタの直後に、深度とLiDAR に別々に適用する（深度は LiDAR と混ぜる前に落とす）
    # ラベルごとの方式（ここに無いラベル・None は何もしない）:
    #   long_side: 距離の分位点を基準距離 ref とし、
    #     ref − Margin 〜 ref + 長辺（DEFAULT_BOX_SIZES）+ Margin の点を残す
    #   dbscan: BEV で DBSCAN を行い、点数上位 2 クラスタのうち手前
    #     （距離の分位点で比較）の方だけを残す。壁のすぐ手前を走る自転車などで、
    #     背景の塊を落とす
    DISTANCE_FILTER_METHODS: dict[str, str | None] = {
        "bicycle": "dbscan",
        "motorcycle": "long_side",
        "pedestrian": "long_side",
        "traffic_cone": "long_side",
    }
    # 距離の分位点 [%]。long_side の基準距離と、dbscan の「手前」の比較に使う
    DISTANCE_FILTER_QUANTILE_DEFAULT: float = 5.0
    DISTANCE_FILTER_QUANTILE_MAX: float = 50.0
    # long_side の余裕 [m]。手前（ref − Margin）と奥の両側に取る
    DISTANCE_FILTER_MARGIN_DEFAULT: float = 0.3
    DISTANCE_FILTER_MARGIN_MAX: float = 3.0
    # dbscan の DBSCAN パラメータ（BEV。深度・LiDAR 共通）。
    # eps は物体と背景の隙間（例: 壁の 1 m 手前の自転車なら 1 m）より小さく、
    # 物体内の点の間隔より大きく。LiDAR は点が疎なので min_samples は小さめ
    DISTANCE_FILTER_DBSCAN_EPS_DEFAULT: float = 0.5
    DISTANCE_FILTER_DBSCAN_EPS_MAX: float = 3.0
    DISTANCE_FILTER_DBSCAN_MIN_SAMPLES_DEFAULT: int = 3
    DISTANCE_FILTER_DBSCAN_MIN_SAMPLES_MAX: int = 50

    # Radius Outlier Removal の nb_points にかけるラベルごとの倍率。
    # 小さい物体（traffic_cone / pedestrian）は点群自体が疎で、
    # 全ラベル共通の nb_points だと点が丸ごと消えてしまう。
    # 実際に使う値は「指定した nb_points × この倍率」を四捨五入したもの。
    # Depth 側・LiDAR 側の両方で共通に使う
    NB_POINTS_RATIO: dict[str, float] = {
        "car": 1.0,
        "truck": 1.0,
        "construction_vehicle": 1.0,
        "bus": 1.0,
        "trailer": 1.0,
        "barrier": 1.0,
        "traffic_cone": 0.5,
        "motorcycle": 1.0,
        "bicycle": 1.0,
        "pedestrian": 0.8,
    }

    # ------ Depth Estimationタブ (Depth点群処理のパラメータ) ------
    # --- 深度推定モデルの候補（HuggingFace のモデル ID）と既定値 (Depth EstimationタブのDepth Estimation Modelセクション) ---
    DEPTH_ESTIMATION_MODELS: list[str] = [
        "depth-anything/da3metric-large",
        "depth-anything/da3nested-giant-large",
        "facebook/map-anything-apache",
    ]
    DEFAULT_DEPTH_ESTIMATION_MODEL: str = "facebook/map-anything-apache"
    # カメラ姿勢を与えて複数フレームをまとめて推論するモデル
    POSE_CONDITIONED_DEPTH_MODELS: list[str] = [
        "depth-anything/da3nested-giant-large",
        "facebook/map-anything-apache",
    ]
    # Pose-conditioned 推論のウィンドウ（1 回にまとめるフレーム数と、ずらす幅）
    DEPTH_WINDOW_SIZE_DEFAULT: int = 9
    DEPTH_WINDOW_SIZE_MAX: int = 40
    DEPTH_WINDOW_STRIDE_DEFAULT: int = 5
    DEPTH_WINDOW_STRIDE_MAX: int = 40
    # 深度マップの保存倍率。1/2 にすると容量は 1/4（1 run 約 565MB → 141MB）
    DEPTH_MAP_DOWNSCALE: float = 0.5

    # --- 小物体の深度点群の ROR nb_points と DBSCAN min_samples にかける倍率 (Depth EstimationタブのSmall Maskセクション) ---
    # nb_points は NB_POINTS_RATIO（ラベル別）をかけた後にさらにかける
    SMALL_MASK_NB_POINTS_RATIO_DEFAULT: float = 0.6

    # --- 深度点群のROR/DBSCAN フィルタ (Depth EstimationタブのROR/DBSCANセクション) ---
    DEPTH_ROR_NB_POINTS_DEFAULT: int = 5
    DEPTH_ROR_NB_POINTS_MAX: int = 50
    DEPTH_ROR_RADIUS_DEFAULT: float = 0.8
    DEPTH_ROR_RADIUS_MAX: float = 5.0
    DEPTH_DBSCAN_EPS_DEFAULT: float = 1.0
    DEPTH_DBSCAN_EPS_MAX: float = 5.0
    DEPTH_DBSCAN_MIN_SAMPLES_DEFAULT: int = 10
    DEPTH_DBSCAN_MIN_SAMPLES_MAX: int = 100

    # --- 深度点群のL-shape Filter (Depth EstimationタブのL-shape Filterセクション) ---
    # 手前輪郭を作る方位ビンの幅 [度] と、各ビンで取る距離の分位点 [%]
    DEPTH_LSHAPE_BIN_DEG_DEFAULT: float = 0.5
    DEPTH_LSHAPE_QUANTILE_DEFAULT: float = 10.0
    # 除去の閾値 max(Margin Min, Margin Ratio × 距離) [m]。
    # 単眼深度の誤差は距離とともに大きくなるので、距離に比例させる
    DEPTH_LSHAPE_MARGIN_MIN_DEFAULT: float = 0.3
    DEPTH_LSHAPE_MARGIN_RATIO_DEFAULT: float = 0.03

    # ------ LiDAR Pointcloudタブ (LiDAR点群処理のパラメータ) ------
    # LiDAR を基準に深度点群を補正する方式。
    # インスタンス単位では深度の範囲が狭く（車 1 台で 4〜5 m）、
    # スケールとオフセットがほぼ区別できないため 1 パラメータで足りる。
    # 実測（LiDAR 3 点）: scale 0.16 m / shift 0.15〜0.22 m / affine 0.22〜0.25 m
    DEPTH_CORRECTION_METHODS: list[str] = ["scale", "shift", "affine"]
    DEFAULT_DEPTH_CORRECTION_METHOD: str = "scale"

    # LiDAR 点群のフィルタ
    # キーフレームあたりの LiDAR sweep 数（LiDAR 20Hz÷キーフレーム2Hz）。
    # LiDAR Sweeps パラメータの最大値に使う
    LIDAR_SWEEPS_PER_SAMPLE: int = 10
    LIDAR_NUM_SWEEPS_DEFAULT: int = 5
    LIDAR_MIN_POINTS_DEFAULT: int = 6
    LIDAR_MIN_POINTS_MAX: int = 200
    # 自車の車体が占める範囲（ego 座標、メートル）。
    # LiDAR は自車の屋根やボンネットの反射も返す。これがインスタンスマスクに
    # 重なると点群へ混入し、**非常に密なので DBSCAN の最大クラスタとして
    # 勝ってしまう**（後段のフィルタでは救えない）。
    #
    # 半径ではなく直方体で切る。半径だと自車のすぐ横の歩行者や
    # 前方 1.5 m の車も消えてしまう。
    # nuScenes の ego は 4.084 x 1.730 x 1.562 m。少し余裕を持たせている
    EGO_BOX_X_RANGE: tuple[float, float] = (-1.5, 4.2)
    EGO_BOX_Y_RANGE: tuple[float, float] = (-1.3, 1.3)
    EGO_BOX_Z_RANGE: tuple[float, float] = (-0.5, 2.3)

    # インスタンスの LiDAR 点がこの数以上なら、**深度点群を使わない**。
    # LiDAR だけで形が十分に取れる場合、深度推定の外れ値が混ざるほうが害になる。
    # 0 なら無効（常に深度点群も使う）
    LIDAR_MAX_POINTS_FOR_DEPTH_DEFAULT: int = 60
    LIDAR_MAX_POINTS_FOR_DEPTH_MAX: int = 500
    # 深度補正（最終の補正）で、点ごとの対応（LiDAR 点と、その周囲の残った
    # 深度点の予測深度の中央値）から係数を求めるのに必要な対応点の数。
    # 届かなければ「残った LiDAR 点の深度の中央値 ÷ 残った深度点の予測深度の
    # 中央値」で求める（遠方の小さな物体では画素の対応がほとんど取れないため）
    LIDAR_MIN_CORRESPONDENCES_DEFAULT: int = 3
    LIDAR_MIN_CORRESPONDENCES_MAX: int = 50
    # 上の対応を取る近傍の半径 [深度マップの画素]（UI には出さない）。
    # 深度マップは元画像の DEPTH_MAP_DOWNSCALE 倍なので、2 で元画像の約 ±4 画素
    DEPTH_CORRECTION_NEIGHBOR_RADIUS_PX: int = 2

    # --- LiDAR 点群の ROR/DBSCAN フィルタ (LiDAR PointcloudタブのROR/DBSCANセクション) ---
    LIDAR_ROR_NB_POINTS_DEFAULT: int = 3
    LIDAR_ROR_NB_POINTS_MAX: int = 50
    LIDAR_ROR_RADIUS_DEFAULT: float = 0.8
    LIDAR_ROR_RADIUS_MAX: float = 5.0
    LIDAR_DBSCAN_EPS_DEFAULT: float = 1.2
    LIDAR_DBSCAN_EPS_MAX: float = 5.0
    LIDAR_DBSCAN_MIN_SAMPLES_DEFAULT: int = 5
    LIDAR_DBSCAN_MIN_SAMPLES_MAX: int = 100

    # --- LiDAR 点群の L-shape Filter (LiDAR PointcloudタブのL-shape Filterセクション) ---
    # LiDAR は点が疎なのでビンを深度より広くする。距離の誤差は小さいので閾値は固定値 [m]
    LIDAR_LSHAPE_BIN_DEG_DEFAULT: float = 1.0
    LIDAR_LSHAPE_QUANTILE_DEFAULT: float = 10.0
    LIDAR_LSHAPE_MARGIN_DEFAULT: float = 0.2

    # ------ Inter-cam Mergeタブ (カメラを跨いだ同一インスタンス結合) ------
    # 点群を揃える基準の座標系を決めるセンサー。
    # カメラごとに sample_data のタイムスタンプが違うため、
    # 「ego 座標」の基準がカメラ間でずれている。カメラを跨いで点群を
    # 比べる前に、このセンサーの ego_pose へ全部揃える。
    # LiDAR を使わない構成へ移す場合は、常に存在するカメラ
    # （CAM_FRONT など）を指定すれば同じ仕組みで動く
    EGO_REFERENCE_CHANNEL: str = "LIDAR_TOP"

    # --- 凸包の重なり率に基づく結合の設定 (Inter-cam MergeタブのHull Based Mergeセクション) ---
    # カメラ間の同一インスタンス結合の判定方法。
    # 手法を増やすときは MERGE_METHODS に足す（UI の Selectbox がこれを使う）
    MERGE_METHODS: list[str] = ["BEV convex-hull"]
    DEFAULT_MERGE_METHOD: str = "BEV convex-hull"
    # 結合してよいラベルの条件（label / category_group / none）。判定にはトラック内で多数決したラベルを使う
    DEFAULT_MERGE_LABEL_MATCH: str = "category_group"
    # BEV 凸包の重なり率の下限。
    # IoS（小さい方の面積で割る）なので、IoU より高めに取れる。
    # カメラごとに物体の違う面しか観測できないため、同一物体でも
    # IoU は 0.3 程度まで落ちる（IoS なら 0.5 前後）
    DEFAULT_MERGE_OVERLAP_THRESHOLD: float = 0.15
    MERGE_OVERLAP_THRESHOLD_MAX: float = 1.0
    # BEV 凸包どうしの隙間の上限 [m]。これを超える組は重なりを計算しない
    # （足切り）。重なっていれば隙間は 0 なので、長い車両がカメラの境界で
    # 分かれても落ちない（重心距離だと車長の半分ほど離れて落ちていた）。
    # 重なり率は凸包が接していないと 0 なので、結合の結果には影響しない。
    # 効くのは計算量と、「重なりは無いが近い組」を診断に残す範囲
    DEFAULT_MERGE_MAX_HULL_GAP: float = 0.5
    MERGE_MAX_HULL_GAP_MAX: float = 5.0
    # 何フレームでマッチしたら結合するか。
    # 同じ物体が複数カメラに写るキーフレームは高々 1〜2 なので、
    # 割合ではなくフレーム数で判定する（割合だと 0/0.5/1 の 3 値しか取れない）
    DEFAULT_MERGE_MIN_MATCH_FRAMES: int = 1
    MERGE_MIN_MATCH_FRAMES_MAX: int = 5
    # 同一カメラの 2 トラックを同じグループへ入れてよい時間的な隔たり
    # （キーフレーム数）。カメラ A → カメラ B → カメラ A と写り込む物体は、
    # カメラ A 側で 2 つのトラックに分断される。同時に存在しないなら
    # 同じ物体でありうるので、この範囲内なら同居を許す。
    # 0 にすると同一カメラの同居を一切許さない（従来の挙動）
    DEFAULT_MERGE_MAX_SAME_CAMERA_GAP: int = 20
    MERGE_MAX_SAME_CAMERA_GAP_MAX: int = 40

    # --- ボックスの規定長と向きに基づく結合 (Inter-cam MergeタブのBox-Orient Based Mergeセクション) ---
    # 凸包の重なりで結合した後（hull_merged_track_id）、角度補正を済ませたボックス
    # どうしを「同じラベル・向きの差が閾値以下・合体しても規定長（DEFAULT_BOX_SIZES）
    # × 倍率に収まる」条件で結合し、global_track_id を決める。
    # 対象ラベル（車両系だけ。歩行者は 2 人が極端に近いことが多く、誤結合が増える）
    BOX_ORIENT_MERGE_LABELS: list[str] = [
        "car", "truck", "bus", "trailer", "construction_vehicle",
    ]
    DEFAULT_BOX_MERGE_ORIENTATION_DIFF: float = 20.0
    BOX_MERGE_ORIENTATION_DIFF_MAX: float = 45.0
    # ラベルごとの合体後の長さの上限（DEFAULT_BOX_SIZES の length に対する倍率）。
    # 長さがまちまちな車種（truck 等）は大きめ、差が小さい car は小さめにする。
    # 実際の上限は これ × UI の Length Ratio Threshold Ratio。ここに無いラベルは 1.0
    LENGTH_RATIO_THRESHOLD: dict[str, float] = {
        "car": 1.3,
        "truck": 1.6,
        "bus": 1.4,
        "trailer": 1.5,
        "construction_vehicle": 1.5,
    }
    DEFAULT_BOX_MERGE_LENGTH_RATIO_SCALE: float = 1.0
    BOX_MERGE_LENGTH_RATIO_SCALE_MIN: float = 0.5
    BOX_MERGE_LENGTH_RATIO_SCALE_MAX: float = 2.0
    DEFAULT_BOX_MERGE_WIDTH_RATIO: float = 1.3
    BOX_MERGE_RATIO_MAX: float = 2.0
    DEFAULT_BOX_MERGE_MAX_HULL_GAP: float = 0.8
    DEFAULT_BOX_MERGE_MIN_MATCH_FRAMES: int = 1

    # ------ Box Fittingタブ (Box Fitting) ------
    # ラベルごとの当てはめ手法。ここに無いラベルは convex_hull_moa。
    #   convex_hull_moa: BEV の凸包に対し、角度を刻んで
    #     センサーから見たオクルージョン面積が最小になる向きを探す
    # 今は全ラベル convex_hull_moa。向き推定（PartHOE / Orient Anything など）を
    # 足すときに、ラベルごとに手法を切り替える差し込み口として辞書にしている
    BOXFIT_METHODS: dict[str, str] = {
        "car": "convex_hull_moa",
        "truck": "convex_hull_moa",
        "bus": "convex_hull_moa",
        "trailer": "convex_hull_moa",
        "construction_vehicle": "convex_hull_moa",
        "bicycle": "convex_hull_moa",
        "motorcycle": "convex_hull_moa",
        "pedestrian": "convex_hull_moa",
        "traffic_cone": "convex_hull_moa",
        "barrier": "convex_hull_moa",
    }

    # 当てはめに必要な点数の下限。これに届かないインスタンスはボックスを
    # 作らない（点群は残す）。点が少なすぎると BEV の凸包が線や点に潰れ、
    # 向きも大きさも意味を持たない。
    # **判定はカメラ間結合の後、グローバルトラック単位・フレームごとに
    # 行う**（実際に当てはめへ渡す点群で数えるため）
    BOXFIT_MIN_POINTS_DEFAULT: int = 10
    BOXFIT_MIN_POINTS_MAX: int = 100

    # --- convex_hull_moaのパラメータ (Box Fittingタブのconvex_hull_moaセクション) ---
    # 向きの探索刻み [度]。細かくしても結果はほぼ変わらず、計算時間だけ伸びる
    BOXFIT_ANGLE_STEP_DEG_DEFAULT: float = 0.5
    BOXFIT_ANGLE_STEP_DEG_MIN: float = 0.1
    BOXFIT_ANGLE_STEP_DEG_MAX: float = 10.0

    # --- 底面の推定 (Box FittingタブのBottom Estimationセクション) ---
    # ラベルごとの方式。ここに無いラベルは percentile。
    #   percentile: 点群の z の分位点（BOXFIT_Z_PERCENTILE_LOW）
    #   ground    : Patchwork++ の地面点のうち、箱の外形から XY で
    #               Ground Estimation Radius 以内の点から求める（Use LiDAR のときだけ）。
    #               地面点が Min Ground Points 以下なら percentile に切り替える
    #   ego       : 基準 ego 座標の z = 0（自車の接地面にそろえる）
    # LiDAR の上下方向のビームは疎なので、分位点だと底面が高く（箱が低く）なりやすい
    BOX_BOTTOM_ESTIMATIONS: dict[str, str] = {
        "car": "ground",
        "truck": "ground",
        "bus": "ground",
        "trailer": "ground",
        "construction_vehicle": "ground",
        "bicycle": "ground",
        "motorcycle": "ground",
        "pedestrian": "ground",
        "traffic_cone": "ground",
        "barrier": "ground",
    }
    # ground 方式で使う地面点の、箱の外形からの距離の上限 [m]
    BOX_BOTTOM_GROUND_RADIUS_DEFAULT: float = 2.0
    BOX_BOTTOM_GROUND_RADIUS_MAX: float = 10.0
    # ground 方式に必要な地面点の数。これ以下なら percentile に切り替える
    BOX_BOTTOM_MIN_GROUND_POINTS_DEFAULT: int = 10
    BOX_BOTTOM_MIN_GROUND_POINTS_MAX: int = 200
    # 地面点から底面の高さを求める方法
    #   percentile    : 地面点の z の分位点（BOX_BOTTOM_GROUND_PERCENTILE）
    #   least_squares : 地面点に平面を最小二乗で当てはめ、箱の XY 中心での高さ
    BOX_BOTTOM_GROUND_HEIGHT_METHODS: list[str] = ["percentile", "least_squares"]
    BOX_BOTTOM_GROUND_HEIGHT_METHOD_DEFAULT: str = "percentile"
    # percentile 方式の分位点 [%]。中央値（50）より低くするのは、歩道や縁石の
    # 点が混ざると路面より高く出るため
    BOX_BOTTOM_GROUND_PERCENTILE_DEFAULT: float = 30.0

    # --- ボックス上面の推定 (Box FittingタブのTop Estimationセクション) ---
    # 高さを決めるパーセンタイル。最小・最大をそのまま使うと
    # 路面やマスクのはみ出し 1 点で箱が縦に伸びる。
    #   LOW : 底面（Bottom Estimation の percentile 方式・ground 方式の切り替え先）
    #   HIGH: 上面（Top Estimation）
    BOXFIT_Z_PERCENTILE_LOW_DEFAULT: float = 1.0
    BOXFIT_Z_PERCENTILE_HIGH_DEFAULT: float = 99.0

    # --- AI モデルによるボックス向きの再推定方式 (Box FittingタブのOrientation Estimationセクション) ---
    # ラベルごとに指定し、None は推定しない（当てはめた向きのまま）。
    # Box Fitting のパス 2（底面推定の後・寸法補正の前）で推定し、ボックスへ反映する。
    #   part_hoe: PartHOE（人物の向き推定）。XY 中心を固定して yaw だけを推定値にする
    #   orient_anything: Orient-Anything V1 Large（物体の向き推定。背景はトラッキングのマスクで落とす）。
    #     XY 中心を固定して yaw だけを推定値にする
    #   orient_anything_combined: 推定は orient_anything と同じ。長辺が短辺の標準値
    #     （DEFAULT_BOX_SIZES の width）× ORIENT_ANYTHING_SWITCH_LENGTH_RATIO 以上なら、
    #     元のボックスの長辺の前後のうち推定に近い方を X 軸にする（8 頂点は動かさない）。
    #     未満なら orient_anything と同じ
    # 推論サーバーへ box_fitting_params["orientation"] で渡す（ラベル体系は webapp 側の設定のため）
    BOX_ORIENTATION_ESTIMATIONS: dict[str, str | None] = {
        "car": "orient_anything_combined",
        "truck": "orient_anything_combined",
        "bus": "orient_anything_combined",
        "trailer": "orient_anything_combined",
        "construction_vehicle": "orient_anything_combined",
        "bicycle": "orient_anything_combined",
        "motorcycle": "orient_anything_combined",
        "pedestrian": "part_hoe",
        "traffic_cone": None,
        "barrier": None,
    }
    # orient_anything_combined で、長辺が短辺の標準値のこの倍率以上なら
    # 長辺が写っているとみなし、元のボックスの前後を選ぶだけにする
    # （長辺の面の点群があれば、当てはめた向きの方が AI の推定より正確なため）
    ORIENT_ANYTHING_SWITCH_LENGTH_RATIO: float = 1.4
    # orient_anything_combined で長辺の前後を選んだとき、推定と選んだ軸のなす角 [度] が
    # これ以上なら「角度推定直角ずれ」の印を付ける（後段の複数フレームを考慮した補正の対象）
    ORIENT_INACCURATE_THRESHOLD: float = 60.0

    # --- グローバルトラックの全フレームを考慮した角度補正 (Box FittingタブのOrientation Correctionセクション) ---
    #   dbscan_interp: グローバル座標の向きを単位円上で DBSCAN し、最大クラスタ以外の
    #     フレームを、前後の最大クラスタのフレームから時刻で線形円周補間した向きに直す
    #   None / 辞書に無い: 補正しない
    # BOX_ORIENTATION_ESTIMATIONS が None のラベルは前後が決まらないので補正しない
    BOX_ORIENTATION_CORRECTIONS: dict[str, str | None] = {
        "car": "dbscan_interp", "truck": "dbscan_interp", "bus": "dbscan_interp",
        "trailer": "dbscan_interp", "construction_vehicle": "dbscan_interp",
        "bicycle": "dbscan_interp", "motorcycle": "dbscan_interp",
        "pedestrian": "dbscan_interp", "traffic_cone": None, "barrier": None,
    }
    # DBSCAN の eps（角度の閾値 [度]）。単位円上の距離 2 * sin(この値 / 2) にして使う
    BOX_ORIENTATION_DBSCAN_EPS_DEGREE: float = 20.0
    BOX_ORIENTATION_DBSCAN_MIN_SAMPLES: int = 3
    # これ未満のフレーム数のトラックは角度補正しない
    BOX_ORIENTATION_CORRECTION_MIN_FRAMES: int = 4    

    # --- グローバルトラック単位の寸法補正 (Box Fitting タブの Height / Width / Length Estimationセクション) ---
    # 各フレームの箱は隠れや疎なビームで小さめに出るので、グローバルトラックの
    # 全フレームから共通の寸法を決めて当て直す。方式はラベルごと:
    #   percentile: 全フレームのその寸法の分位点（フレームが
    #               BOX_SIZE_CORRECTION_MIN_FRAMES に届かなければ補正しない）
    #   default   : DEFAULT_BOX_SIZES の値
    #   None / 辞書に無い: 補正しない
    # 高さは底面を固定して上面を、幅・長さはセンサーから見えている面を固定して
    # 反対側の面を動かす（見えていない軸は中心を固定して対称に伸ばす）
    BOX_HEIGHT_CORRECTIONS: dict[str, str | None] = {
        "car": "percentile", "truck": "percentile", "bus": "percentile",
        "trailer": "percentile", "construction_vehicle": "percentile",
        "bicycle": "percentile", "motorcycle": "percentile",
        "pedestrian": "percentile", "traffic_cone": "percentile",
        "barrier": "percentile",
    }
    # 幅は車種でそれほど変わらないので、車両系は default に寄せる
    BOX_WIDTH_CORRECTIONS: dict[str, str | None] = {
        "car": "default", "truck": "default", "bus": "default",
        "trailer": "default", "construction_vehicle": "default",
        "bicycle": "percentile", "motorcycle": "percentile",
        "pedestrian": "percentile", "traffic_cone": "percentile",
        "barrier": "percentile",
    }
    # 長さは車種によってまちまちなので percentile に寄せる
    BOX_LENGTH_CORRECTIONS: dict[str, str | None] = {
        "car": "percentile", "truck": "percentile", "bus": "percentile",
        "trailer": "percentile", "construction_vehicle": "percentile",
        "bicycle": "percentile", "motorcycle": "percentile",
        "pedestrian": "percentile", "traffic_cone": "percentile",
        "barrier": "percentile",
    }
    # percentile 方式の分位点 [%]。各フレームの寸法は小さめに出るので高めにする。
    # 幅はノイズで大きくなりやすいので、やや低めにする
    BOX_HEIGHT_CORRECTION_PERCENTILE_DEFAULT: float = 80.0
    BOX_WIDTH_CORRECTION_PERCENTILE_DEFAULT: float = 70.0
    BOX_LENGTH_CORRECTION_PERCENTILE_DEFAULT: float = 80.0
    # percentile 方式に必要なフレーム数（これ未満のトラックは補正しない）
    BOX_SIZE_CORRECTION_MIN_FRAMES_DEFAULT: int = 3
    BOX_SIZE_CORRECTION_MIN_FRAMES_MAX: int = 20

    # ------ Track Smoother タブ（静止 / 移動で場合分けしたKalman Smoother による補正） ------
    # box_merged_track ごとに、グローバル座標の XY でカルマンフィルタ＋RTS スムーザー
    # による平滑化を行い、静止 / 移動を判定して補正する。分割されたトラックを結合した
    # ものが global_track_id になる（詳細は inference/CLAUDE.md）
    TRACK_SMOOTHER_ENABLED_DEFAULT: bool = True
    # 平滑化・静止判定の対象にするトラックの最小フレーム数（未満は結合の候補にだけする）
    TRACK_SMOOTHER_MIN_FRAMES_DEFAULT: int = 3
    TRACK_SMOOTHER_MIN_FRAMES_MAX: int = 10

    # --- Kalman Smootherによる平滑化 (Track SmootherタブのKalman Smootherセクション) ---
    # 平滑化で外れ値とするマハラノビス距離の二乗（χ²(2, 0.99)）と、除去の反復回数の上限
    TRACK_SMOOTHER_OUTLIER_THRESHOLD_DEFAULT: float = 9.21
    TRACK_SMOOTHER_NUM_ITERS_DEFAULT: int = 3
    TRACK_SMOOTHER_NUM_ITERS_MAX: int = 10
    # 初期状態の速度の分散 [(m/s)²]（観測が 1 つのときも速度は分からないので大きく取る）
    TRACK_SMOOTHER_INITIAL_VELOCITY_VAR: float = 100.0
    # 観測ノイズ σ = a + b·d [m]（d はセンサーからの距離）。ボックスの長さ方向と
    # 幅方向で別の値を持ち、当てはめた点群に LiDAR 点を含むか（lidar）/ 深度のみ（depth）
    # で切り替える。lidar は tools/estimate_track_noise.py で nuScenes 6 シーン
    # （約 3100 ボックス）の GT との残差から推定した値。depth は深度のみのボックスが
    # 無く推定できなかったので仮の値（LiDAR の 2〜3 倍）
    TRACK_SMOOTHER_MEASUREMENT_NOISE: dict[str, dict[str, list[float]]] = {
        "lidar": {"length": [0.036, 0.0164], "width": [0.01, 0.0097]},
        "depth": {"length": [0.8, 0.03], "width": [0.4, 0.015]},
    }
    # 等速度モデルの加速度の強さ q [m²/s³]。辞書に無いラベルは default。
    # tools/estimate_track_noise.py で GT（850 シーン、移動中）から推定した中央値は
    # car / truck / bus 0.17〜0.18、pedestrian 0.007 だが、中央値は曲がる・加減速する
    # 区間を拾わず小さすぎる（6 シーンで試すと速度の向きによる角度の修正が悪化し、
    # 正しい結合も減った）。そのため経験的に決めた値を使う
    TRACK_SMOOTHER_PROCESS_NOISE: dict[str, float] = {
        "default": 1.0, "pedestrian": 0.5, "traffic_cone": 0.1, "barrier": 0.1,
    }
    # --- 静止判定 (Track SmootherタブのStatic Detectionセクション) ---
    # 中心の幅方向（ボックスの向きに垂直）のばらつき（この分位点）が
    # Static Radius 未満、正味の移動量（最初と最後の 3 分の 1 の中央値の位置の差）が
    # Static Max Displacement 未満、平滑化した速さの中央値が Static Max Speed 未満。
    # 止まっている車両でも、見えている面が変わると中心が長さ方向に数 m 跳ぶので、
    # 長さ方向のばらつきは見ない（nuScenes 6 シーンの GT 属性で決めた値）
    TRACK_SMOOTHER_STATIC_RADIUS_DEFAULT: float = 1.0
    TRACK_SMOOTHER_STATIC_SPREAD_PERCENTILE: float = 90.0
    TRACK_SMOOTHER_STATIC_MAX_DISPLACEMENT_DEFAULT: float = 2.0
    TRACK_SMOOTHER_STATIC_MAX_SPEED_DEFAULT: float = 1.0
    
    # --- 静止判定されたトラックの補正 (Track SmootherタブのStable Track Correctionセクション) ---
    TRACK_SMOOTHER_STATIC_LINK_DISTANCE_DEFAULT: float = 1.0
    TRACK_SMOOTHER_STATIC_BOTTOM_PERCENTILE_DEFAULT: float = 25.0
    TRACK_SMOOTHER_TRACK_ORIENT_DIFF_DEFAULT: float = 15.0
    TRACK_SMOOTHER_NEIGHBOR_ORIENT_DIFF_DEFAULT: float = 15.0
    # 周囲の静止車両のボックスを集める範囲 [m]（対象ボックス中心から、その向きの
    # 前後方向 / 左右方向）
    STABLE_TRACK_ORIENT_SEARCH_LENGTH: float = 15.0
    STABLE_TRACK_ORIENT_SEARCH_WIDTH: float = 3.0

    # --- 移動トラックの結合 (Track SmootherタブのMoving Track Mergeセクション) ---
    # 時間差の上限 [s] と、D² の閾値（位置と速度の 4 次元 /
    # 片方が短いトラックのときの位置だけの 2 次元。いずれも χ² の 95% 点。
    # 99% 点では誤った結合が 3 割を超えた）
    TRACK_SMOOTHER_MAX_LINK_GAP_DEFAULT: float = 3.0
    TRACK_SMOOTHER_LINK_THRESHOLD_4D: float = 9.49
    TRACK_SMOOTHER_LINK_THRESHOLD_2D: float = 5.99

    # --- 移動トラックの位置の修正 (Track SmootherタブのMoving Track Correctionセクション) ---
    # 移動トラックの位置の修正を行うか。nuScenes 6 シーンでは、修正したフレームの GT との
    # 誤差が平均 1.3 m → 6 m に悪化した（外れ値とされた観測の多くは正しく、トラック内の
    # 別物体の混入や、長さ方向の中心の跳びで平滑化の方がずれていた）ので既定はオフ
    TRACK_SMOOTHER_CORRECT_MOVING_POSITION_DEFAULT: bool = False
    # 移動トラックの修正: 位置はマハラノビス距離の二乗がこれを超え、かつずれが
    # Min Position Correction を超えたら平滑化した位置へ移す
    TRACK_SMOOTHER_POSITION_GATE: float = 9.21
    TRACK_SMOOTHER_MIN_POSITION_CORRECTION_DEFAULT: float = 0.5
    TRACK_SMOOTHER_MIN_SPEED_HEADING_DEFAULT: float = 2.0
    TRACK_SMOOTHER_HEADING_DIFF_DEFAULT: float = 20.0

    # ------ 表示関係 ------
    # 点群ビューの Global View の視点。
    # global 座標に対する既定の視線位置と上方向で、
    # ego_pose の回転を打ち消して適用される（車両が回っても向きが変わらない）
    DEFAULT_GLOBAL_POINTCLOUD_EYE: list[float] = [-1.0, -0.5, 1.5]
    DEFAULT_GLOBAL_POINTCLOUD_UP: list[float] = [0.0, 0.0, 1.0]

    # 点群ビューに描く自車姿勢の軸の長さ [m]（x=赤 / y=緑 / z=青）
    POINTCLOUD_AXIS_LENGTH_M: float = 5.0

    # 点群ビューの既定表示
    SHOW_RAW_LIDAR: bool = False
    SHOW_RAW_LIDAR_GROUND: bool = False
    SHOW_RAW_DEPTH_POINTCLOUD: bool = False

    # DB に保存するインスタンス点群の上限（ボクセル間引き後）
    BOXFIT_STORED_POINTS_MAX: int = 500
    # Plotly へ渡す点数の上限。超えると転送量と描画が重くなる
    POINTCLOUD_DISPLAY_MAX_POINTS: int = 50_000

    # ------ 記録・管理関係 ------
    # 保持する run 数。派生ファイルが 1 run 約 280MB あるため少なめにする
    DEPTH_MAX_RUNS_PER_SCENE: int = 3
    BOXFIT_STUB_DELAY_SEC: float | None = 0.02

    @property
    def database_url(self) -> str:
        """SQLAlchemy 用の同期 DSN.

        Streamlit は同期実行モデルであり、DB も単一ファイルの SQLite なので
        AsyncSession は利点がなく、複雑さだけが増すため同期ドライバを使う。
        """
        return f"sqlite+pysqlite:///{self.SQLITE_PATH}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """設定のシングルトン。プロセス内で1度だけ読み込む。"""
    return Settings()
