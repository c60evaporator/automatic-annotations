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

    LOG_LEVEL: str = "INFO"

    # --- データ配置 -------------------------------------------------------
    # コンテナ内でのデータセットルート（ホスト側 HOST_DATA_ROOT のマウント先）
    DATA_ROOT: Path = Path("/data")
    # 推論の派生成果物（深度マップ .npz 等）の保存先
    DERIVED_ROOT: Path = Path("/data/_derived")

    # --- データベース -----------------------------------------------------
    SQLITE_PATH: Path = Path("/db/app.db")
    SQL_ECHO:    bool = False
    # 書き込みロック待ちの上限（ミリ秒）。推論結果の一括書き込みと
    # UI の読み取りが競合したときの "database is locked" を防ぐ
    SQLITE_BUSY_TIMEOUT_MS: int = 10_000

    # --- 推論サーバー -----------------------------------------------------
    # タイムアウトは用途ごとに差が大きいので、ここではなく
    # services/inference_client.py の定数で持つ
    # （投入・ポーリング・再フィルタでそれぞれ別の値）
    INFERENCE_BASE_URL: str = "http://inference:8000"

    # --- ラベル変換 -----------------------------------------------------
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

    LABEL_TO_CATEGORY_GROUP: dict[str, str] = {  # 検出ラベル -> カテゴリグループ
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

    # --- 表示 -------------------------------------------------------------
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

    # SigLIP2 によるラベル再判定。
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
                "truck cab": "truck",
                "window reflection": None,
                "painting": None},
        "construction_vehicle": {"construction vehicle": "construction_vehicle",
                                 "truck": "truck"},
        "bus": {"bus": "bus",
                "truck": "truck",
                "building": None},
        "motorcycle": {"motorcycle": "motorcycle",
                       "human legs": None,
                       "mural": None},
        "bicycle": {"bicycle": "bicycle",
                    "human legs": None},
        "pedestrian": {"pedestrian": "pedestrian",
                       "mural": None,
                       "rider": None},
    }
    # --- 検出 0 件のフレームの救済 ------------------------------------------
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

    # --- 重複検出の抑制（IoS）----------------------------------------------
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
    # ここに無いラベルは判定しない
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

    # 切り出し時にボックスを広げる比率。
    # 文脈が写らないと zero-shot 分類が当たらないため、少し広めに取る
    DEFAULT_RECLASSIFICATION_CROP_MARGIN_RATIO: float = 0.1
    RECLASSIFICATION_CROP_MARGIN_RATIO_MAX: float = 1.0

    # --- 2D Object Detection ---------------------------------------------
    # 4 だとインターバルの間に現れて消えるインスタンスを取りこぼす
    # （自車や対象の速度が速い場面）
    DET2D_DEFAULT_SAMPLE_INTERVAL: int = 3
    DET2D_DEFAULT_SCORE_THRESHOLDS: dict[str, float] = {
        "vehicle": 0.35,
        "road_object": 0.25,
        "two_wheeler": 0.3,
        "pedestrian": 0.3,
    }
    DET2D_NMS_SAME_CLASS_IOUS: dict[str, float] = {
        "vehicle": 0.7,
        "road_object": 0.6,
        "two_wheeler": 0.6,
        "pedestrian": 0.6,
    }
    DET2D_NMS_CROSS_CLASS_IOU: float = 0.85

    # 再実行時、この IoU 以上で重なる手修正ボックスがあれば、
    # 推論ボックスを手修正ボックスで置き換える（推論サーバーへは送らない）
    DET2D_MANUAL_REPLACE_IOU: float = 0.5

    # 1シーンあたり保持する run の上限。超えたら古いものから削除する。
    # ただし Instance Tracking から参照されている run は削除しない
    # （消すとトラッキング結果と 3D ボックスまで CASCADE で消えるため）
    DET2D_MAX_RUNS_PER_SCENE: int = 10

    # run の記録に残すモデル名（推論サーバー側の実体と合わせる）
    DET2D_MODEL_NAME: str = "groundingdino_swinb_cogcoor"

    # --- Instance Tracking -----------------------------------------------
    SWEEPS_PER_SAMPLE: int = 6
    DEFAULT_TRACKING_NUM_SWEEPS: int = 2
    DEFAULT_TRACKING_IOU_THRESHOLD: float = 0.4
    DEFAULT_TRACKING_IOU_METHOD: str = "box"
    DEFAULT_TRACKING_IOU_LABEL_MATCH: str = "category_group"
    DEFAULT_TRACKING_MASK_SCORE_THRESHOLD: float = 0.5
    # track_id の引き継ぎ方式。
    #   continuous_id            … Forward のみ。区間境界で次のプロンプトと照合
    #   forward_backward_matching … Forward と Backward を走らせ、区間内で照合。
    #                               区間途中に現れたインスタンスを拾える
    DEFAULT_TRACK_ID_INHERITANCE: str = "forward_backward_matching"
    TRACKING_MAX_RUNS_PER_SCENE: int = 10
    TRACKING_MODEL_NAME: str = "sam2.1_hiera_large"
    TRACKING_STUB_DELAY_SEC: float | None = 0.05

    # --- Depth Estimation & Box Fitting ------------------------------------
    DEPTH_MODEL_NAME: str = "depth-anything-3-large"
    # 深度マップの保存倍率。1/2 にすると容量は 1/4（1 run 約 565MB → 141MB）
    DEPTH_MAP_DOWNSCALE: float = 0.5
    # 保持する run 数。派生ファイルが 1 run 約 280MB あるため少なめにする
    DEPTH_MAX_RUNS_PER_SCENE: int = 3
    # DB に保存するインスタンス点群の上限（ボクセル間引き後）
    BOXFIT_STORED_POINTS_MAX: int = 500
    # Plotly へ渡す点数の上限。超えると転送量と描画が重くなる
    POINTCLOUD_DISPLAY_MAX_POINTS: int = 50_000

    # マスクのクロージング（General タブ）
    MASK_DILATION_DEFAULT: int = 2
    MASK_DILATION_MAX: int = 31
    MASK_EROSION_DEFAULT: int = 4
    MASK_EROSION_MAX: int = 31

    # 深度点群のフィルタ（Depth Estimation タブ）
    DEPTH_ROR_NB_POINTS_DEFAULT: int = 5
    DEPTH_ROR_NB_POINTS_MAX: int = 50
    DEPTH_ROR_RADIUS_DEFAULT: float = 0.8
    DEPTH_ROR_RADIUS_MAX: float = 5.0
    DEPTH_DBSCAN_EPS_DEFAULT: float = 1.0
    DEPTH_DBSCAN_EPS_MAX: float = 5.0
    DEPTH_DBSCAN_MIN_SAMPLES_DEFAULT: int = 10
    DEPTH_DBSCAN_MIN_SAMPLES_MAX: int = 100

    # LiDAR 点群のフィルタ（LiDAR Pointcloud タブ）
    LIDAR_NUM_SWEEPS_DEFAULT: int = 5
    LIDAR_MIN_POINTS_DEFAULT: int = 6
    # インスタンスの LiDAR 点がこの数以上なら、**深度点群を使わない**。
    # LiDAR だけで形が十分に取れる場合、深度推定の外れ値が混ざるほうが害になる。
    # 0 なら無効（常に深度点群も使う）
    LIDAR_MAX_POINTS_FOR_DEPTH_DEFAULT: int = 100
    LIDAR_MAX_POINTS_FOR_DEPTH_MAX: int = 500
    LIDAR_MIN_POINTS_MAX: int = 200
    LIDAR_ROR_NB_POINTS_DEFAULT: int = 3
    LIDAR_ROR_NB_POINTS_MAX: int = 50
    LIDAR_ROR_RADIUS_DEFAULT: float = 0.8
    LIDAR_ROR_RADIUS_MAX: float = 5.0
    LIDAR_DBSCAN_EPS_DEFAULT: float = 1.2
    LIDAR_DBSCAN_EPS_MAX: float = 5.0
    LIDAR_DBSCAN_MIN_SAMPLES_DEFAULT: int = 5
    LIDAR_DBSCAN_MIN_SAMPLES_MAX: int = 100

    # Use LiDAR の既定。インスタンスごとの LiDAR 点群を作るかどうか。
    # 深度点群との混合（座標補正）は未実装で、現在は表示と外れ値除去まで
    BOXFIT_USE_LIDAR_DEFAULT: bool = True

    # LiDAR を基準に深度点群を補正する方式。
    # インスタンス単位では深度の範囲が狭く（車 1 台で 4〜5 m）、
    # スケールとオフセットがほぼ区別できないため 1 パラメータで足りる。
    # 実測（LiDAR 3 点）: scale 0.16 m / shift 0.15〜0.22 m / affine 0.22〜0.25 m
    DEPTH_CORRECTION_METHODS: list[str] = ["scale", "shift", "affine"]
    DEFAULT_DEPTH_CORRECTION_METHOD: str = "scale"

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

    # --- カメラ間の結合 ----------------------------------------------------
    # 点群を揃える基準の座標系を決めるセンサー。
    # カメラごとに sample_data のタイムスタンプが違うため、
    # 「ego 座標」の基準がカメラ間でずれている。カメラを跨いで点群を
    # 比べる前に、このセンサーの ego_pose へ全部揃える。
    #
    # LiDAR を使わない構成へ移す場合は、常に存在するカメラ
    # （CAM_FRONT など）を指定すれば同じ仕組みで動く
    EGO_REFERENCE_CHANNEL: str = "LIDAR_TOP"

    # カメラ間の同一インスタンス結合の判定方法。
    # 手法を増やすときは MERGE_METHODS に足す（UI の Selectbox がこれを使う）
    MERGE_METHODS: list[str] = ["BEV convex-hull"]
    DEFAULT_MERGE_METHOD: str = "BEV convex-hull"
    # BEV 凸包の重なり率の下限。
    # IoS（小さい方の面積で割る）なので、IoU より高めに取れる。
    # カメラごとに物体の違う面しか観測できないため、同一物体でも
    # IoU は 0.3 程度まで落ちる（IoS なら 0.5 前後）
    DEFAULT_MERGE_OVERLAP_THRESHOLD: float = 0.15
    MERGE_OVERLAP_THRESHOLD_MAX: float = 1.0
    # 重心距離の上限 [m]。これを超える組は凸包を作る前に捨てる
    DEFAULT_MERGE_MAX_CENTROID_DISTANCE: float = 3.0
    MERGE_MAX_CENTROID_DISTANCE_MAX: float = 10.0
    # 何フレームでマッチしたら結合するか。
    # 同じ物体が複数カメラに写るキーフレームは高々 1〜2 なので、
    # 割合ではなくフレーム数で判定する（割合だと 0/0.5/1 の 3 値しか取れない）
    DEFAULT_MERGE_MIN_MATCH_FRAMES: int = 1
    # 同一カメラの 2 トラックを同じグループへ入れてよい時間的な隔たり
    # （キーフレーム数）。カメラ A → カメラ B → カメラ A と写り込む物体は、
    # カメラ A 側で 2 つのトラックに分断される。同時に存在しないなら
    # 同じ物体でありうるので、この範囲内なら同居を許す。
    # 0 にすると同一カメラの同居を一切許さない（従来の挙動）
    DEFAULT_MERGE_MAX_SAME_CAMERA_GAP: int = 20
    MERGE_MAX_SAME_CAMERA_GAP_MAX: int = 40
    MERGE_MIN_MATCH_FRAMES_MAX: int = 5
    # 結合してよいラベルの条件（label / category_group / none）。
    # 判定にはトラック内で多数決したラベルを使う
    DEFAULT_MERGE_LABEL_MATCH: str = "category_group"

    # --- Box Fitting -------------------------------------------------------
    # 当てはめ手法。convex_hull_moa は BEV の凸包に対し、角度を刻んで
    # 外接矩形の面積が最小になる向きを探す
    BOXFIT_METHOD_DEFAULT: str = "convex_hull_moa"
    BOXFIT_METHODS: list[str] = ["convex_hull_moa"]
    # 向きの探索刻み [度]。細かくしても結果はほぼ変わらず、計算時間だけ伸びる
    BOXFIT_ANGLE_STEP_DEG_DEFAULT: float = 0.5
    BOXFIT_ANGLE_STEP_DEG_MIN: float = 0.1
    BOXFIT_ANGLE_STEP_DEG_MAX: float = 10.0
    # 高さを決めるパーセンタイル（下限, 上限）。最小・最大をそのまま使うと
    # 路面やマスクのはみ出し 1 点で箱が縦に伸びる
    BOXFIT_Z_PERCENTILE_LOW_DEFAULT: float = 1.0
    BOXFIT_Z_PERCENTILE_HIGH_DEFAULT: float = 99.0
    # 当てはめに必要な点数の下限。これに届かないインスタンスはボックスを
    # 作らない（点群は残す）。点が少なすぎると BEV の凸包が線や点に潰れ、
    # 向きも大きさも意味を持たない。
    # **判定はカメラ間結合の後、グローバルトラック単位・フレームごとに
    # 行う**（実際に当てはめへ渡す点群で数えるため）
    BOXFIT_MIN_POINTS_DEFAULT: int = 10
    BOXFIT_MIN_POINTS_MAX: int = 100

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

    BOXFIT_STUB_DELAY_SEC: float | None = 0.02

    # スタブ推論の1回あたりの待ち時間（本実装に差し替えたら None にする）
    DET2D_STUB_DELAY_SEC: float | None = 0.05

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
