# automatic-annotation-app
nuScenesデータセットの自動アノテーションを実施するデモ用Streamlit Webアプリ

## Project Overview
NuScenes datasetをDBに読み込み、推論サーバーでAIモデルによる自動アノテーション（3D bounding box）を実施。Streamlitで表示
- Frontend: Streamlit
- DB: SQLite（`webapp/app/models`フォルダにあるSQL Alchemy形式スキーマを使用する）
- Data: ホスト側のフォルダ（環境変数`HOST_DATA_ROOT`で指定）にnuscenesデータセットを格納してコンテナにマウント。メタデータを初期化時にDatabaseに読み込み、画像、点群データはローカルフォルダから直接読込
- 推論サーバー: 以下3種類の推論をパイプライン的に実施。FastAPIでトリガーと結果を返すREST APIを提供
  1. Detection2D: Grounding DINOを用いて、与えたラベルの2D bounding boxを検出し、SigLIP2でラベルを再判定
  2. Instance-Tracking: 1で検出したbounding boxesをプロンプトとして与えたSAM2を用いて、各インスタンスのマスクとtrack_idを取得
  3. Depth Estimation & Box Fitting: Depth-Anything-3でカメラ画像から推論した点群とLiDAR点群をミックスして、2で検出したマスク範囲にprojectionし、カメラ間の同一インスタンスを結合したうえで3D bounding boxを割り当て
- 結果はnuScenes形式（JSONテーブル群）でエクスポートできる
- 各推論のアルゴリズム詳細は`inference/CLAUDE.md`、各ページのUI仕様は`webapp/app/streamlit/CLAUDE.md`参照
- 以下サービスをDockerコンテナで構成
  - webapp: Streamlitによるフロントエンド＋SQLiteによるDB
  - inference: FastAPI＋AIによる推論サーバー
  - `common/`（webapp・inference共有モジュール）は両コンテナの `/opt/shared/common` に読み取り専用でマウントする

## Directory Structure
```
project-root/
:
└── webapp/                                  # docker compose はここで実行する
    ├── CLAUDE.md
    ├── README.md                            # セットアップ・使い方
    ├── docker-compose.yml
    ├── .env                                 # HOST_DATA_ROOT, HOST_UID/HOST_GID, GPU_DEVICE_ID 等（git管理外）
    ├── tools/                               # 調査用スクリプト（webapp コンテナで標準入力から実行）
    │   └── check_depth_boxfitting.py        # Box Fitting の結果（深度補正・距離フィルタ等）を DB から確認
    ├── common/                              # webapp・inference の共有モジュール（両コンテナの /opt/shared/common に :ro マウント）
    │   ├── box_ops.py                       # 2D ボックスの基本演算（IoU/IoS 等）
    │   ├── mask_rle.py                      # インスタンスマスクの RLE 表現
    │   ├── point_ops.py                     # 点群の間引き
    │   └── transform3d.py                   # 3D 座標変換の最小セット
    ├── checkpoints/                         # GroundingDINO/SAM2 の重み（GitHub 版イメージのみ使用。Transformers 版は ~/.cache/huggingface を使う）
    ├── webapp/                              # Streamlit＋SQLite（コンテナの /workspace）
    │   ├── Dockerfile
    │   ├── docker-entrypoint.sh             # 起動時に alembic upgrade head を実行
    │   ├── requirements.txt
    │   ├── alembic.ini
    │   ├── alembic/
    │   │   ├── env.py                       # render_as_batch=True 必須（SQLite）
    │   │   ├── script.py.mako
    │   │   └── versions/
    │   ├── scripts/
    │   │   └── import_nuscenes.py           # 空（未実装）。インポートは to_nusc_db.py を使う
    │   └── app/
    │       ├── json_conversion/             # 元のJSON形式データセットとDBとを相互変換するためのモジュール集
    │       │   ├── schemas_nuscenes.py      # NuScenes本体データセットJSONのPydantic形式スキーマ
    │       │   ├── to_nuscenes.py           # 空（未実装）。エクスポートは services/nuscenes_export_service.py が担う
    │       │   └── to_nusc_db.py            # NuScenes本体データセットをJSONからDBに変換（CLI）
    │       ├── core/
    │       │   ├── config.py                # 環境変数・設定（Pydantic Settings）
    │       │   └── logging.py               # ロギング設定
    │       ├── db/
    │       │   ├── base.py                  # DeclarativeBase（naming_convention 必須）
    │       │   ├── engine.py                # Engine生成 + SQLite PRAGMA（Alembic用は create_migration_engine）
    │       │   └── session.py               # 同期 Session ファクトリ
    │       ├── models/                      # ★手動作成・変更禁止ゾーン。SQLAlchemy ORMモデル（唯一の正）
    │       │   ├── __init__.py              # Alembicがモデルを検出できるよう全importを記載
    │       │   ├── dataset.py               # Dataset（データセット単位）
    │       │   ├── scene.py                 # Log, Scene, Sample
    │       │   ├── annotation.py            # Category, Attribute, Visibility, Instance, SampleAnnotation, annotation_attribute, SOURCE_* 定数
    │       │   ├── sensor.py                # Sensor, CalibratedSensor, EgoPose, SampleData
    │       │   ├── map.py                   # MapMeta
    │       │   └── ann_intermediate.py      # 自動アノテーションの中間出力（Detection2DParams, Detection2D, InstanceTracking2DParams, InstanceTracking2D, DepthEstimationParams, DepthEstimation, LidarPointcloud, BoxFitting3D）
    │       ├── repositories/                # DBアクセスの抽象化（クエリの責務）。戻り値は必ず dict
    │       │   ├── dataset.py               # データセット単位のクエリ
    │       │   ├── scene.py                 # シーン単位のクエリ
    │       │   ├── sensor.py                # センサーフレーム単位のクエリ
    │       │   ├── annotation.py            # アノテーション単位のクエリ
    │       │   ├── detection2d.py           # 2D検出 run と結果のクエリ
    │       │   ├── instance_tracking.py     # Instance Tracking run と結果のクエリ
    │       │   ├── depth_boxfitting.py      # 深度推定＋Box Fitting run と結果のクエリ
    │       │   └── nuscenes_export.py       # エクスポート対象行の取り出し
    │       ├── services/                    # ビジネスロジック層（Streamlit非依存。frame_image.py のみキャッシュ付き）
    │       │   ├── inference_client.py      # 推論サーバーとの通信（ジョブ投入・ポーリング・キャンセル・refilter）
    │       │   ├── detection2d_service.py   # 2D検出 run の保存と読み出し
    │       │   ├── instance_tracking_service.py # Instance Tracking run の保存と読み出し
    │       │   ├── depth_boxfitting_service.py  # 深度推定＋Box Fitting run の保存と読み出し（Instance/SampleAnnotation の prev/next チェーン構築を含む）
    │       │   ├── manual_merge.py          # 再実行時に手修正ボックスを引き継ぐ
    │       │   ├── nuscenes_export_service.py   # nuScenes 形式エクスポート（出力対象のみで prev/next を張り直す）
    │       │   ├── label_service.py         # ラベル体系の変換
    │       │   ├── sweep_service.py         # トラッキング入力フレーム（sweep）の選択
    │       │   ├── sample_service.py        # サンプル間引き・スライディングウィンドウのインデックス計算
    │       │   ├── camera_service.py        # カメラの表示順
    │       │   ├── annotation_2d.py         # 3D アノテーションのカメラ画像への 2D 投影
    │       │   ├── gt_bev.py                # GT ボックスの BEV 表示用変換
    │       │   ├── depth_pointcloud.py      # 深度マップから点群を作る
    │       │   ├── data_file_service.py     # センサーデータ（画像・点群）のファイル読み込み
    │       │   ├── derived_file_service.py  # DERIVED_ROOT 配下の派生ファイル読み込み
    │       │   ├── frame_image.py           # センサー画像の取得（Streamlit キャッシュ付き）
    │       │   ├── basemap_service.py       # basemap のリサイズキャッシュ（DERIVED_ROOT配下）
    │       │   └── geometry/                # 座標変換・点群投影・3D box 頂点生成
    │       └── streamlit/                   # StreamlitのUI実装（フロントエンド）
    │           ├── CLAUDE.md                # 各ページのUI仕様
    │           ├── main.py                  # メインページ（データセット選択）
    │           ├── state.py                 # セッション状態管理・ページガード（require_*）
    │           ├── data_access.py           # @st.cache_data 付きのデータ取得（clear_caches）
    │           ├── pages/
    │           │   ├── 1_Scene_Selection.py
    │           │   ├── 2_Detection2D.py
    │           │   ├── 3_Instance_Tracking.py
    │           │   ├── 4_Depth_Boxfitting.py
    │           │   └── 5_Export.py          # nuScenes 形式へのエクスポート
    │           └── components/
    │               ├── basemap.py           # シーンに対応する basemap の解決
    │               ├── waypoint_viewer.py   # basemap 上の自車軌跡
    │               ├── det2d_viewer.py      # 2D bounding box の描画
    │               ├── instance_tracking_viewer.py # インスタンスマスク・トラックの描画
    │               ├── depth_viewer.py      # 深度マップの可視化
    │               ├── pointcloud_viewer.py # 点群の 3D 表示（Plotly）
    │               ├── box3d_viewer.py      # 3D ボックスのカメラ画像への投影描画
    │               ├── bev_viewer.py        # BEV での 3D ボックス表示
    │               └── sample_token.py      # 選択中 sample_token の表示
    └── inference/                           # 推論サーバー（コンテナの /workspace）
        ├── CLAUDE.md                        # 推論フロー・パラメータ・実行単位の仕様
        ├── Dockerfile_devel                 # 既定（Transformers 版、CUDA 12.8）。compose の IMAGE_RUNTIME で切替
        ├── Dockerfile_devel_github          # GitHub 公式実装版（CUDA 12.4、checkpoints/ を使用）
        ├── Dockerfile                       # compose の既定では未使用
        ├── requirements.txt
        └── app/
            ├── main.py                      # FastAPIアプリ初期化・ルーター登録・モデルローダー登録・/health, /system/*
            ├── core/
            │   ├── config.py                # 環境変数・設定（ラベル体系・各種閾値もここ）
            │   ├── logging.py               # ロギング設定
            │   ├── jobs.py                  # 非同期推論ジョブの管理（POST→job_id→GETポーリング、ワーカー1本）
            │   └── models.py                # モデルの遅延ロード（LRU で MAX_RESIDENT_MODELS 個まで常駐）と gpu_lock
            ├── routers/                     # 各ステップの /jobs エンドポイント（depth_boxfitting は同期の /refilter も持つ）
            │   ├── det2d.py
            │   ├── instance_tracking.py
            │   └── depth_boxfitting.py
            ├── schemas/                     # リクエスト／レスポンススキーマ
            │   ├── det2d.py
            │   ├── instance_tracking.py
            │   └── depth_boxfitting.py
            ├── models_impl/                 # モデルのラッパー。*_stub.py は USE_STUB_MODELS=true 時の GPU 不要スタブ
            │   ├── grounding_dino.py, groundingdino_predict.py, prompt.py   # Grounding DINO（Transformers 版）
            │   ├── siglip2_classifier.py    # SigLIP2 zero-shot 分類（ラベル再判定）
            │   ├── sam2_tracker.py, sam2_video.py, sam2_masks.py           # SAM2 Video（Transformers 版）
            │   ├── da3_depth.py             # Depth-Anything-3（公式リポジトリ版）
            │   └── boxfitting_pipeline.py   # Depth Estimation & Box Fitting の本体
            └── services/                    # 推論の前後処理（モデル非依存）
                ├── postprocess.py, reclassify.py, whole_image.py, crop.py  # Detection2D: IoS結合・再判定・全画面救済
                ├── tracking_match.py, fb_matching.py                       # Instance Tracking: track_id 引き継ぎ
                ├── depth_ops.py, depth_correction.py, lidar_ops.py         # 深度→点群、LiDAR 基準の深度補正、LiDAR 読込・地面除去
                ├── distance_filter.py, lshape_filter.py, refilter.py       # 点群ノイズ除去・保存済み深度からの再フィルタ
                ├── inter_cam_merge.py       # カメラ間の同一インスタンス結合（global_track_id）
                └── box_fitting.py, box_fitting_moa.py, box_bottom.py, box_size.py  # 3D ボックス当てはめと補正
```

## webappコンテナ（Streamlitによるフロントエンド＋SQLiteによるDB）
### Schema Rules（最重要）
- `webapp/app/models/` が唯一のスキーマ定義とする
- CRUDは必ずmodelsから派生させる
- **カラム追加・変更は必ずmodels/を先に修正してから伝播させる**
- モデル変更時はAlembicマイグレーションも同時に生成すること
- 実際のデータ構造は環境変数`HOST_DATA_ROOT`で指定したフォルダ内のデータセットも参照する（カメラ画像・LiDAR点群データ）
- 同じデータセットやマップデータを複数のメタデータが参照（例：trainvalとminiが同じデータを参照）すると、tokenの重複が発生し、DBのprimary keyのidentity制約でエラーが出る。このケースをハンドルするにはtokenをprimary keyにする前に名前空間を付与する等が有効だが、今回はこのようなハンドリングは実施せず、token重複を防ぐよう運用側でカバーする。よって**1データセットで読み込めるメタデータ・マップデータは1つのみという制約をドキュメントに明記するものとする**
  - この制約は `to_nusc_db.py` の衝突ガードが機械的に検出して中断する（運用任せにしない）
  - 同一データセットルート内の `v1.0-mini` と `v1.0-trainval` は mini が trainval の部分集合であり token が完全に重複するため、同じDBには入れられない。切り替えは `--replace` を使う

#### モデル定義の必須ルール
- **全てのモデルに `dataset: Mapped["Dataset"] = relationship()` を持たせる**
  - `Dataset` へのrelationshipが無いと、SQLAlchemyのunit-of-workが `datasets` → 子テーブルのINSERT順を解決できず、`FOREIGN KEY constraint failed` になる。FK列があるだけでは順序は決まらない
  - 同様に `scene_token` を持つ `*Params` テーブルには `scene: Mapped["Scene"] = relationship()` も必要
  - `Dataset` 側に逆方向のコレクションは張らない（削除はDBのCASCADEに委ねる方が速い）
- **モジュール間のimportは `if TYPE_CHECKING:` に置き、relationshipは文字列のフォワード参照にする**（循環import防止）
- boolean の既定値は `server_default=text("0")` を使う
- 自己参照FK（`prev`/`next`）と `ondelete="SET NULL"` / `"RESTRICT"` の対象列には `index=True` を付ける（削除時のトリガをindex scanにするため）

#### アノテーションの生成元管理
- `Instance.source` / `SampleAnnotation.source` は `'imported' | 'auto' | 'manual'`（定数は `app/models/annotation.py`）
- `SampleAnnotation.depth_estimation_params_id`（nullable FK）で、どのBox Fitting実行が生成したボックスかを辿れる
  - 実行単位の行を削除するとCASCADEでその実行の成果物だけが消え、パラメータを変えた再推論がクリーンにやり直せる
  - GT（`source='imported'`）はこのFKがNULLなので巻き込まれない
- 推論3ステップは `*Params` テーブルが「1回の実行」を表し、後段が前段の `*Params.id` を参照する（系譜: `DepthEstimationParams` → `InstanceTracking2DParams` → `Detection2DParams`）
- 深度マップは絶対にDBに入れず `.npz` としてディスクに置き `DepthEstimation.depth_path` にパスのみ保持する。マスクはCOCO RLE（`InstanceTracking2D.mask_rle`）で保持し、肥大化したらファイル方式に切り替える

#### 集計・性能計測
- シーン横断の集計は相関 EXISTS を使う。JOIN + COUNT(DISTINCT) はアノテーション全行を走査するため、trainval 規模で 4〜5 倍遅くなる
- 性能計測は min-of-N で行う。初回はページキャッシュが未温で桁が変わる

### DB / Alembic Rules
- **同期 Session を使う（AsyncSessionは使わない）**。Streamlitは同期実行モデルであり、DBも単一ファイルのSQLiteなのでasyncの利点がなく複雑さだけが増す
- `sessionmaker(expire_on_commit=False)` 必須。Trueだとcommit直後に属性が期限切れになり、UI描画時に `DetachedInstanceError` になる
- SQLiteのPRAGMAは**接続ごと**に適用する（`event.listen(engine, "connect")`）。一度実行して終わりではない
  - `foreign_keys=ON`（これが無いと `ondelete='CASCADE'` が効かない）
  - `journal_mode=WAL`（推論の書き込み中もUIの読み取りをブロックしない）
  - `synchronous=NORMAL`, `busy_timeout`
- `db/base.py` の `naming_convention` は**初回マイグレーションを切る前に**決めること。SQLiteのbatchモードは制約名が無いと `Constraint must have a name` で失敗する。後から変えると既存DBの制約名と食い違う
- Alembicの `env.py` は `render_as_batch=True` 必須（SQLiteは `ALTER TABLE` で制約変更ができない）
- **Alembic専用のEngineはFK PRAGMAを適用しない**（`create_migration_engine`）。batchはテーブルを作り直すため、FKがONだと再作成の過程で `ON DELETE CASCADE` が発火して行が消える危険がある
- `alembic check` をCIに入れるとモデルとDBの乖離を防げる
- SQLiteファイルは名前付きボリュームに置く（バインドマウントだとホストOSによってWALのロックが正しく効かない）

### Config Rules
- **モジュールレベルで `Settings()` を評価しない**。環境変数が1つ欠けただけでモジュールのimport自体が失敗し、alembicのenv.pyやCLIまで巻き込んで落ちる。必ず `@lru_cache` 付きの `get_settings()` 経由で遅延評価する
- `.env` は `pydantic-settings` の `env_file` に頼らず、docker-composeの `env_file:` で環境変数として注入する。`env_file` はプロセスのCWD基準で解決されるためコンテナ内では当てにならない
- アプリが参照するのは `DATA_ROOT`（コンテナ内パス）であり、`HOST_DATA_ROOT` はcompose側のマウント指定にのみ使う

### Docker Rules
- **`./webapp` を `/workspace` にマウントし、WORKDIRは `/workspace`**。`./webapp/app` を `/app` にマウントすると `/app` が `app` パッケージの「中身」になり、`from app.core.config import ...` が `ModuleNotFoundError` になる。alembic.ini と alembic/ もコンテナ内に必要
- **`ENV PYTHONPATH=/workspace:/opt/shared` 必須**。`streamlit run` はスクリプトのあるディレクトリをsys.pathに入れるだけでCWDは入れない。`/opt/shared` は共有モジュール `common` の親
- **共有モジュール `./common` は `/workspace` の外（`/opt/shared/common`）に `:ro` でマウントする**。`/workspace` の内側に重ねると、ホストの `./webapp/webapp` に空ディレクトリが作られる
- **派生物（DERIVED_ROOT）は `/data` の外に置く**。`${HOST_DATA_ROOT}:/data:ro` の内側にボリュームを重ねると、マウントポイントを作れず `read-only file system` で起動に失敗する。`/derived` を使う
- **`USER` 切り替え前に `mkdir -p /db /derived && chown`** する。Dockerは空の名前付きボリュームを初期化する際にイメージ側ディレクトリの所有者をコピーするため、これが無いとボリュームがroot所有になり非rootユーザーがSQLiteを書けない
- **起動コマンドは `ENTRYPOINT` ではなく `CMD`** に置く。`ENTRYPOINT` はマイグレーション適用スクリプト専用にし、`docker compose run --rm webapp python -m ...` でCLIを差し替え実行できるようにする
- データセットは `:ro` でマウントし、元データを壊す事故を防ぐ
- **SQLite本体のインストールは不要**。Pythonの `sqlite3` は標準ライブラリで、公式イメージにlibsqlite3がリンク済み（bookworm系はSQLite 3.40+で、SQLAlchemy 2.xがINSERTに使うRETURNING構文の要件3.35+を満たす）。`sqlite3` CLIはデバッグ用に任意で入れる
- ビルド時のUID/GIDはホストのユーザーに合わせる（`UID=$(id -u) GID=$(id -g) docker compose up --build`）

### Import CLI（to_nusc_db.py）Rules
- 実行例: `docker compose run --rm webapp python -m app.json_conversion.to_nusc_db --name mini --version v1.0-mini --dataroot nuscenes`
- `--dataroot` は必須。`DATA_ROOT` 直下には複数のデータセットルートが並ぶ想定（`/data/nuscenes/v1.0-mini/...`）
- **INSERTはFK依存の位相順に流す**
  `dataset → log → map_meta → sensor → calibrated_sensor → ego_pose → scene → sample → sample_data → category/attribute/visibility → instance → sample_annotation → annotation_attributes`
- **`prev`/`next` の自己参照FKは第2パスのUPDATEで埋める**。1パスで入れると必ず前方参照になりFK違反で落ちる
- **ORMではなくCoreのTable（`Model.__table__`）に対してexecutemanyする**。`session.execute(insert(Model), [dict,...])` はORMのバルク処理パスに入り、主キー同期や永続オブジェクト追従の制約に引っかかる
- **大きいJSONはストリーミングで読む**。trainvalの `sample_annotation.json` は数百MBあり、一括読み込みするとPythonオブジェクト展開後に数GBを占めてメモリ不足で落ちる
  - 対象は `ego_pose.json` / `sample_data.json` / `sample_annotation.json`。ijson（Cバックエンド yajl2_c）で逐次パースし、5000件ずつINSERTする
  - `ijson.items(f, "item", use_float=True)` の `use_float=True` は必須。既定では小数がDecimalで返り、JSON列への書き込みで落ちる
  - 第2パスのリンクもリストに溜めず、JSONを流し直す
  - 実測: 168万アノテーションで約190秒、ピークRSS 751MB
- `MapMeta` は3ファイルの突き合わせで作る。`map.json` 単体にはlocationもversionもcanvas_edgeも無い
  - `location`: `log_tokens` から `log.location` を引く
  - `version` / `canvas_edge`: `maps/expansion/<location>.json`
  - Map Expansionが無い場合はbasemap PNGの画素数 × 0.1 m/px から推定（nuScenesのbasemapは0.1 m/pixel）
- インポート全体を1トランザクションで実行し、失敗時は全ロールバックする

### Streamlit Rules
- 各ページのUI仕様は`webapp/app/streamlit/CLAUDE.md`参照
- **ウィジェットキーと正規キーを分離する**（`app/streamlit/state.py`）
  - Streamlitのマルチページでは、ウィジェットに紐づく `session_state` のキーは、そのウィジェットが描画されない実行で破棄される。ページ1のselectboxの値はページ2に移った時点で消え得る
  - ウィジェットは `_w_` 接頭辞、アプリが参照する正規キーは `sel_` 接頭辞。ウィジェットの `on_change` で正規キーへ書き写す
  - 上位の選択が変わったら下位を再帰的にクリアする（データセット変更 → シーン・サンプル・各推論の実行IDまで全て破棄）
- **各ページの先頭で `require_*()` を1行呼ぶ**。未選択時はメッセージとリンクを出して `st.stop()` する
- **`st.page_link` は必ず例外を握って呼ぶ**（`_safe_page_link`）。ページ未登録時に `KeyError: 'url_pathname'` を投げ、ガード自体が例外死してメッセージすら出なくなる
- **Repositoryは ORM インスタンスではなく dict を返す**。`@st.cache_data` は戻り値を保持するため、ORMオブジェクトを返すとSessionが閉じた後に `DetachedInstanceError` になる
- **`@st.cache_data` の引数に `dataset_id` / `scene_token` を必ず含める**。キャッシュキーは引数から作られるので、含め忘れるとデータセットを切り替えても古い結果が返り続ける。データ書き換え後は `clear_caches()` を呼ぶ
- Engine / sessionmaker を `@st.cache_resource` で包む必要はない。`app/db/` 側の `lru_cache` で1つに保たれる（Streamlitはスクリプトを再実行するがimport済みモジュールは再読み込みしない）
- `use_container_width` は非推奨。`width="stretch"` を使う
- テーブルの行選択だけでは正規キーに書かず、ボタンで明示的に確定させる（ページ移動で選択が失われるため）
- basemapのリサイズキャッシュは `DERIVED_ROOT/basemap_cache/<dataset_id>/<name>_x<scale>.png` に置く。`/data` は読み取り専用なのでデータセット配下には書けない
- `canvas_edge` をコードに直書きしない。DBの `MapMeta` から取る
- Figureの組み立て（`build_*_figure`）と描画（`render_*`）を分ける。テストしやすく、使い回しも効く
- 軌跡表示はマップ全体ではなく軌跡の範囲 ± マージンにズームする（マップ全体だとシーンが点にしか見えない）
- ファイル I/O は`services/`に置き、Streamlit にも DB にも依存させない。token → filename の解決は data_access の責務
- 画像など重いリソースのキャッシュキーは、それが実際に対応する単位（basemap ならロケーション/パス）にする。呼び出し側の単位（scene_token）でキャッシュすると同じ実体が重複して載る
- コンポーネントは「純粋な描画関数」と「取得込みの高レベル関数」を分ける。前者は data_access に依存させない
- ウィジェットの key に値を書き込んで「自動リセット」しない。ウィジェット状態と競合し、以後の操作が反映されなくなるため。無効化は使用時に AND を取る
- 選択肢や初期値が動的に変わるウィジェットには必ず key を付ける。key 無しで value= を変えると別ウィジェット扱いになり、ユーザーの選択が失われるため

### Testing
- Streamlitのページは `streamlit.testing.v1.AppTest` でヘッドレスに実行して検証できる
  ```python
  at = AppTest.from_file("app/streamlit/pages/1_Scene_Selection.py")
  at.session_state["sel_dataset_id"] = dataset_id
  at.run()
  assert not at.exception
  ```

## inferenceコンテナ（FastAPI＋AIによる推論サーバー）

### API / モデル管理 Rules
- 推論はジョブ方式（`POST /<step>/jobs` → 202 で job_id → `GET` でポーリング、`DELETE` でキャンセル）。ジョブはプロセス内メモリに保持し、ワーカーは1本（uvicorn も worker=1 前提）
- モデルは起動時にロードしない。`main.py` の lifespan でローダーを登録するだけにし、最初のリクエストでロードする（`MAX_RESIDENT_MODELS` を超えたら古いものから解放）。GPU 実行は `gpu_lock` で直列化する
- 重いライブラリの import はローダー関数の内側に置く。トップレベルで import すると1つ壊れただけでサーバーが起動しなくなる
- `/health` はモデルのロードを待たずに即座に返す。compose の healthcheck で webapp の起動を待たせているため
- `USE_STUB_MODELS=true` で GPU 不要のスタブ（`models_impl/*_stub.py`）に切り替わる。新しいモデルを追加するときはスタブも用意する

### Docker Rules
- 既定のイメージは `Dockerfile_devel`（Transformers 版）。GitHub 公式実装版を使う場合は `IMAGE_RUNTIME=devel_github IMAGE_TAG=cu124-devel-github` を指定する
- 推論サーバーの requirements.txt には、ベースイメージに既にあるパッケージを書かない。無指定で書くと pip が最新版へ上げ、numpy が 2.x に置き換わってベースイメージの C 拡張（matplotlib 等）が壊れる。GroundingDINO は matplotlib を import するため直撃する
- Dockerfile の末尾に import のスモークテストを置く。依存の壊れは最初の推論まで発覚せず、原因の切り分けに時間を取られる
