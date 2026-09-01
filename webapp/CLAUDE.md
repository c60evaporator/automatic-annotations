# automatic-annotation-app
nuScenesデータセットの自動アノテーションを実施するデモ用Streamlit Webアプリ

## Project Overview
NuScenes datasetをDBに読み込み、推論サーバーでAIモデルによる自動アノテーション（3D bounding box）を実施。Streamlitで表示
- Frontend: Streamlit
- DB: SQLite（`webapp/app/models`フォルダにあるSQL Alchemy形式スキーマを使用する）
- Data: ホスト側のフォルダ（環境変数`HOST_DATA_ROOT`で指定）にnuscenesデータセットを格納してコンテナにマウント。メタデータを初期化時にDatabaseに読み込み、画像、点群データはローカルフォルダから直接読込
- 推論サーバー: 以下3種類の推論をパイプライン的に実施。FastAPIでトリガーと結果を返すREST APIを提供
  1. 2D Object Detection: Grounding DINOを用いて、与えたラベルの2D bounding boxを検出
  2. Instance-Tracking: 1で検出したbounding boxesをプロンプトとして与えたSAM2を用いて、各インスタンスのマスクとtrack_idを取得
  3. Depth Estimation & Box Fitting: Depth-Anything-3でカメラ画像から推論した点群とLiDAR点群をミックスして、2で検出したマスク範囲にprojectionして3D bounding boxを割り当て
- 以下サービスをDockerコンテナで構成
  - webapp: Streamlitによるフロントエンド＋SQLiteによるDB
  - inference: FastAPI＋AIによる推論サーバー

## Directory Structure
```
project-root/
:
└── webapp/                                  # ★ここがコンテナの /workspace になる
    ├── CLAUDE.md
    ├── docker-compose.yml
    ├── .env
    ├── webapp/
    │   ├── Dockerfile
    │   ├── docker-entrypoint.sh            # 起動時に alembic upgrade head を実行
    │   ├── requirements.txt
    │   ├── alembic.ini
    │   ├── alembic/
    │   │   ├── env.py                      # render_as_batch=True 必須（SQLite）
    │   │   ├── script.py.mako
    │   │   └── versions/
    │   └── app/
    │       ├── json_conversion/            # 元のJSON形式データセットとDBとを相互変換するためのモジュール集
    │       │   ├── schemas_nuscenes.py     # NuScenes本体データセットJSONのPydantic形式スキーマ
    │       │   ├── to_nuscenes.py          # NuScenes本体データセットをDBからJSONに変換
    │       │   └── to_nusc_db.py           # NuScenes本体データセットをJSONからDBに変換（CLI）
    │       ├── core/
    │       │   ├── config.py               # 環境変数・設定（Pydantic Settings）
    │       │   └── logging.py              # ロギング設定
    │       ├── db/
    │       │   ├── base.py                 # DeclarativeBase（naming_convention 必須）
    │       │   ├── engine.py               # Engine生成 + SQLite PRAGMA
    │       │   └── session.py              # 同期 Session ファクトリ
    │       ├── models/                     # ★手動作成・変更禁止ゾーン。SQLAlchemy ORMモデル（唯一の正）
    │       │   ├── __init__.py             # Alembicがモデルを検出できるよう全importを記載
    │       │   ├── dataset.py              # Dataset（データセット単位）
    │       │   ├── scene.py                # Log, Scene, Sample
    │       │   ├── annotation.py           # Category, Attribute, Visibility, Instance, SampleAnnotation, annotation_attribute
    │       │   ├── sensor.py               # Sensor, CalibratedSensor, EgoPose, SampleData
    │       │   ├── map.py                  # MapMeta
    │       │   └── ann_intermediate.py     # 自動アノテーションの中間出力（Detection2DParams, Detection2D, InstanceTracking2DParams, InstanceTracking2D, DepthEstimationParams, DepthEstimation）
    │       ├── services/                   # ビジネスロジック層（Streamlit非依存）
    │       │   ├── annotation_service.py   # prev/next チェーンの書き換え
    │       │   ├── basemap_service.py      # basemap のリサイズキャッシュ（DERIVED_ROOT配下）
    │       │   ├── nuscenes_export_service.py  # nuScenes 形式エクスポート
    │       │   └── nuscenes_export_builders.py # 各 JSON ファイルの組み立て
    │       ├── repositories/               # DBアクセスの抽象化（クエリの責務）。戻り値は必ず dict
    │       │   ├── dataset.py              # DatasetRepository
    │       │   ├── scene.py                # SceneRepository
    │       │   ├── annotation.py           # AnnotationRepository
    │       │   ├── sensor.py               # SensorRepository
    │       │   └── map.py                  # MapRepository
    │       └── streamlit/                  # StreamlitのUI実装（フロントエンド）
    │           ├── main.py                 # メインページ（データセット選択）
    │           ├── state.py                # セッション状態管理・ページガード
    │           ├── data_access.py          # @st.cache_data 付きのデータ取得
    │           ├── pages/
    │           │   ├── 1_Scene_Selection.py
    │           │   ├── 2_Detection2D.py
    │           │   ├── 3_Instance_Tracking.py
    │           │   └── 4_Depth_Boxfitting.py
    │           └── components/
    │               ├── waypoint_viewer.py
    │               ├── det2d_viewer.py
    │               ├── instance_tracking_viewer.py
    │               ├── depth_est_viewer.py
    │               └── pointcloud_viewer.py
    ├── inference/                       # 推論サーバー
    │   ├── Dockerfile
    │   └── app/
    │       ├── main.py                  # FastAPIアプリ初期化・ルーター登録
    │       ├── core/
    │       │   ├── config.py            # 環境変数・設定（Pydantic Settings）
    │       │   ├── logging.py           # ロギング設定
    │       │   ├── jobs.py              # 非同期推論ジョブの管理
    │       │   └── models.py            # モデルの遅延ロードと GPU 占有の管理
    │       ├── routers/
    │       │   ├── det2d.py
    │       │   ├── instance_tracking.py
    │       │   └── depth_boxfitting.py
    │       └── schemas/
    │           ├── det2d.py
    │           ├── instance_tracking.py
    │           └── depth_boxfitting.py
    ├── checkpoints/                    # GroundingDINO/SAM2の重み
    └── common/                         # Webアプリ・推論の共通処理（座標変換へルパ等）
    │   ├── Dockerfile
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
- **`ENV PYTHONPATH=/workspace` 必須**。`streamlit run` はスクリプトのあるディレクトリをsys.pathに入れるだけでCWDは入れない
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
- 画像など重いリソースのキャッシュキーは、それが実際に対応する単位（basemap ならロケーション/パス）にする。呼び出し側の単位（scene_token）でキャッシュすると同じ実体が重複して載る
- コンポーネントは「純粋な描画関数」と「取得込みの高レベル関数」を分ける。前者は data_access に依存させない

### Testing
- Streamlitのページは `streamlit.testing.v1.AppTest` でヘッドレスに実行して検証できる
  ```python
  at = AppTest.from_file("app/streamlit/pages/1_Scene_Selection.py")
  at.session_state["sel_dataset_id"] = dataset_id
  at.run()
  assert not at.exception
  ```

## inferenceコンテナ（FastAPI＋AIによる推論サーバー）

### Docker Rules
- 推論サーバーの requirements.txt には、ベースイメージに既にあるパッケージを書かない。無指定で書くと pip が最新版へ上げ、numpy が 2.x に置き換わってベースイメージの C 拡張（matplotlib 等）が壊れる。GroundingDINO は matplotlib を import するため直撃する
- Dockerfile の末尾に import のスモークテストを置く。依存の壊れは最初の推論まで発覚せず、原因の切り分けに時間を取られる
