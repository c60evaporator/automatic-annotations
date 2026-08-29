# nuscenes-viewer
nuScenesデータセットを可視化＆データやアノテーションを追加・修正するためのWebアプリ

## Project Overview
NuScenes dataset + Map expansion visualizer / annotation tool
- Backend: FastAPI (Python 3.12)
- Frontend: React + TypeScript + Deck.gl
- Data: ローカルフォルダのnuscenesデータセットのうち、メタデータを初期化時にDatabaseに読み込み、画像、点群データはローカルフォルダから直接読込
- DB schema: backend/app/modelsフォルダにあるSQL Alchemy形式スキーマを使用する
- 全サービスをDockerコンテナで構成

## Directory Structure
```
project-root/
├── CLAUDE.md
├── docker-compose.yml
├── .env
├── backend/
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── app/
│       ├── main.py                         # FastAPIアプリ初期化・ルーター登録
│       ├── dependencies.py                 # DBセッションなど共通依存関係
│       ├── json_conversion/                # 元のJSON形式データセットとDBとを相互変換するためのモジュール集
│       │   ├── schemas_nuscenes.py         # NuScenes本体データセットJSONのPydantic形式スキーマ
│       │   ├── to_nuscenes.py              # NuScenes本体データセットをDBからJSONに変換
│       │   ├── to_nusc_db.py               # NuScenes本体データセットをJSONからDBに変換
│       │   ├── schemas_mapexpansion.py     # Map expansionデータセットJSONのPydantic形式スキーマ
│       │   ├── to_map_db.py                # Map expansionデータセットをJSONからDBに変換
│       │   ├── to_mapexpansion.py          # Map expansionデータセットをDBからJSONに変換
│       │   └── tokens.py                   # token名前空間化（uuid5）ユーティリティ
│       ├── core/
│       │   ├── config.py                   # 環境変数・設定（Pydantic Settings）
│       │   └── logging.py                  # ロギング設定
│       ├── db/
│       │   ├── base.py                     # DeclarativeBase
│       │   ├── session.py                  # AsyncSession ファクトリ
│       │   └── poitgis.py                  # PostGIS初期化・拡張確認（ファイル名は typo だが変更禁止）
│       ├── models/                         # ★手動作成・変更禁止ゾーン。SQLAlchemy ORMモデル（唯一の正）
│       │   ├── __init__.py                 # Alembicがモデルを検出できるよう全importを記載
│       │   ├── dataset.py                  # Dataset（データセット単位。map_set_idでマップを参照）
│       │   ├── map_set.py                  # MapSet（マップ一式の単位。複数datasetで共有）
│       │   ├── scene.py                    # Scene, Sample
│       │   ├── annotation.py               # SampleAnnotation, Instance, Category
│       │   ├── sensor.py                   # Sensor, CalibratedSensor, EgoPose
│       │   └── map.py                      # Map expansion（PostGISジオメトリ含む）
│       ├── schemas/                        # Pydantic スキーマ（APIスキーマ）
│       │   ├── scene.py                    # SceneResponse, SampleResponse
│       │   ├── annotation.py               # BoundingBox3DResponse, AnnotationResponse
│       │   ├── sensor.py                   # CalibratedSensorResponse, EgoPoseResponse
│       │   ├── map.py                      # Map expansion (MapResponse, GeoJSONFeature)
│       │   └── common.py                   # Point3D, Quaternion, Dimensions3D など共通型
│       ├── lib/
│       │   ├── storage.py                  # ローカル / S3 のファイル読み出し
│       │   └── basemap.py                  # basemap パス解決（投入時）とデータルート検証（配信時）
│       ├── converters/                     # DB → APIスキーマへの変換ロジック
│       │   ├── annotation.py               # SampleAnnotation → BoundingBox3D
│       │   ├── scene.py                    # Scene → SceneResponse
│       │   ├── sensor.py                   # EgoPose → 変換行列など
│       │   ├── dataset.py                  # Dataset / MapSet → レスポンス
│       │   ├── map.py                      # MapMeta → MapMetaResponse（GPS原点を settings.yml から解決）
│       │   └── geometry.py                 # GeoAlchemy2 → GeoJSON変換など
│       ├── services/                       # ビジネスロジック層（Repository + Converter の組合せ）
│       │   ├── map_set_link_service.py     # dataset ↔ map_set の紐付け（CLI 3本で共有）
│       │   ├── scene_import_service.py     # POST /scenes/import
│       │   ├── scene_delete_service.py     # DELETE /scenes/{token}
│       │   ├── annotation_edit_service.py  # prev/next チェーンの書き換え
│       │   ├── annotation_merger.py        # SampleAnnotation + AnnotationEdit のマージ
│       │   ├── nuscenes_export_service.py  # nuScenes 形式エクスポート（dataset スコープ）
│       │   └── nuscenes_export_builders.py # 各 JSON ファイルの組み立て
│       ├── repositories/                   # DBアクセスの抽象化（クエリの責務）
│       │   ├── dataset.py                  # DatasetRepository, MapSetRepository（作成・削除含む）
│       │   ├── scene.py                    # SceneRepository
│       │   ├── annotation.py               # AnnotationRepository
│       │   ├── sensor.py                   # SensorRepository
│       │   └── map.py                      # MapRepository（空間クエリ含む） Map expansion用
│       └── api/
│           └── v1/
│               ├── router.py               # v1ルーターの集約
│               └── endpoints/
│                   ├── scenes.py           # GET /scenes, GET /scenes/{token}, GET /scenes/{token}/samples, GET /scenes/{token}/ego-poses
│                   ├── samples.py          # GET /samples/{token}, GET /samples/{token}/annotations, GET /samples/{token}/sensor-data, GET /samples/{token}/instances
│                   ├── annotations.py      # GET /annotations, GET /annotations/{token}, PATCH /annotations/{token}
│                   ├── sensors.py          # GET /sensors, GET /calibrated-sensors, GET /ego-poses, GET /sensor-data/{token}/image, GET /sensor-data/{token}/pointcloud
│                   ├── maps.py             # GET /maps, GET /maps/{token}, GET /maps/{token}/geojson, GET /maps/{location}/basemap
│                   ├── categories.py       # GET /categories
│                   ├── attributes.py       # GET /attributes
│                   ├── visibilities.py     # GET /visibilities
│                   ├── datasets.py         # GET /datasets, GET /datasets/{id}, GET /map-sets（dataset_id 不要）
│                   ├── export.py           # GET /export/nuscenes, GET /export/nuscenes/{scene_token}
│                   ├── instances.py        # GET /instances, GET /instances/{token}, GET /instances/{token}/annotations, GET /instances/{token}/best-camera
│                   └── logs.py             # GET /logs
├── frontend/
│   ├── Dockerfile
│   ├── vitest.config.ts
│   └── src/
│       ├── pages/               # ScenePage, SamplePage, InstancePage, AnnotationPage, MapPage, SampleMapPage
│       ├── components/          # layout/, common/, scene/, sample/, instance/, annotation/, map/, sample-map/, ui/
│       ├── api/                 # TanStack Query hooks（client.ts の apiFetch / apiUrl が URL に /datasets/{id} を差し込む）
│       ├── store/               # viewerStore, navigationStore, mapLayerStore, layerStore
│       ├── types/               # annotation, scene, sensor, map, navigation, common
│       ├── layers/              # MapAnnotationLayers.ts（Deck.gl レイヤー定義）
│       └── lib/                 # coordinateUtils.ts, canvasUtils.ts, utils.ts
└── db/
    └── initdb.d/
        ├── 01_init.sh
        └── 02-init.sql
```

## Schema Rules（最重要）
- `backend/app/models/` が唯一のスキーマ定義とする
- Pydanticスキーマ・CRUDは必ずmodelsから派生させる
- **カラム追加・変更は必ずmodels/を先に修正してから伝播させる**
- モデル変更時はAlembicマイグレーションも同時に生成すること
- 実際のデータ構造は`./data/nuscenes`フォルダのシンボリックリンク先のデータセットも参照する

### 複数データセット対応（datasets / map_sets）
- データ系の全テーブルは `dataset_id`（FK → datasets, CASCADE）、マップ系の全テーブルは
  `map_set_id`（FK → map_sets, CASCADE）を持つ。どちらも **NOT NULL・server_default なし**
  → 書き込み経路（サービス層 / エンドポイント / インポート）が必ず明示的に値を決める
- `datasets.map_set_id`（RESTRICT）でデータセットが使うマップ一式を指す。
  trainval と mini のように同じマップを使うデータセットは同一 map_set を共有し、複製しない
- `datasets.map_set_id` は **nullable**（マップなしのデータセット登録を許容する）。
  後から `scripts/link_map_set.py` で紐付け／解除できる。紐付けロジックは
  `app/services/map_set_link_service.py` に集約し、CLI 3 本で共有する
  （location 整合性チェック: dataset の logs.location ⊆ map_set の map_meta.location）
- **`/datasets` `/map-sets` を除く全エンドポイントが URL に dataset を含む**
  （`/api/v1/datasets/{dataset_id}/...`。`dependencies.get_dataset` → `CurrentDataset` が
  Path から読む）。dataset を含まない URL はルートが無いので 404、不明な ID も 404。
  既定データセットへのフォールバックは作らない。
  プレフィックスは `api/v1/router.py` の `DATASET_SCOPED_PREFIX` で一括付与し、
  各エンドポイントのデコレータには書かない
- maps 系だけは dataset ではなく **その dataset が参照する map_set** にスコープする。
  `map_set_id` が NULL のときは `GET /maps` が空リスト、他は 404
- 「存在しない token」と「他データセットの token」は**区別せず同一の 404 + 同一 detail** を返す
  （区別すると他データセットのレコードの存在有無が漏れる）。読み取りだけでなく
  書き込み（PATCH/POST/DELETE annotations, DELETE scenes）も同じガードを通す
- 未紐付けデータセットへの `POST /scenes/import` は log.location の検証をスキップし、
  `SceneImportResult.warnings` で通知する。スキップ判定は必ずサービス層が
  DB の `dataset.map_set_id` を見て行う（リクエスト側のフラグは受け付けない）。
  なお **投入先は URL の `/datasets/{dataset_id}` で一意に決まる**（Form では受け取らない）
- インポート時に token を `uuid5(dataset_id または map_set_id, 元token)` で名前空間化し、
  元 token は各テーブルの `source_token` に保存する（mini と trainval の token 衝突対策）。
  変換の実装は `app/json_conversion/tokens.py` の `make_token_mapper` のみを使う
- POST /scenes/import に渡す JSON の token は常に「元データセットの token（source 空間）」。
  DB の既存行を指す参照（calibrated_sensor.sensor_token 等）も同じ規則で変換して解決する
- **エクスポートは既定で source 空間の token を出力する**（`token_format=source`）。
  逆変換は uuid5 を逆算できないため DB の `source_token` を引く。実装は
  `app/services/export_token_map.py` に集約し、builder には書かない
  （組み立て済みレコードをキー名ルールで一括変換する。`token` / `prev` / `next` /
  `*_token` / `*_tokens` が対象）。`source_token` が NULL のレコード
  （`annotation_edits` / `instance_edits` / `map_meta`）は名前空間化後の token のまま出力する。
  `token_format=internal` は DB 突き合わせ用で、**再インポートには使えない**
  （source 空間を前提とする import 側で uuid5 が二重に掛かる）
- URL のパスに現れる token（`/scenes/{token}` 等）は常に **DB の token**。
  token_format が変えるのはエクスポート JSON の中身だけ
- リポジトリは dataset_id を自力で推論しない（**引数で必須**。デフォルト値 None を付けない）
- **JOIN では起点テーブル 1 箇所にだけ dataset_id 条件を付ける**（JOIN 先には付けない）。
  どのテーブルを起点にしたかは必ずコメントで残す
- sample_annotations と annotation_edits をマージする経路（一覧・エクスポート）は
  **両テーブルとも** dataset_id で絞る。片方だけだと他データセットの編集差分が混入する
- `app/services/annotation_merger.py` だけは例外で dataset_id を取らない（名前空間化済みの
  単一 token 起点の引き当てのみで、呼び出し元が必ずスコープ済み）

### basemap 画像の配信
- 配信元は **`map_meta.basemap_path`**（DB）。エンドポイント側にファイル名を持たない
- 投入時に `app/lib/basemap.py` の `resolve_basemap_path()` が
  `maps/{location}.png` → nuScenes 標準 4 ロケーションの対応表 → `maps/basemap/{location}.png`
  の順に**実在するファイル**を探して決める（無ければ NULL）
- 配信時は location の正規表現チェック（400）に加え、DB 由来のパスも
  `assert_within_dataroot()` でデータルート外を弾く
- 既存 map_set の backfill はマイグレーション `a1c74f9be2d0`（純 SQL・冪等）で行う

## Frontend
- 描画:      Deck.gl 9.x
- UI:        React 19 + TypeScript 5.x + Vite 6.x
- スタイル:  Tailwind CSS 4.x + shadcn/ui
- 状態管理:  Zustand 5.x
- API通信:   TanStack Query（@tanstack/react-query）5.x
- フォーム:  React Hook Form 7.x + Zod 3.x（アノテーション編集部分）
- テスト:    Vitest 3.x

### 型定義
- `src/types/` がフロントエンドの唯一の型定義
- バックエンドの `schemas/` と1対1で対応させる
- Claude Codeは型を勝手に作らず必ず `src/types/` を参照する

### 状態管理
- サーバーデータ（APIレスポンス）→ TanStack Query で管理
- UIの選択状態・表示設定 → Zustand で管理
- ローカルのフォームstate → React Hook Form で管理
- この3つを混在させない

### Deck.glレイヤー
- レイヤー定義は `src/layers/` に集約する
- コンポーネント内にレイヤー定義を直接書かない

### APIアクセス
- fetch は必ず `src/api/client.ts` の `apiFetch` を経由する
- コンポーネントから直接 fetch を呼ばない
- **URL への `/datasets/{id}` の差し込みは `apiFetch` / `apiUrl` に一元化**する
  （呼び出し側は `/scenes?limit=50` のようなリソース相対パスだけを書く）。
  バイナリ配信（画像 / basemap / export ZIP）だけは生 fetch が必要なので、
  URL 組み立てに `apiUrl()` を使う
- **全 queryKey の第2要素に datasetId を入れる**（`['scenes', datasetId, ...]`）。
  入れ忘れるとデータセット切り替え時に前のデータが表示される。特に
  `staleTime: Infinity` のもの（sensor-image / calibrated-sensors / basemap 等）は必須。
  無効化（`invalidateQueries`）のキーも同じ並びに合わせる
- 保険として Header のデータセット切り替え時に `queryClient.removeQueries()` を実行する

## Database
- Engine: PostgreSQL 16 + PostGIS 3.4
- ORM: SQLAlchemy 2.x + GeoAlchemy2
- Migration: Alembic

### ジオメトリ型のルール
- DBカラム型: GeoAlchemy2の `Geometry` 型を使用
  - Point    → `Geometry('POINT', srid=4326)`
  - LineString → `Geometry('LINESTRING', srid=4326)`
  - Polygon  → `Geometry('POLYGON', srid=4326)`
- SRID: 常に4326（WGS84）を使用
- API入出力: 常にGeoJSON形式（`{"type": "Point", "coordinates": [...]}` 等）
- GeoJSON ↔ PostGIS変換: **geoalchemy2.shape と shapely を使用**
  - 変換ロジックは `app/converters/geometry.py` に集約する
  - RouterやCRUDに変換コードを直接書かない

### geometry.pyの変換パターン（参考実装）
```python
# GeoJSON dict → WKBElement（DB保存時）
from geoalchemy2.shape import from_shape
from shapely.geometry import shape

def geojson_to_wkb(geojson: dict):
    return from_shape(shape(geojson), srid=4326)

# WKBElement → GeoJSON dict（APIレスポンス時）
from geoalchemy2.shape import to_shape

def wkb_to_geojson(wkb) -> dict:
    return to_shape(wkb).__geo_interface__
```

## API Design
### 共通ルール
- prefix: `/api/v1`
- レスポンスは常にPydantic schemaを通す
- ジオメトリフィールドはGeoJSON形式で返す
- エラーは `{"detail": "..."}`形式で返す

### エンドポイント構成
実装済みリソース: **datasets / map-sets / scenes / samples / annotations / sensors / maps /
categories / attributes / visibilities / instances / logs / export**

**`/datasets` と `/map-sets` 以外の全エンドポイントは URL に dataset を含む**
（`/api/v1/datasets/{dataset_id}/...`）。dataset を含まない URL はルートが存在せず
404 / 不明な ID も 404。`POST /scenes/import` も URL で投入先が決まる（Form には入れない）。

**新規エンドポイントを追加するときは、デコレータにプレフィックスを書かない。**
`app/api/v1/router.py` の `DATASET_SCOPED_PREFIX` を付けて `include_router` する
（デコレータ側に書くと付け忘れが起き、しかも FastAPI は起動時に検査しないので
実行時 422 になるまで気付けない）。スコープ外に置くのは
「まだ選択していないデータセットを参照する」エンドポイントだけで、
`tests/integration/test_api_route_scope.py` の許可リストに理由付きで追加する。

主要エンドポイント一覧（`/datasets` `/map-sets` 以外は `/api/v1/datasets/{dataset_id}` 配下）:

| Method | Path | 用途 |
|--------|------|------|
| GET | `/api/v1/datasets` | データセット一覧（map_set_name 付き。dataset_id 不要） |
| GET | `/api/v1/datasets/{id}` | データセット1件（dataset_id 不要） |
| GET | `/api/v1/datasets/{id}/stats` | データセットの統計（件数 / locations / sensor_channels。**スコープ外**） |
| GET | `/api/v1/map-sets` | マップセット一覧（dataset_id 不要） |
| GET | `/api/v1/datasets/{dataset_id}/scenes` | シーン一覧（limit/offset） |
| GET | `/api/v1/datasets/{dataset_id}/scenes/{token}` | シーン1件 |
| GET | `/api/v1/datasets/{dataset_id}/scenes/{token}/samples` | シーン内サンプル一覧 |
| GET | `/api/v1/datasets/{dataset_id}/scenes/{token}/ego-poses` | シーン内全 Ego Pose |
| GET | `/api/v1/datasets/{dataset_id}/samples/{token}` | サンプル1件 |
| GET | `/api/v1/datasets/{dataset_id}/samples/{token}/annotations` | サンプルのアノテーション一覧 |
| GET | `/api/v1/datasets/{dataset_id}/samples/{token}/sensor-data` | サンプルのセンサーデータマップ（channel→SensorDataBrief） |
| GET | `/api/v1/datasets/{dataset_id}/samples/{token}/instances` | サンプル内インスタンスサマリ一覧 |
| GET | `/api/v1/datasets/{dataset_id}/annotations` | アノテーション一覧（limit/offset） |
| GET | `/api/v1/datasets/{dataset_id}/annotations/{token}` | アノテーション1件 |
| PATCH | `/api/v1/datasets/{dataset_id}/annotations/{token}` | アノテーション部分更新 |
| GET | `/api/v1/datasets/{dataset_id}/calibrated-sensors` | キャリブレーション済みセンサー一覧 |
| GET | `/api/v1/datasets/{dataset_id}/sensor-data/{token}/image` | センサー画像バイナリ配信 |
| GET | `/api/v1/datasets/{dataset_id}/sensor-data/{token}/pointcloud` | 点群 JSON 配信 |
| GET | `/api/v1/datasets/{dataset_id}/maps` | マップ一覧（canvas_edge と GPS 原点 origin_lat/lon を含む） |
| GET | `/api/v1/datasets/{dataset_id}/maps/{token}/geojson` | マップ GeoJSON |
| GET | `/api/v1/datasets/{dataset_id}/maps/{location}/basemap` | ベースマップ画像バイナリ配信（map_meta.basemap_path 経由） |
| GET | `/api/v1/datasets/{dataset_id}/categories` | カテゴリ一覧（全件・ページネーションなし） |
| GET | `/api/v1/datasets/{dataset_id}/attributes` | 属性一覧 |
| GET | `/api/v1/datasets/{dataset_id}/visibilities` | 可視性レベル一覧 |
| GET | `/api/v1/datasets/{dataset_id}/instances` | インスタンス一覧（scene_token/category_name フィルタ対応） |
| GET | `/api/v1/datasets/{dataset_id}/instances/{token}/annotations` | インスタンスの全アノテーション（timestamp 昇順） |
| GET | `/api/v1/datasets/{dataset_id}/instances/{token}/best-camera` | インスタンスが最もよく写るカメラチャンネルと sample_data_token |
| GET | `/api/v1/datasets/{dataset_id}/logs` | ログ一覧 |
| GET | `/api/v1/datasets/{dataset_id}/export/nuscenes` | 指定データセット全体の nuScenes 形式 ZIP（`token_format` 対応） |
| GET | `/api/v1/datasets/{dataset_id}/export/nuscenes/{scene_token}` | 単一シーンの nuScenes 形式 ZIP（`token_format` 対応） |

フルCRUD（POST/PUT/DELETE）は現時点では annotations と scenes（import/delete）のみ実装。
将来的に POST/PUT/DELETE を追加する場合は各エンドポイントファイルに追記すること。

### ページネーション
```python
# 全GETリストエンドポイントで共通化（dataset は URL の階層で指定）
GET /api/v1/datasets/<uuid>/scenes?limit=50&offset=0
```

### LiDAR点群の形式
センサーデータ（LiDAR点群）はPotree形式に変換せず`.pcd.bin`バイナリ直接配信でよい
- フォーマット: float32 × 5列（x, y, z, intensity, ring_index）
- DBの fileformat カラム値: `pcd`
- APIレスポンス: JSON形式 `{"points": [[x,y,z,intensity], ...], "num_points": N}`

## Docker構成
### コンテナ一覧
プロジェクトルートの `docker-compose.yml` を参照。

### 環境変数（backend）
プロジェクトルートの `.env` を参照。

### 起動コマンド
```bash
make dev      # docker compose up --build
make migrate  # alembic upgrade head
make test     # pytest + vitest
```

### データ投入コマンド
マップ（map_set）を先に作り、データセットからそれを参照する。
```bash
# 1) Map expansion を map_set として投入（4ロケーション）
docker compose run --rm api python scripts/import_nuscenes_map.py \
  --map-set-name nuscenes-map-v1.3 --map-set-version 1.3

# 2) NuScenes 本体を dataset として投入（マップは投入せず map_set を参照するだけ）
docker compose run --rm api python scripts/import_nuscenes.py \
  --dataset-name nuscenes-trainval --dataset-version v1.0-trainval \
  --map-set-name nuscenes-map-v1.3

# 同名が既にある場合は --overwrite で削除→再投入（配下データも消える）

# 3) マップなしでデータセットを投入する（--map-set-name を省略 → map_set_id = NULL）
docker compose run --rm api python scripts/import_nuscenes.py \
  --dataset-name my-dataset --dataset-version v1.0-mini

# 4) 後からマップを紐付ける／解除する
docker compose run --rm api python scripts/link_map_set.py \
  --dataset-name my-dataset --map-set-name nuscenes-map-v1.3
docker compose run --rm api python scripts/link_map_set.py --dataset-name my-dataset --detach
#   別の map_set が既に紐付いている場合は --force、
#   dataset の location が map_set に足りない場合は --allow-missing-locations（既定はエラー）

# マップ投入と同時に紐付けることもできる
docker compose run --rm api python scripts/import_nuscenes_map.py \
  --map-set-name nuscenes-map-v1.3 --attach-to-dataset my-dataset
```

basemap 画像は map_meta.basemap_path から配信する。マップ投入時に自動で解決されるが、
このカラムが入る前に投入した map_set はマイグレーション `a1c74f9be2d0` が backfill する
（`make migrate` で適用される）。

## 実装上の制約
- SQLAlchemy 2.xの `Session` は `Annotated` + `Depends` でDI
- 非同期（async/await）を使用する（ドライバ: asyncpg）
- CORSは開発時 `*` 許可、本番は環境変数で制御
- テストDBは別コンテナ（postgresのみ、PostGIS不要）ではなく同一イメージを使う
- NuScenesのデータパスは環境変数 NUSCENES_DATAROOT で渡す
- map expansionのレイヤー（drivable_area, lane等）はGeoJSON形式でフロントに渡す

## 行動原則
- 3ステップ以上のタスクは必ずPlanモードで開始する
- コードを読まずに書かない。必ず既存コードを確認してから変更する

## よくある実装ミスの禁止事項
- RouterにDB変換ロジックを書かない → app/service/ と app/converters/ に集約
- Pydanticモデルをmodels/と独立して定義しない → schemas/はmodels/から派生
- ジオメトリをWKBのままAPIレスポンスに含めない → 必ずGeoJSONに変換
- 新リソース追加時は endpoints/ + service/ + repository/ + schemas/ をセットで追加する
