# db/ + Alembic セットアップメモ

## ファイル配置

```
webapp/
├── alembic.ini                 ← 本メモの alembic.ini
├── alembic/
│   ├── env.py                  ← 本メモの alembic/env.py
│   ├── script.py.mako          ← 本メモの alembic/script.py.mako
│   └── versions/
│       └── 20260830_1408_2137e2545219_init_schema.py
└── app/
    ├── core/
    │   ├── config.py           ← core/config.py
    │   └── logging.py          ← core/logging.py
    ├── db/
    │   ├── base.py             ← db/base.py
    │   ├── engine.py           ← db/engine.py
    │   └── session.py          ← db/session.py
    └── models/                 ← 前回作成済み
```

`alembic.ini` の `prepend_sys_path = .` は `webapp/` を指す。
alembic コマンドは `webapp/` ディレクトリで実行すること。

`__init__.py` が必要なディレクトリ: `app/`, `app/core/`, `app/db/`
（`app/models/__init__.py` は作成済み）

## .env の例

```
LOG_LEVEL=INFO

# ホスト側の nuScenes データ配置先（docker-compose でマウント）
HOST_DATA_ROOT=/path/to/nuscenes

# コンテナ内のパス
DATA_ROOT=/data
DERIVED_ROOT=/data/_derived
SQLITE_PATH=/db/app.db

SQL_ECHO=False
SQLITE_BUSY_TIMEOUT_MS=10000

INFERENCE_BASE_URL=http://inference:8000
INFERENCE_TIMEOUT_SEC=1800
```

## docker-compose での注意

SQLite ファイルは名前付きボリュームに置く。バインドマウントだと
ホスト OS によっては WAL のロックが正しく効かない。

```yaml
services:
  webapp:
    volumes:
      - ${HOST_DATA_ROOT}:/data:ro     # データセットは読み取り専用
      - derived:/data/_derived         # 深度マップ等の派生物は書き込み可
      - dbdata:/db                     # SQLite
volumes:
  derived:
  dbdata:
```

`DATA_ROOT` を `:ro` にしておくと、元データセットを壊す事故を防げる。
派生物の書き込み先 `DERIVED_ROOT` だけ別ボリュームで書き込み可にする。

## 常用コマンド

```bash
cd webapp

# モデル変更後にマイグレーションを生成
alembic revision --autogenerate -m "add xxx"

# 適用
alembic upgrade head

# モデルと DB に差分が無いか確認（CI に入れると事故が減る）
alembic check

# 1つ戻す
alembic downgrade -1

# 一時 DB を対象にする（テスト用）
alembic -x db_url=sqlite:///./tmp.db upgrade head
```

## 検証済みの動作

- `alembic revision --autogenerate` → 21 テーブル / 63 インデックス生成
- `alembic upgrade head` 後の `alembic check` が差分なし
- カラム追加 → `batch_alter_table` に展開されて upgrade / downgrade 成功
- batch によるテーブル再作成後も、FK 7 本・index 9 本・`ondelete` 指定が保持
- 再作成後も `DELETE FROM datasets` の CASCADE が全テーブルに伝播
- PRAGMA: `foreign_keys=1`, `journal_mode=wal`, `synchronous=1`, `busy_timeout=10000`
- `expire_on_commit=False` により commit 後も属性アクセス可能
