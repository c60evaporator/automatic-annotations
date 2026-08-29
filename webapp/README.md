## Prerequisites

以下を準備してください

- Docker & Docker Compose
- Chrome browser
- nuScenes formatted dataset

## Getting Started

### 1. リポジトリクローン

```bash
git clone nuscenes-cutter.git
```

### 2. データセットを準備

以下のように`data`フォルダ直下に、シーンを切り出したいnuScenes形式のデータセットを設置します。

```
root/
├── webapp
|   ├── docker-compose.yml
:   :
└── data
    ├─ input
    |   ├─ .gitkeep
    |   └─ <dataset_name> <- ここにnuScenes形式のデータセットを設置する
    |       ├─ <version_name>
    |       ├─ samples
    |       ├─ sweeps
    |       └─ maps
    |           └─ <map_token>.png <- ファイル名がそのMapのmap_tokenと一致しているように注意
    └─ output
        └─ .gitkeep
```
