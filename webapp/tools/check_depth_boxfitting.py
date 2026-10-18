"""Depth Estimation & Box Fitting の結果（深度補正・距離フィルタなど）を確認する.

DB（SQLite）を直接読み、box_fittings の行をインスタンス × フレーム単位で出す。
webapp コンテナの中で動かす（DB はコンテナ内の SQLITE_PATH にある）。

使い方（docker-compose.yml のある webapp フォルダで）:

    docker compose run --rm -T webapp python - [オプション] < tools/check_depth_boxfitting.py

  ``python -`` と ``-T`` で、ホスト側のこのファイルを標準入力で渡す
  （コンテナへのマウントは不要）。オプションは ``-`` の後ろに書く。

オプション（いずれも省略可）:
  --run RUN_ID        Depth Boxfitting の run。省略すると最新の run。
                      画面の「保存済みの推論結果」の各行末の run=... の値
  --sample TOKEN      sample_token（前方一致）。画面右上の sample=... の値
  --track TRACK_ID    track_id（カメラ内の ID）。カメラをまたぐと同じ値が別物体になる
  --global-track ID   global_track_id（カメラ間結合後の ID）
  --label LABEL       ラベル
  --db PATH           DB のパス（既定は環境変数 SQLITE_PATH、無ければ /db/app.db）

例:
    # 最新 run の、ある sample の全インスタンス
    docker compose run --rm -T webapp python - --sample bb4f4d9a < tools/check_depth_boxfitting.py
    # 特定 run の track_id=0
    docker compose run --rm -T webapp python - --run 8e8b1282-... --track 0 < tools/check_depth_boxfitting.py
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="check_depth_boxfitting",
        description="Depth Boxfitting の結果を track / sample 単位で確認する",
    )
    parser.add_argument("--run", help="Depth Boxfitting の run id（省略で最新）")
    parser.add_argument("--sample", help="sample_token（前方一致）")
    parser.add_argument("--track", help="track_id（カメラ内の ID）")
    parser.add_argument("--global-track", dest="global_track", help="global_track_id")
    parser.add_argument("--label", help="ラベル")
    parser.add_argument(
        "--db", default=os.environ.get("SQLITE_PATH", "/db/app.db"),
        help="SQLite のパス",
    )
    return parser.parse_args()


def _json(value):
    return json.loads(value) if value else None


def main() -> None:
    args = _parse_args()
    con = sqlite3.connect(args.db)

    if args.run:
        run = args.run
    else:
        row = con.execute(
            "SELECT id FROM depth_estimation_params ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            print("Depth Boxfitting の run がありません")
            return
        run = row[0]
    print("run:", run)

    where = ["b.depth_estimation_params_id = ?"]
    params: list = [run]
    if args.sample:
        where.append("sd.sample_token LIKE ?")
        params.append(args.sample + "%")
    if args.track is not None:
        where.append("b.track_id = ?")
        params.append(str(args.track))
    if args.global_track is not None:
        where.append("b.global_track_id = ?")
        params.append(str(args.global_track))
    if args.label:
        where.append("b.label = ?")
        params.append(args.label)

    rows = con.execute(
        f"""
        SELECT sd.sample_token, b.sample_data_token, s.channel, b.track_id,
               b.label, b.status, b.global_track_id, b.depth_used,
               b.num_points_depth, b.num_points_depth_kept,
               b.num_points_lidar, b.num_points_lidar_kept,
               b.depth_correction, b.distance_filter, b.lshape_filter
        FROM box_fittings AS b
        JOIN sample_data AS sd ON sd.token = b.sample_data_token
        LEFT JOIN calibrated_sensors AS cs ON cs.token = sd.calibrated_sensor_token
        LEFT JOIN sensors AS s ON s.token = cs.sensor_token
        WHERE {" AND ".join(where)}
        ORDER BY sd.timestamp, s.channel, b.track_id
        """,
        params,
    ).fetchall()

    for r in rows:
        (sample, sd, channel, tid, label, status, gid, depth_used,
         nd, ndk, nl, nlk, corr, dist, lsh) = r
        print("=" * 80)
        print(f"sample={sample}  sample_data={sd[:8]}  {channel}  track_id={tid}")
        print(f"  label={label} status={status} global={gid} "
              f"depth_used={bool(depth_used)}")
        print(f"  depth raw/kept={nd}/{ndk}  lidar raw/kept={nl}/{nlk}")
        print("  depth_correction:", _json(corr))
        print("  distance_filter :", _json(dist))
        print("  lshape_filter   :", "あり" if lsh else None)
    print(f"\n{len(rows)} 行")


if __name__ == "__main__":
    main()
