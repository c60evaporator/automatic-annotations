"""Box Fitting の結果（補正・距離フィルタなど）を track_id 単位で確認する.

使い方（webapp フォルダで）:
    docker compose run --rm -T webapp python - < check_fit.py

TRACK_ID / RUN_ID / SAMPLE_TOKEN を書き換えて使う。
"""
import json
import sqlite3

TRACK_ID = "0"       # 調べたい track_id（カメラ内の ID）。None なら全トラック
RUN_ID = None        # None なら最新の run。特定の run を見るなら id を文字列で
SAMPLE_TOKEN = None  # 画面右上に出ている sample_token で絞り込む（前方一致可）。None なら全 sample

con = sqlite3.connect("/db/app.db")
run = RUN_ID or con.execute(
    "SELECT id FROM depth_estimation_params ORDER BY started_at DESC LIMIT 1"
).fetchone()[0]
print("run:", run)

where = ["b.depth_estimation_params_id = ?"]
args: list = [run]
if TRACK_ID is not None:
    where.append("b.track_id = ?")
    args.append(TRACK_ID)
if SAMPLE_TOKEN:
    where.append("sd.sample_token LIKE ?")
    args.append(SAMPLE_TOKEN + "%")

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
    ORDER BY sd.timestamp, s.channel
    """,
    args,
).fetchall()

for r in rows:
    (sample, sd, channel, tid, label, status, gid, depth_used,
     nd, ndk, nl, nlk, corr, dist, lsh) = r
    print("=" * 80)
    print(f"sample={sample}  sample_data={sd[:8]}  {channel}  track_id={tid}")
    print(f"  label={label} status={status} global={gid} depth_used={bool(depth_used)}")
    print(f"  depth raw/kept={nd}/{ndk}  lidar raw/kept={nl}/{nlk}")
    print("  depth_correction:", json.loads(corr) if corr else None)
    print("  distance_filter :", json.loads(dist) if dist else None)
    print("  lshape_filter   :", "あり" if lsh else None)
print(f"\n{len(rows)} 行")
