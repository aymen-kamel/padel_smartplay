# Experiment inventory

Date: 2026-11-13. Read-only. Tags: **MEASURED**, **CODE**, **UNVERIFIED**. Aliases: `MAIN` = `padel_analytics_repo/padel_analytics_repo/`, `BO` = `ball_only/ball_only/`.
"Valid evidence?" asks one question: does this experiment contain a measurement that can be compared with ground truth? **No experiment below has ground truth**, so none is valid evidence of accuracy. The column says what it *is* good for.

Clip facts (container headers, MEASURED): `rally.mp4` = 1920×1080, **25 fps**, 324 frames, 12.96 s, with audio (47 identical-size copies). `rally_001.mp4` = 1920×1080, 30 fps, 1,116 frames, 37.2 s, with audio. `highlights.mp4` = 1162×640, 30 fps, 1,244 frames, 41.6 s. `match3.mp4` = 1306×666, 14.2 s, 421 frames.

| ID | Date | Experiment | Script / entry | Input | Artefacts on disk | What was measured | Valid evidence? |
|---|---|---|---|---|---|---|---|
| X01 | 2026-03-27/28 | YOLO ball detector, raw vs robust-filtered vs tracked | `BO/src/yolomain.py` | `rally.mp4` | `BO/output/output_yolo_{raw_yolo,robust_filtered,tracking}.mp4`, `output_yolo_trajectory*.csv` (3) | Nothing quantitative except X06 | Visual only |
| X02 | 03-27 → 03-31 | TrackNet ("TrackNetFlat") with geometry filter | `BO/src/run_tracknet.py` | `match3.mp4` (1306×666) | `output_detection.mp4`, `output_rendered.mp4`, `output_geofilter.mp4` | Nothing | Visual only |
| X03 | 03-27 → 04-10 | Rally detection and FSM scoring overlay | `BO/src/run_rally_score.py`, `run_rally_score_yolo.py`, `simple_score.py` | `rally.mp4` | `output_rally_detection.mp4`, `output_rally_score.mp4` (04-10), `output_rally_score_parfait.mp4` (03-28; "parfait" = "perfect") | Final score printed to console only; not saved. No evidence the score was correct | No |
| X04 | 04-01/02 | Bounce classification, 3D camera model, glass planes | `BO/src/analyze_bounces.py`, `bounce_classifier.py`, `calibrate_court_3d.py`, `court_homography.py` | `rally.mp4` + `config/calibration_3d.json` | `output_bounce_analysis.mp4`, `output_3d_arc.mp4`, `court_3d.png`, `interactive_padel_3d.html`, `simulated_rally_3d.mp4` (800×800, **synthetic**) | Nothing; README references `src/plot_3d_interactive.py`, which does not exist | No |
| X05 | 04-14 | Robust TrackNet tracking + bounce + phase labels | `BO/src/main.py:319` (`highlights.mp4`) | `highlights.mp4` | `output_robust_tracking.mp4`, `output_trajectory.csv` (1,244 rows: zone, phase, court coords, bounce flag) | Nothing | Visual only |
| X06 | 03-28 → 04-02 | Benchmark of 3 ball pipelines | `BO/src/benchmark_all.py` | `rally.mp4` | `BO/output/benchmark_results.csv` | FPS and "frames with output" (not accuracy): YOLO 61.9 FPS / 30.0 %; TrackNet 4.15 / 65.4 %; +Inpaint 3.95 / 77.5 % | **Partly**: speed is real; "detection rate" is not accuracy. Report table differs (U15) |
| X07 | 05-09 → 05-11 | Compare two TrackNet checkpoints | `BO/scratch/{compare_models,compare_inference,model_defs}.py` | `rally.mp4` | `BO/output/comparison_models.mp4` (3840×1080 side by side), `model_layers.txt`, `full_keys.txt`, `models/ckpt_summary.txt` | Layer lists only | No |
| X08 | 03-31 | Padex shot-detection demo | `padex/scripts/run_shot_detection_test.py`, `run_padex*.py` | `rally.mp4` | `output/2026033*/shot_detection.mp4` (3, one 1306×666), `padex/output/M_001_{analytics.json,report.html}`, `rally_tracking_cache.pkl`, `rally_calibration.json` | One point of rally length 12; player `P_011`: 6 of 7 shots typed `unknown` | No |
| X09 | 05-04 → 05-06 | Shot-feature collection, manual labelling, RandomForest | `MAIN/collect_shot_data.py`, `annotate_shots.py`, `train_shot_classifier.py`; `MAIN/evaluate_pipeline.py` | rally clips (which one: UNVERIFIED) | `shot_features*.csv` (MAIN 05-04, root 05-06), `weights/shot_classifier.pkl` (2 versions), `output_with_ml_shots.mp4`, `shot_collection_debug.mp4`, `rapport_evaluation.json` | n = 39, 4-fold CV 71.8 % vs 59.0 % baseline; smash n = 4 | **Weak**: tiny set, shuffled CV on correlated windows (see DATASET_INVENTORY) |
| X10 | — | LSTM shot classifier | `MAIN/train_lstm_classifier.py` | `shot_features_labeled.csv` ("advanced" mode) | **none**: `weights/lstm_shot_classifier.pth`, `lstm_scaler.pkl` absent | Nothing | No (never completed, or outputs lost) |
| X11 | 05-15 → 06-09 | 19 API jobs through FastAPI + React UI | `MAIN/api_server.py` | `rally.mp4` (all 19) | `MAIN/results_json/<uuid>/` ×19 (output.mp4, data.csv, shot_events.json, summary.json, job_meta.json, first_frame.jpg), `uploads/` ×19, `storage/` ×11 | Timings: 4.2–7.3 min (11 jobs), 18–38 min (7 jobs), 1 failed (RAM). Same input, so no diversity | Timing only |
| X12 | 08-07/08 | Full 37 s run of the CLI pipeline | `MAIN/main.py` → `run_analysis.py` | `input/rally_001.mp4` | `results_new.mp4`, `data_new.csv` (1,116 rows), `summary.json` (0–0), `cache/*.json`, `match_stats_table.tex`, `rapport_evaluation.json` | Cache statistics (S5), telemetry anomalies (C13) | Diagnostic only |
| X13 | 04-14 | Static-appearance player re-ID | `MAIN/trackers/players_tracker/players_tracker.py`; notes `today_review_2026_04_14.md`, `player_tracker_documentation.md` | `rally.mp4` | none | No IDF1/HOTA, no ID-switch count | No |
| X14 | 05-30 | CPU/GPU latency benchmarks | `benchmark_performance.py` (root), `MAIN/benchmark_pipeline.py` | `rally_001.mp4` | **none**: both scripts write `benchmark_results.json`, which does not exist | Nothing on disk | No (artefact missing; see U09) |
| X15 | 05-12/13 | Rally extraction from broadcast footage | `the_best/paddle_rally_extractor.py` | `the_best/vedio.mp4` (446 s) | `the_best/rallie/` (5 clips 37–118 s), `rallies/` (7 clips 4–15 s, one 262-byte broken file), compilations (346 s, 397 s), `rallies_output.json` (other schema, no producing script) | Clip durations only | No. 37–118 s clips cannot be single rallies |
| X16 | 08-07 | Evaluation harness | `MAIN/evaluate_pipeline.py` | saved CSV/JSON | `rapport_evaluation.json` | See X09; hit detection `evaluable: false` | Yes, as a method |
| X17 | 06 → 08 | Match KPI report | `MAIN/analytics/match_report.py`, `extract_latex_stats.py` | `data_new.csv`, `summary.json` | `match_stats_table.tex` | Distance/speed KPIs; inconsistent (C13) | No |

Out of scope: the basketball work in `iit_courtvison/` (Nov 2026).

## What the experiments collectively show
- All ball-tracking and scoring experiments use **one 13-second clip** (`rally.mp4`, apparently the upstream project's sample video — it is also in `MAIN/examples/videos/`, dated 2026-03-27; UNVERIFIED), plus short snippets. No experiment spans a full point sequence, a game or a match.
- Output videos exist for visual inspection; **no experiment stores a score next to a ground-truth score**.
- Parameters were retuned between runs (bounce count on the same clip: 5, 6, 7) without a metric to choose between settings.

## UNVERIFIED
- Which video produced `shot_features*.csv` and which of the two label files trained each pickle.
- Whether `output_rally_score_parfait.mp4` shows a correct score (not viewed).
- Whether `futuristic_rally.mp4` (800×800, 324 frames, no audio) is synthetic like `simulated_rally_3d.mp4` (same format).
- What produced `rallies_output.json`.
