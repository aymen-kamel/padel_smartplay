# State of the art (Phase 2)

Date: 2026-11-14. Web research only: nothing was downloaded, installed or run.
Order follows the weak links found in `docs/audit/SCORE_DIAGNOSIS.md`: events → rallies/rules → ball → court → players → shots → data.

**Evidence labels**
- **[V]** = I opened the primary source (paper page, official repo, official rulebook) and read the fact there.
- **[S]** = secondary source (search snippet, aggregator, blog). Treat as a lead, not a fact.
- **UNVERIFIED** = not confirmed anywhere I could open.

All accuracy numbers are **self-reported by the authors on their own data**. They are not comparable with each other or with padel. The only numbers that will count are the ones measured on your own fixed-camera footage in Phase 3/4.

Context fixed by the user (2026-11-14): one fixed camera, after-match processing assumed, v1 = points → full score, clean rewrite under the user's own licence, 4 GB VRAM design budget (machine has 6 GB).

---

## 0. The rules we must implement (FIP, verified)

Source: *FIP Rules of Padel*, review applicable 01.01.2026, https://www.padelfip.com/wp-content/uploads/2025/12/FIP_Rules-of-Padel.pdf **[V]** (pages read directly).

| Topic | Rule (paraphrased) | Why it matters for us |
|---|---|---|
| Ball | Diameter 6.35–6.77 cm (The Ball, §1) | ~5–15 px in a 1080p wide fixed view (estimate, UNVERIFIED) — a tiny-object problem. Confirms the report's "40 mm" (U12) is wrong. |
| Game scoring | Three official options: **Advantage**, **Star Point** (deuce 1/2/3, then a deciding "Star Point"), **Golden Point** (one deciding point at 40-40, receiving pair chooses side) (Rule 1) | The score engine must be configurable per match. The current code supports none of these fully. |
| Set | First to 6 games with a 2-game margin; 7-5 possible; tie-break at 6-6 (Rule 1) | |
| Tie-break | First to 7 points with a 2-point margin; serve order 1-2-2…; set won 7-6 (Rule 1 Tie-break) | |
| Alternatives | Mini set (4 games), match tie-break (7), super tie-break (10) (Rule 1 Alternative) | Config options. |
| Changes of ends | After games 1, 3, 5 … of each set; in tie-breaks after every 6 points (Rule 5) | **Strong anchor**: in a fixed-camera video, players visibly swap sides → game boundaries can be cross-checked. Teams must be tracked by identity, not by side. |
| Time | Max 20 s between points; 90 s at changes of ends (Rule 2) | Pauses between points are short but real → usable for point segmentation. |
| Serve | Bounce then hit at or below waist from behind the service line, diagonal into the receiver's box; two attempts (Rules 6–7) | A serve is a visually distinctive start-of-point event. |
| Ball in play | After a valid bounce on the opponent's ground the ball may hit walls/fence and must be returned before the second bounce (Rule 12.3) | Wall contacts are **legal**; only the floor bounce count matters. |
| Point lost | e.g. second bounce before return (13.1.c); return hits the opponent's walls/fence **before** bouncing (13.1.g); ball out over the enclosure after bouncing (13.1.d); net; double hit; ball hits own fence/ground after own hit; two serve faults (Rule 13) | The current FSM only covers 13.1.c. v1 needs at least: second bounce, direct-to-wall/out, net/fault — or a learned "who won" decision. |

---

## 1. Hit and bounce event detection (weakest link)

| Approach | What it is | Links | Licence | Reported results | Fit / remarks |
|---|---|---|---|---|---|
| **Audio hit detection for padel** (Decorte et al., CVPRW 2024) | Detects racket–ball hits from the match audio; also assigns hits to players/teams | Paper: https://openaccess.thecvf.com/content/CVPR2024W/CVsports/html/Decorte_Multi-Modal_Hit_Detection_and_Positional_Analysis_in_Padel_Competitions_CVPRW_2024_paper.html; abstract **[V]** at https://mlanthology.org/cvprw/2024/decorte2024cvprw-multimodal | Paper only; **no code or dataset link found** | Avg F1 **92 %** hit detection; player assignment 83.7 %, team 86.8 % **[V abstract]**; dataset ≈ 5 h 28 min incl. highlight videos and 2 full matches **[S]** | **Most relevant padel result.** Audio is cheap (CPU). Your clips have audio tracks (DATASET_INVENTORY). Must re-implement. |
| Classical audio onset/energy | High-pass filter + energy peaks or onset detection | e.g. table-tennis bounce audio https://arxiv.org/html/2409.11760v2 **[S]**; padel/tennis onset study (Spanish acoustics congress 2022) **[S]** | Method only | not comparable | Good **first baseline**: no training, minutes to build. |
| Trajectory bounce classifier (tennis) | Gradient-boosted trees over 16 scale-free trajectory features (kink, curvature, speed ratio, direction change, court position) on TrackNet output | https://huggingface.co/kjfk/tennis-bounce-detector **[V]** | **MIT** **[V]** | Precision 0.888, recall 0.944, F1 0.915 on held-out tennis clips; 920 labelled bounces, 10 matches **[V]** | Shows the right formulation: **"a bounce is a corner in the trajectory, not a peak"** (vs. current image-y peak). Tennis broadcast; retrain on padel. |
| Precise event spotting (video) — E2E-Spot | End-to-end frame-accurate spotting (CNN + GRU) incl. tennis hits | Code https://github.com/jhong93/spot; paper https://arxiv.org/abs/2207.10213 **[S]** | **UNVERIFIED** (fork says BSD-3) | tennis mAP ≈ 96.9 (δ=1) **[S aggregator]** | Strong but needs labelled frames and GPU training (likely > 4 GB at their settings — UNVERIFIED). Later option. |
| Pose + ball graph spotting — UMEG-Net (2025) | Few-shot spotting from skeletons and ball keypoints | https://github.com/LZYAndy/UMEG-Net **[S]** | UNVERIFIED | — | Research option once pose and ball are solid. |
| **Current implementation** | Bounce = local max of image y; hit = velocity change + pixel distance | `MAIN/trackers/ball_tracker/ball_tracker.py:17-65`, `analytics/shot_counter.py` | own/upstream | never measured | Cannot separate floor/wall/racket (audit). |

**Takeaway**: combine **audio for hits** (padel evidence: 92 % F1) with a **trajectory-corner bounce classifier** made height-aware by the 3D court model. Video event spotting is a later upgrade.

## 2. Rally / point segmentation and score inference

| Approach | Links | Licence | Notes |
|---|---|---|---|
| Pause-based boundaries (gap in ball activity ≥ 3 s) | Padex `events/point.py` (GPL-3.0, local) | GPL — idea only | Weak alone: misses short pauses, splits long lobs. FIP's 20 s max between points makes pauses a useful *feature*. |
| Audio hit density / silence | Decorte et al. (above); tennis broadcast rally splitting from audio-only impacts (arXiv 2603.13397) **[S]** | — | Rally = run of hits; point end = silence + no ball. |
| Single-camera rally segmentation + score inference — *Vollytics* (beach volleyball, MDPI Electronics 15(21):5114) | https://www.mdpi.com/2079-9292/15/21/5114 (**403 when opened**, content UNVERIFIED) | UNVERIFIED | Closest published analogue to "points → score" from one camera **[S]**. |
| Score from on-screen scoreboard — *SmartTennisTV* | https://arxiv.org/pdf/1801.01430 **[S]** | — | Reads the broadcast scorecard and smooths it over time. Only applies to broadcast (e.g. your highlights video), **not** to the fixed-camera product. |
| Score-change-anchored rally detection (table tennis broadcast) | https://dl.acm.org/doi/abs/10.1002/ecj.70019 **[S]** | — | Same idea: use scoreboard changes as labels. |
| **Constrained score decoding** (proposal) | Standard HMM / Viterbi over the padel score automaton | own | Treat each point's winner as a *probability*; decode the most likely full score sequence under FIP constraints (game ends at 4 points, server rotation per game, side change after odd games — Rules 1, 5, 6). Visible side changes and server changes act as anchors that correct isolated errors. **Not found in literature for padel (UNVERIFIED novelty).** |
| **Current implementation** | `MAIN/analytics/scoring.py` | — | One rule, no uncertainty, 0 points awarded in all runs. |

No published padel system doing automatic point-by-point scoring from a fixed camera was found (search on 2026-11-14) — PadelIQ (blog) notes automatic scoring still needs labelled data **[S]**.

## 3. Ball detection and tracking

| Method | Links | Licence | Weights | Reported results | 4 GB fit |
|---|---|---|---|---|---|
| **TrackNetV3** (2023) — heatmap net + trajectory rectification (InpaintNet) | https://github.com/qaz812345/TrackNetV3 **[V]**; paper https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf | **MIT** (README) **[V]** | Yes, Google Drive (badminton) **[V]** | Shuttlecock test: acc 97.51 %, F1 98.56 %, 25.11 FPS **[V]** | Very likely (288×512 input) — measure. Your current tracker is already a TrackNet-family model. |
| **WASB** (BMVC 2023) — HRNet heatmaps, position-aware training, temporal-consistency inference | https://github.com/nttcom/WASB-SBDT **[V]**; https://arxiv.org/abs/2311.05237 | **MIT** **[V]** | Model zoo incl. tennis, badminton, soccer, volleyball, basketball **[S/V partial]** | Beats 6 methods on 5 datasets (paper) **[S]** | Likely; a sibling project measured WASB basketball at 436 MB reserved VRAM for 3×288×512 (`iit_courtvison/environment/PHASE1_RECORD.md`, MEASURED on this machine). Tested on Python 3.8/CUDA 11.3 only. |
| TrackNetV4 (ICASSP 2025) — motion-attention add-on | https://arxiv.org/abs/2409.14543; https://tracknetv4.github.io **[S]** | UNVERIFIED | UNVERIFIED | Improves V2/V3 on tennis & shuttlecock (authors); a third-party reproduction found mixed results **[S]** | Add-on; low cost. |
| TrackNetV5 (arXiv 2512.02789, v4 Jan 2026) — signed motion polarity + transformer refinement | https://arxiv.org/abs/2512.02789 **[S]** | UNVERIFIED | UNVERIFIED | F1 0.9859 on TrackNetV2 data (authors) **[S]** | Code availability UNVERIFIED. |
| TOTNet (arXiv 2508.09650) — occlusion-aware 3D conv | https://arxiv.org/abs/2508.09650 **[S]** | UNVERIFIED | UNVERIFIED | Occluded table-tennis RMSE 37.30 → 7.19 (authors) **[S]** | Occlusion by glass/players is a padel issue — watch. |
| BlurBall (arXiv 2509.18387) — joint ball + blur | **[S]** | UNVERIFIED | — | F1 96.52 % (table tennis) **[S]** | Blur labelling idea useful. |
| YOLO-family ball detector | Ultralytics | **AGPL-3.0** (code and weights, per Ultralytics licence page) **[S]** | — | Your measured: 30.0 % frames (BO benchmark) | Single-frame detectors miss blurred fast balls; licence conflicts with own-licence goal. |
| Physics filtering | Constant-acceleration Kalman with a **coasting limit**; visibility states (visible / occluded / inferred) | — | — | — | Replaces unlimited interpolation (audit TD-06). |

## 4. Court calibration (fixed camera)

| Approach | Links | Licence | Results | Remarks |
|---|---|---|---|---|
| **Manual one-time calibration** (click 4–12 known points, RANSAC homography, show reprojection error) | OpenCV | Apache-2.0 | — | With **one fixed camera** this is done once per installation. Simplest and most accurate for v1. |
| Full camera pose (solvePnP) with known 3D court incl. net height 0.88 m, walls 3–4 m (FIP court specs) | OpenCV; own `BO/src/court_homography.py` | own | — | Needed to tell floor from wall and to undo airborne-ball parallax. |
| Heatmap court keypoints (TrackNet-like, 14 points) — TennisCourtDetector | https://github.com/yastrebksv/TennisCourtDetector **[V]** | **No licence file** **[V]** → default copyright, do not reuse | Precision 0.963, median 1.83 px with refinement (tennis, 8,841 images) **[V]** | Shows the method works; re-implement only if automatic calibration is ever needed. |
| Sports field registration with geometric keypoints + RANSAC/DLT | https://arxiv.org/abs/2404.08401 **[S]** | UNVERIFIED | — | For moving cameras (broadcast) — not v1. |

## 5. Players: detection, tracking, re-ID, pose

| Component | Option | Links | Licence | Reported | 4 GB fit |
|---|---|---|---|---|---|
| Detector | **RF-DETR Nano/Small** | https://github.com/roboflow/rf-detr **[V]** | **Apache-2.0 code + weights for N/S/M/L** **[V]** (other sizes PML 1.0) | COCO AP50:95 48.4 (N), 53.0 (S) **[V]** | Inference yes: a sibling project measured RF-DETR-S at 256 MB reserved VRAM on this machine (MEASURED, `iit_courtvison/environment/PHASE1_RECORD.md`). Fine-tuning VRAM: UNVERIFIED. |
| Detector | RT-DETR / RT-DETRv2 | https://github.com/lyuwenyu/RT-DETR **[V]** | **Apache-2.0** **[V]** | R18 46.5 AP; v2-S 48.1 AP **[V]** | Likely. |
| Detector | Ultralytics YOLO26/YOLO11 (current) | ultralytics.com/license **[S]** | **AGPL-3.0** (Enterprise licence for closed use) **[S]** | — | Yes, but licence conflicts with an own-licence release. |
| Tracker | **ByteTrack** | https://github.com/ifzhang/ByteTrack **[V]** | **MIT** **[V]** | MOT17 test HOTA 63.1 / IDF1 77.3 **[V]** | CPU-cheap. |
| Tracker | **OC-SORT** | https://github.com/noahcao/OC_SORT **[V]** | **MIT** **[V]** | MOT17 private HOTA 63.2 / IDF1 77.5 **[V]** | Better under occlusion/non-linear motion (authors). |
| Tracker | BoT-SORT (with re-ID / camera motion) | NirAharon/BOT-SORT | MIT **[S]** | — | Camera motion not needed (fixed). |
| Tracker library | Roboflow `trackers` (SORT, ByteTrack, OC-SORT, BoT-SORT…) | https://trackers.roboflow.com **[S]** | **Apache-2.0** **[S]** | — | Already installed in the venv (`trackers==2.7.0`, PHASE1_RECORD) — would avoid re-implementing. |
| Pose | **RTMPose-s** (MMPose) | https://arxiv.org/abs/2303.07399; MMPose repo | **Apache-2.0** (MMPose) **[S]** | COCO AP 72.2 % (authors) **[S]** | Sibling project measured RTMPose-S 94 MB for 6 crops (MEASURED, PHASE1_RECORD). |
| Pose | RF-DETR Keypoint (preview) | rf-detr repo **[V]** | Apache-2.0 **[V]** | 71.8 AP, 40.7 M params **[V]** | Preview status. |

Padel-specific: 4 players, fixed court → a **4-identity constraint** (exactly two per side, teams swap sides only at changes of ends) is a strong prior that generic MOT trackers do not use.

## 6. Shot detection and classification

| Source | Links | Licence | Notes |
|---|---|---|---|
| PadelTracker100 shot labels: binary shot flag + 6 classes (forehand, backhand, smash, serve, dropshot, other) over 40,135 frames | see §7 | CC BY 4.0 **[V]** | First public padel shot labels found. |
| Pose-sequence classifiers (ST-GCN/LSTM style) | — | — | Standard; needs pose + labelled shots. |
| Current RF on 39 samples | audit | own | 71.8 % CV vs 59.0 % baseline. |
Shot type is **not needed for score v1**; except "serve", which marks point starts.

## 7. Public datasets and weights

| Dataset | Content | Links | Licence | Fit |
|---|---|---|---|---|
| **PadelTracker100** (NTT Data, *Data in Brief* 2026) | ~100k frames (45,934 + 53,953) of the 2022 World Padel Tour Finals (men's and women's finals), **one fixed main-view camera ~7.6 m high, 15.5 m behind the glass**, 1920×1080 @ 30 fps; ball boxes (COCO), player boxes + 17 keypoints, court positions in metres, shot labels (6 classes); bounces **not** annotated | Article https://pmc.ncbi.nlm.nih.gov/articles/PMC12926558/ **[V]**; data https://doi.org/10.5281/zenodo.17020011 **[V]** | **CC BY 4.0** **[V]** | **Best match for a fixed-camera padel product.** Caveats **[V]**: Zenodo holds only annotations (labels.zip 69.7 MB) — frames must come from the two linked YouTube videos (download terms: UNVERIFIED, decision needed); ball labels were partly produced by a YOLOv8 model then reviewed (inter-annotator ball IoU 0.677); professional venue only. |
| Decorte et al. padel hit dataset | ≈ 5 h 28 min, highlights + 2 full matches, hits, rally start/end frames **[S]** | paper (above) | availability UNVERIFIED | Ideal for audio hits and rally boundaries if obtainable. |
| PADELVIC (2024) | multi-camera amateur match + motion capture | https://www.researchgate.net/publication/377568366 **[S]** | UNVERIFIED | Player tracking/pose, not ball. |
| TrackNet tennis dataset | ~10 matches broadcast, ball positions, used by many trackers | via TrackNet papers **[S]** | UNVERIFIED | Ball pre-training; broadcast domain. |
| Shuttlecock Trajectory Dataset | badminton, TrackNetV3 training data | TrackNetV3 README **[V]** | UNVERIFIED | Pre-training only. |
| Roboflow Universe padel sets (ball, court keypoints) | small, undocumented | e.g. https://universe.roboflow.com/general-pmycw/padel_court_keypoints **[S]** | per-project | Informal; licence per project. |
| **Your own club recordings** | full matches, fixed camera, real score | — | yours | **The only valid end-to-end test set.** |

## 8. What changes versus the current project

| Weak link (audit) | State of the art says | Direction |
|---|---|---|
| Bounce = image-y peak | Bounces are trajectory corners; hits are best heard (padel audio F1 92 %) | Audio hits + trajectory bounce classifier + 3D court |
| One-rule FSM, no uncertainty | Score must follow FIP options; side changes and serves are observable anchors | Probabilistic point winners + constrained score decoding |
| Ball 65.9 % visible, unlimited interpolation | TrackNetV3/WASB families; occlusion/blur-aware variants; physics-limited gap filling | Retrain/fine-tune on padel frames; Kalman with coasting limit |
| AGPL / NC / GPL dependencies | Apache/MIT alternatives exist for every perception block | RF-DETR or RT-DETR, ByteTrack/OC-SORT, RTMPose, TrackNetV3/WASB code (MIT) |
| No ground truth | PadelTracker100 (CC BY 4.0) + own recordings | Phase 3 labelling plan |

## UNVERIFIED register
1. Licences of E2E-Spot, TrackNetV4/V5, TOTNet, BlurBall, MMPose sub-projects, Roboflow `trackers` (secondary only), Ultralytics (secondary, though consistent).
2. Availability of the Decorte padel dataset and code.
3. Vollytics content (page returned 403).
4. Whether downloading the PadelTracker100 source videos from YouTube is permitted for your use.
5. WASB model-zoo contents per sport (README points to MODEL_ZOO.md; not opened).
6. Training VRAM of every model on a 4 GB budget — to be measured, not assumed.
7. Novelty of constrained score decoding for padel (no prior work found in one search pass).
