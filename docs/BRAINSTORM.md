# Brainstorm: padel problems solved with physics and logic

Status: ideas, not results. Each idea says what it needs, how to test it, and what we already measured.
Tags: **HAVE** (data/code exists), **NEED** (missing input), **HYPOTHESIS** (not tested). Numbers marked MEASURED are
from `docs/BENCHMARK_BASELINE.md`.

## 0. Where we are (MEASURED)
Players: 99.7-100 % tracked, positions in metres. Ball in the image: F1 0.78 at 5 px / 0.86 at 10 px (WASB + TrackNetV3
merged, two clips). Hits: audio F1 0.64 at +-3 frames; trajectory kinks cover 83-92 % of labelled hits but with precision
0.14-0.2. Score from the broadcast scoreboard: 29 points, all transitions reachable under FIP rules.
Missing: the ball in 3D (only pixels), bounces typed (floor / glass / fence), shot types, point winner from video.

## 1. Score from video

| Idea | Logic / physics | Needs | Test |
|---|---|---|---|
| Rules as a state machine | serve -> must land in the diagonal service box -> teams alternate hits -> point ends on 2nd bounce on one side, ball out of bounds after the glass, net, or fault | typed bounces (idea 2.x) | replay on labelled points, compare winner |
| Decode, don't threshold | Viterbi over the event sequence: a missed bounce or a false hit costs probability, it does not reset the point (the old project reset on every false hit) | event probabilities | accuracy of point winner vs hard rules |
| Score automaton as a prior | `scoring/automaton.py` already encodes FIP scoring; decode the whole match so impossible scores are never output; serve rotation and change of ends after odd games are free constraints | per-point winner probabilities | HAVE automaton; evaluate on highlights (scoreboard = ground truth) |
| End-of-point cues | no hit for > 2 s, ball gone, 4 players stop/converge, crowd noise, <= 20 s between points (FIP) | audio, tracks | segmentation F1 on CVSPORTS rallies (start/end labelled) |
| Winner = last hitter + where the ball died | the side where the last floor bounce lands loses the point unless the last hitter hit it out/net | typed bounces | HYPOTHESIS |
| Free ground truth | scoreboard OCR labels every point of broadcast footage; CVSPORTS gives 99 rallies with start/end | HAVE | train + validate winner classifier |

## 2. Ball in 3D and bounces

1. **Camera from DLT.** 12 floor points plus >= 6 points of known height (net posts 0.92 m, net centre 0.88 m, glass top 3 m)
   give the 3x4 projection matrix by linear least squares; no focal-length guess. NEED: ~6 clicks per camera.
   Test: reprojection and leave-one-out error in px/cm (as `CourtCamera` already reports for the floor).
2. **Ballistic flight fit.** p(t) = p0 + v0 t - 1/2 g t^2 z (+ small drag). Six unknowns, two equations per frame:
   four frames pin a flight down. Least squares on reprojection error. Outputs height, speed in km/h, apex.
3. **Bounce links two flights.** On the floor the vertical velocity flips with a restitution coefficient e. The FIP ball
   rebounds 135-145 cm from 254 cm (to verify in the FIP PDF), so e = sqrt(1.40/2.54) = 0.74. Fitting both flights jointly
   makes each more robust and separates a floor bounce (z = 0, vz flips) from a racket hit (any z, any new direction).
4. **Contact type from 3D position.** z ~ 0: floor. |x| = 5 m or |y| = 10 m with z < 3 m: glass. Above 3 m: fence (lower e).
   y = 0 with z < 0.88 m: net. Test: hand-label bounce frames on the two benchmark clips.
5. **Reflections become measurements.** A glass reflection is the mirror image of the ball across the glass plane. Once the
   glass plane is known, a blob that mirrors the 3D ball position confirms the ball instead of confusing the tracker.
   (Today they are only rejected.) HYPOTHESIS.
6. **Shadow trick.** Under floodlights the ball and its shadow lie on the line to the lamp; the gap gives height as an
   independent check on the 3D fit. HYPOTHESIS, only works on lit courts with visible shadows.
7. **Sound as a second sensor.** Racket, floor, glass and fence sound different (spectrum, decay); cross-check typed bounces.
   HYPOTHESIS; needs labelled audio events (CVSPORTS has hits only).
8. **Two cameras (Padmi-like).** Triangulation removes the depth ambiguity completely; one camera + ballistics is the
   cheaper alternative and we should measure how far it goes first.

## 3. Hits and shot types

1. **Hit = velocity jump near a player.** In 3D: |dv| above a threshold and the ball inside a reach sphere around a racket
   hand (~2.2 m from the torso, up to ~2.6 m high). Kills the audio-only false positives (steps, glass rattle).
2. **Alternation logic.** Hits must alternate between teams (a team has at most 1 hit per exchange, in doubles either partner).
   Decode the hitter sequence with Viterbi so a double detection collapses and a missing hit is inferred.
3. **Learned fusion.** Candidates = kinks (recall 83-92 %) + audio onsets; a small classifier on ball speed change, distance
   to each player's wrist, audio strength, and time since last hit, trained on the 2 377 CVSPORTS hits. Replaces hand rules
   (which gained only +0.01-0.02 on 2 clips). NEED: run the ball pipeline on the 99 rallies (download in progress).
4. **Shot type by explainable rules first**, then ML on PadelTracker100's six classes:
   - serve: first hit of the point, contact below the waist, ball bounced before;
   - volley: no bounce before the hit, contact within ~4 m of the net;
   - smash: contact above ~2.5 m, fast, downward;
   - bandeja / vibora: high contact 4-7 m from the net, slower with spin (bandeja) / faster, sidespin (vibora);
   - lob: steep upward exit, apex > 4 m;
   - chiquita: slow, dips at the feet of the net players;
   - bajada / glass defence: hit after the ball rebounded off the back glass.
5. **Forehand / backhand** from the wrist side relative to the torso midline; learn each player's handedness over the match.
6. **Pose is the missing sensor** (RTMPose / ViTPose, see `docs/research/EXPERIMENT_CANDIDATES.md`): wrists for the hitter,
   shoulders for the swing, hips for stance.

## 4. Tactics and statistics (what Padmi-like products show)

- Formation per second: both at the net / one up one back / both back; share of time at the net; transitions after lobs.
- Pressure index: time the opponent had to reach the ball = distance to the landing point / their speed at the hit.
- Coverage: Voronoi cells of the two partners on their half; the gap between them is where a smart opponent aims.
- Placement maps: landing points (3D), serve and return maps per player.
- Rally patterns: n-grams of shot types (lob -> bandeja -> ...) against who won the point.
- Workload: distance (HAVE, smoothed), sprints above 4 m/s, accelerations, recovery between points.
- Net-play physics: reaction time = time between the opponent's contact and the ball crossing the net plane.

## 5. Suggested order (impact on score accuracy / effort)
1. Learned hit detector on CVSPORTS (data exists, no new clicks).
2. 3D calibration (6 clicks) -> ball height, speed, typed bounces.
3. Point segmentation + winner with Viterbi through the score automaton.
4. Pose + shot types.
5. Tactics dashboard on top of the above.

## 6. Risks
- One fixed camera gives an ambiguity along the viewing ray: ballistics resolves it only with enough visible frames per flight.
- The ball is lost in ~20 % of frames; every physics idea must degrade to "unknown" rather than invent.
- Our benchmark is two clips from one broadcast; nothing here is validated on club footage yet.
