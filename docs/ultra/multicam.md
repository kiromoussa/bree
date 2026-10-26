# Multi-camera store-wide fusion (ultra track: multicam)

Status: plan (written first, results are added at the end of the run).

## Goal

Make several real channels of the Truman Rd. store (7, 29, 12, 14, 15, 10, others if useful) act as ONE store:
a shared floor plan in metres, a floor calibration per camera derived from real stills, one global identity per
person across cameras (no faces, body colour only, behind the existing re-ID flag), a store-wide journey record
(door, aisle, cooler, register, exit), and a handoff evaluation on REAL-MODEL-LABELLED sequences.

## What already exists (and was only ever run on SIMULATED clips)

* `bree/track/multicam.py` online handoff by floor position, time and optional re-ID, one ledger per store.
* `bree/track/floor.py` a store-wide floor tracker for cameras that see whole people.
* `bree/calib/camera.py` a pinhole camera and a homography fit from marks.

None of them has been fed a real camera. The real work is therefore the real inputs: a plan, calibrations, per-camera
tracklets from cached detections and a labelled evaluation.

## Plan

1. `bree.fusion.detect_cache`: person boxes plus clothing colour of every recorded segment, cached once (local data).
2. `bree.fusion.calibrate`: floor calibration helper. Automatic proposals (floor line vanishing points, accent
   tiles) and the human step (floor marks with metres -> homography, error, overlay). Output is a JSON that a person can correct.
3. `bree.fusion.floorplan`: the shared plan (JSON): regions with kinds (door, aisle, cooler, register, counter-staff), cameras
   with their homography and the floor area they can place people on, the camera adjacency graph with expected transit times.
4. `bree.fusion.registry`: the independent check, and the refinement: people seen by two cameras at one moment must
   land on the same floor point. Reports the disagreement in metres per camera pair.
5. `bree.fusion.tracklets` + `bree.fusion.handoff`: per-camera tracklets, then one constrained clustering over all
   tracklets of the store (overlapping fields of view: agree on the floor at the same moment; non-overlapping: topology,
   walking time, body colour).
6. `bree.fusion.journey`: store-wide journey record per global visit and the aggregate region-to-region flow.
7. `bree.fusion.eval`: labelled evaluation (identities per person, purity, handoff precision and recall) against
   the no-fusion baseline and against the existing online handoff.

Labels are made by the agent looking at synchronised frames from several cameras (REAL-MODEL-LABELLED). The store is
nearly empty in the recorded footage, so the numbers are tiny-sample. Nothing here is accuracy.
