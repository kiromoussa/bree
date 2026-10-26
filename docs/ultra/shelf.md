# Shelf and cooler state (track: shelf)

Plan written first, results appended at the end (section "Results"). Every number carries a label:
SIMULATED, INJECTED (a synthetic change pasted into real frames), REAL-UNLABELLED, REAL-MODEL-LABELLED.

## What exists and what is missing
`bree.shelf.*` (round 1-5 work) reads takes and puts at a calibrated slot from a planogram. That needs camera
calibration, a layout and a SKU detector; none exist for the real store. This track adds `bree.shelfstate`: state
monitoring from one fixed camera with no calibration, no product model, no planogram: per-cell comparison with a
person-free reference.

## Plan
1. Sample at 1 fps, 640 px wide. Person boxes from the repo's YOLO weights at low confidence (gate, not counting).
2. Reference = per-pixel median of person-free frames; compared cell by cell (10 px cells) with NCC (brightness
   invariant), chroma and a smooth gain field (auto-exposure, lamp, daylight).
3. A change is an event only when it lasted `settle_s`, stopped moving, had no person for `clear_s`; the reference is then
   replaced there (one change, one event). Flickering cells (TV, sign) are learned and ignored; burned-in clock masked.
4. Region kinds from a per-camera YAML: shelf, cooler_door, floor, display, ignore (window glass, back counter).
5. Event kinds: stock_down, restock, layout_change, floor_object, floor_spill, floor_cleared, door_change,
   door_ajar, low_stock (against the golden picture), plus a status stream (light change, camera unreliable).
6. Evidence: before/after crop and context view, every person box mosaicked.
7. Evaluation: false events per hour on normal footage, REAL-MODEL-LABELLED event check by viewing before/after,
   recall on INJECTED changes (pasted items, spill, door, removal) at several exposure/noise settings.
