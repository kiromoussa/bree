## Fixed benchmark (simulated clips)

Everything in this section is SIMULATED (browser store simulator copy). It is the one benchmark every pipeline
change is measured on.

```
make bench-clips     # render the DEV and TEST clips once (Chrome, node, ffmpeg, BREE_PLAYWRIGHT)
make bench-dev       # pipeline on the 6 DEV clips -> results/bench_dev.json and .md (scorecard + per-pick funnel)
make bench-test      # the 6 TEST clips: final numbers only, never for tuning
make bench-smoke     # one clip, 60 frames per camera, writes nothing to results/
```

- Splits are fixed by seed in `scripts/bench/manifest.json`: train 1000 to 4999, dev 7001 to 7006, test 9001 to
  9006. Training data may only come from train seeds.
- A clip is 4 to 8 shoppers over about 60 to 90 s on the recommended-3d-45 layout: honest shoppers and thieves,
  picks at gondolas, in the cooler and at the counter, put-backs, two shoppers at one shelf, register payments.
  16 to 20 cameras per clip as H.264 video, with calibration, the planogram, the register feed and ground truth.
- The scorer reports theft recall, alert precision, false alerts and reviews on honest shoppers, pick recall
  and precision, right SKU rate, time to alert, identities per shopper, and a funnel that says at which stage
  each true pick was lost. The pipeline never sees the ground truth: it is run on a view of the clip without it.
- Clip format, commands and the scorer's input contract: `scripts/bench/README.md` and the docstring of
  `src/bree/sim/bench.py`.
- The recorded baseline (per-camera engine at 19f51dc) is `results/bench_dev_baseline.md`: 0 of 20 thefts caught, 2 of
  70 picks, 4.0 identities per shopper. Rerun it with `make bench-dev BENCH_ARGS="--runner bree.sim.bench:run_pipeline --name baseline"`.
