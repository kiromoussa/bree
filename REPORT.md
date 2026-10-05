# BREE vision: report

Twelve parts, newest first. **Audit fixes (2026-10-05)** comes first. Then, from the integration of four parallel work streams: **Integration summary (2026-10-05)**, **End to end on simulated data**, **SKU detector trained on simulated frames**, **Calibration, 3D slots and store-wide identity**, **Camera node first pass**, **Human review feedback loop**. Then, unchanged: **Closed-world identity (2026-10-04)**, **Re-ID without the face (2026-10-03)**, **Overnight (2026-10-01)**, **Phase 2 (2026-09-30, real data)** and the **Phase 1 report**. Every number names the file it came from; real-data results and simulated or synthetic results are kept apart and labelled.

# One pipeline path: shelf events, floor tracks, association, ledger (2026-10-05)

All numbers in this section are SIMULATED: the DEV split of the fixed benchmark (6 clips from the browser store
simulator, 421 s, 37 shoppers, 15 thieves, 70 picks, 20 stolen items). The TEST split was not run.

**Bottom line.** The three core streams are wired into one runner (`bree.shelf.store:run`, the default of
`make bench-dev`). On DEV it puts 16 of 20 stolen items in front of a reviewer (was 0 of 20), finds 67 of 70 picks
(was 2) with the right product for 91 percent of them, and keeps 1.27 identities per shopper (was 4.0). The goal is
not met: 4 of 22 honest shoppers get a review (the bar is 1 in 10), and no theft reaches alert tier because nothing
detects concealment yet.

| DEV, SIMULATED | per-camera engine (baseline) | first join of the streams | after the two fixes | goal |
|---|---|---|---|---|
| Theft recall, alert or review | 0 of 20 | 12 of 20 (0.60) | 16 of 20 (0.80) | 0.80 |
| Theft recall, alert tier | 0 of 20 | 0 of 20 | 0 of 20 | |
| Pick recall | 2 of 70 (0.03) | 69 of 70 (0.99) | 67 of 70 (0.96) | 0.80 |
| Right SKU of detected picks | 1 of 2 | 0.913 | 0.910 | 0.85 |
| Pick precision | 2 of 2 | 0.552 (125 PICK events) | 0.744 (90 PICK events) | |
| Right shopper of detected picks | 2 of 2 | 0.928 | 0.940 | |
| Alerts on honest shoppers | 0 | 0 | 0 | |
| Reviews on honest shoppers | 0 of 22 | 6 of 22 | 4 of 22 (1.8 per 10) | 1 per 10 |
| Identities per shopper | 4.0 | 1.27 | 1.27 | 1.5 |
| Identities covering two shoppers | 93 | 15 | 6 | |

Files: `results/bench_dev_baseline.md`, `results/bench_dev_first_join.md` (run before the fixes),
`results/bench_dev.md` (final, `make bench-dev`, 577 s for the six clips on this laptop).

### The funnel after the fixes (70 true picks, each counted at the first stage it fails)

| stage | reached | lost here | passed on its own |
|---|---|---|---|
| in view | 70 | 0 | 70 |
| frame reached pipeline | 70 | 0 | 70 |
| hand or item detected | 70 | 8 | 62 |
| shelf event emitted | 62 | 0 | 67 |
| right slot | 62 | 10 | 57 |
| right SKU | 52 | 0 | 61 |
| associated to a shopper | 52 | 0 | 67 |
| right shopper | 52 | 0 | 63 |
| conceal or pay classified | 52 | 18 | 49 |
| ledger basket | 34 | 8 | 56 |
| alert | 26 | 8 | 31 |

- **Conceal or pay classified loses the most, 18 picks, 17 of them stolen items.** There is no CONCEAL event at all:
  the overhead pose does not show concealment in this simulator (association section below) and no item camera
  reports it. This is why alert tier is 0 of 20. The stolen items still reach a reviewer through the ledger (18 of 20
  are listed as unpaid on some record).
- **"Hand or item detected" (8) is a logging gap, not a miss.** The shelf cameras do not write per-frame boxes to the
  frame log; the runner logs the reach point of each shelf event only. 67 of 70 picks have a PICK event.
- **Right slot (10).** Mostly the neighbouring facing, as in the shelf-events section. The SKU is right more often
  than the slot (61 against 57).

### What was wrong after the first join, and the two fixes

1. **One reach was read as several takes.** 125 PICK events for 70 picks. Of the 60 false takes, 45 were repeats of a
   true take within 5 s and 1.5 m (one camera at two facings, or two cameras at slots too far apart to be fused), 7
   more sat next to another false take. Each repeat is an unpaid item on somebody. Fix:
   `bree.shelf.store.one_act_per_reach`, after association: takes (or puts) by one shopper within 4 s and 1 m are one
   act, and the best-evidenced reading is kept. 34 takes and 11 puts were merged on DEV. Merging before association
   was tried first and lost true picks of two shoppers at neighbouring cooler doors (pick recall 0.91), so the merge
   is per shopper.
2. **Identities were swapped when two shoppers passed close to each other** (15 identities covered two shoppers; a
   thief's basket left the store under an honest shopper's identity). Fix: person boxes carry clothing colour
   (`bree.track.people.appearance`, upper and lower body, never the head) and `bree.track.floor` uses it twice: to
   pick who reappeared after being lost, and after two people came within 0.8 m, to check who is who once both are
   seen apart. 6 swaps were put right on DEV, identities covering two shoppers fell from 15 to 6, and identities
   marked uncertain from 31 to 19 of 41.

What-ifs on the stored shelf events and person boxes (same six clips, tracking, association and ledger only):

| change | thefts flagged | reviews on honest shoppers | pick recall | pick precision |
|---|---|---|---|---|
| first join | 12 | 6 | 0.986 | 0.552 |
| + clothing colour | 11 | 6 | 0.986 | 0.552 |
| + clothing colour + one act per reach (4 s, 1.0 m): the final setting | 16 | 4 | 0.957 | 0.744 |
| same with 3 s, 0.6 m | 15 | 4 | 0.986 | 0.676 |
| same with 6 s, 1.0 m | 16 | 4 | 0.957 | 0.761 |
| final setting, strangers never grouped (`group_window_s=0`) | 15 | 7 | 0.957 | 0.744 |

Clothing colour alone does not move theft recall (11 against 12), it moves identity; the two fixes together do.

### What is left (DEV, final run)

- **4 missed thefts.** One pick was never seen. Two were picked by the right identity, which then became another
  shopper before the exit (a hand-back after 7 s out of view in 7001, a group record in 7002 that the scorer credits
  to the other shopper). One thief stood at the register next to another payer and the other shopper's receipt
  cleared the stolen item (7006).
- **4 reviews on honest shoppers.** Their 10 flagged items: 4 false takes (two from the slot watch cue alone), 2
  true picks of another shopper, 2 own picks that were paid, 1 own pick that was put back, 1 own pick with the wrong
  SKU of a sibling product.
- **Tests:** the full suite (`.venv/bin/python -m pytest`) ran after the merge: 398 passed, 1 skipped (the browser test of
  the review page, `BREE_PLAYWRIGHT` not set in that shell), none failed.
- **Not measured:** the shelf rules were tuned with the old weights and `make shelf-eval` was not repeated with
  `sim_sku_hands_v3`; the old weights were not run through the DEV benchmark; nothing was run on TEST.

## 2026-10-05: improvement round 4 (SIMULATED): one person, one track

All numbers are simulated. TEST was not touched. No training, no new clips. DEV was run from scratch
(`make bench-dev`, stamped 65f4103: 8b5e265 plus a comment); the TRAIN-seed clips were rejoined from their stored shelf events and person
boxes (`make bench-train BENCH_ARGS=--keep`: nothing before tracking changed this round).

**Result: the headline did not move. 17 of 20 thefts flagged, 3 of 22 honest shoppers reviewed, the goal is still
missed by one honest shopper. What improved is underneath: fewer identities, fewer wrong hand-backs, fewer false
picks, on both sets, with no theft lost and no honest shopper added on either.**

| | DEV before | DEV now | TRAIN-seed before | TRAIN-seed now |
|---|---|---|---|---|
| theft recall, alert or review | 0.85 (17/20) | 0.85 (17/20) | 0.789 (15/19) | 0.789 (15/19) |
| reviews on honest shoppers | 3 of 22 | 3 of 22 | 3 of 36 | 3 of 36 |
| pick recall | 0.957 | 0.957 | 0.915 | 0.915 |
| pick precision | 0.848 (79 PICK events) | 0.905 (74) | 0.858 (113) | 0.898 (108) |
| right SKU of paired picks | 0.940 | 0.940 | 0.938 | 0.938 |
| identities per shopper | 1.27 (47) | 1.216 (45) | 1.453 (77) | 1.377 (73) |
| people "first seen inside" | 6 | 2 | 13 | 8 |
| hand-backs to the wrong person | 6 of 34 | 3 of 31 | 25 of 74 | 22 of 72 |
| identities covering two shoppers | 6 | 9 | 26 | 21 |

The last row got worse on DEV (two clips, 7001 and 7006): a second track that used to become an extra person and
soak up a mix-up no longer exists, so the mix-up sits on the two real identities, marked uncertain.

### Stage looked at: identity

Round 3 named it the next bottleneck (one honest review and one missed theft on DEV go through it). Every hand-back
of the floor tracker was checked against the true shopper before and after (scoring side only). The log line of a
hand-back now says how far from where the person was lost the new track started, how different the clothing colour
is, and how many lost people fitted.

**Root cause in one sentence: when two cameras place one person more than 0.8 m apart, the tracker started a second
track on them, let it grow into "a person first seen inside the store", and the original track, the one holding the
picks, starved and stayed lost in the store for ever, to be handed to the next stranger who appeared.**

Checked first: of 18 people "first seen inside" on DEV plus the TRAIN-seed clips (counted with the clothing gate
below switched on), 9 were a second track of the tracked
person standing nearest (all within 0.9 m of them); the other 9 were 1.4 m or more away, or across the counter from
staff. In DEV 7001 the thief P001 was tracked as id 2 until 17.2 s and as "new person" id 6 from 16.0 s; id 2, holding
the stolen item, was never seen again and was given to P007 when she walked in at 24.9 s, 11.2 m away.

### Kept (commit 8b5e265)

1. **A track that starts inside within 1.0 m of a tracked person, with nobody lost, is a second track of them**, not
   an entry nobody saw (`FloorConfig.second_track_m`; staff behind the counter and a customer in front of it are not
   merged). It gets no identity. If the person's own track is then lost, the ordinary hand-back gives the second
   track their identity. The 1.0 m was read off the TRAIN-seed clips (second tracks at 0.4 to 0.7 m, real other people
   at 1.4 m and more) and holds on DEV (0.4 to 0.9 m).
2. **Pixels that change within reach of where the payer stands, with no item seen in a hand, are not a pick**
   (`pay_reach_m` 0.75 m around `poi.register` of the layout, in `bree.events.shelf.confirm_puts`). The backbar
   camera reads the payer's arm or their goods on the counter as a take from the two counter slots next to the
   register: 0 of 5 such events were a real take on the TRAIN-seed clips (0 of 4 on DEV), while 2 of 2 pixel-only
   takes further along the counter were real (2 of 3 on DEV). This is what moved pick precision. A take there with
   the item seen in a hand is passed on as before.

DEV 7001 after the change: P001 keeps id 2 from the door to the exit and her theft is flagged. The clip's count does
not change, because the other thief (P003) now shares the register with P005 at 53.7 s, the two identities swap, and
both of P003's `peanut_pilot` (one stolen, one paid) are covered by the two receipts that land on that identity.

### Tried on both sets and not kept

Thefts flagged / honest shoppers reviewed, DEV then TRAIN-seed. Kept version: 17 and 3, 15 and 3. All by rejoining
stored events (`scripts/bench/whatif.py`, logs in `out/bench/round4/`).

| variant | DEV | TRAIN-seed | |
|---|---|---|---|
| clothing gate: a track more than 2 m from where a person was lost, colour distance over 0.25, is not them | 17 and 3 | 14 and 4 | refuses 12 of 25 wrong hand-backs and 0 of 45 right ones on TRAIN-seed, 5 of 6 and 0 of 26 on DEV. In the code, off (`FloorConfig.other_m`) |
| an act is timed by the readings no put took back (`standing`) | 16 and 2 | 14 and 3 | fixes the 7006 double read. In the code, off |
| `standing` plus no discount for a pick two people fit (`ambiguous_factor` 1.0) | 17 and 2 | 15 and 4 | meets every DEV bar; one more honest review on TRAIN-seed |
| the same plus "a put cannot return an act still read as taken afterwards" | 17 and 2 | 16 and 5 | reverted: reading times are late by seconds (a take at 14.7 s read at 18.3 s) |
| no discount alone | 17 and 3 | 16 and 4 | |
| `from_last` (round 3) on the new identities | 17 and 2 | 14 and 3 | as in round 3 |

The pattern in the first three rows is the finding of this round: **when identity or the event list gets more right,
the headline gets worse, because several flagged thefts were flagged through an error.** In TRAIN-seed 4951 the gate
keeps thief P005 on one identity for the whole visit, and his theft drops from review to nothing: the pick also fits
a second person, which halves its score, and it ends at 0.25 against a bar of 0.4. Before, a wrong identity carried
it over the bar. In DEV 7005 `standing` merges a false second
take into the real one, and the theft that the false take had pushed over the bar is lost the same way.

### Why the goal is still missed, and what cannot be fixed in the join

The three honest shoppers reviewed on DEV each have their own cause, and none is identity any more:

- 7001 P007: a camera on the far gondola read 456 changed pixels at a top-shelf slot (1,284 visible pixels) as a
  take while she walked past 1.06 m in front of it. Nothing measured separates it from real pixel-only takes. With
  the pay-point rule, 9 of 11 one-camera pixel-only takes that reach the ledger are real (5 of 5 on the TRAIN-seed
  clips, 4 of 6 on DEV); the two false ones are both on DEV, and their size, slot score, duration and distance to
  the shopper all sit inside the range of the real ones. Two stolen items on DEV are read this way and no other, so
  the class cannot be dropped.
- 7005 P008: unchanged from round 3 (the cooler camera that read the take never reported the put-back).
- 7006 P001: one reach read twice 4.2 s apart. `standing` fixes it and costs a theft, see above.

## 2026-10-05: improvement round 3 (SIMULATED): a put names the take it undoes

All numbers are simulated. TEST was not touched. No training, no new clips. Both sets were run from scratch
(`make bench-dev`, `make bench-train`, code at f72e3b4; `results/bench_dev.md` is stamped "28989f3 plus uncommitted
changes" because it ran just before the commit, the code is the same).

**Result: one false review fewer on DEV, two more thefts flagged on the TRAIN-seed clips, nothing worse on either.
The goal is still missed by one honest shopper.**

| | DEV before | DEV now | TRAIN-seed before | TRAIN-seed now |
|---|---|---|---|---|
| theft recall, alert or review | 0.85 (17/20) | 0.85 (17/20) | 0.684 (13/19) | 0.789 (15/19) |
| reviews on honest shoppers | 4 of 22 | 3 of 22 | 4 of 36 | 3 of 36 |
| pick recall | 0.957 | 0.957 | 0.915 | 0.915 |
| pick precision | 0.848 (79) | 0.848 (79) | 0.866 (112) | 0.858 (113) |
| right SKU of paired picks | 0.910 | 0.940 | 0.938 | 0.938 |
| right slot of paired picks | 0.851 | 0.881 | 0.825 | 0.835 |
| right shopper of paired picks | 0.940 | 0.940 | 0.979 | 0.969 |
| identities per shopper | 1.27 | 1.27 | 1.453 | 1.453 |

Goal check on DEV: theft recall 0.85, pick recall 0.957, right SKU 0.940 and 1.27 identities per shopper meet their
bars. Reviews on honest shoppers do not: 3 of 22 is 1.4 per 10 against 1 per 10 (that is 2 of 22). All three are
review tier; there is still no alert tier (0 of 20).

**Stage picked and what the cases showed.** The failing goal is reviews on honest shoppers, and round 2 named the put
events as the weakest input. Reading the four DEV cases frame by frame and the put events of both sets gave the root
cause in one sentence: a put event of the pixel comparison is a true statement about one camera and one slot ("this
place looks again as it did before my own take of it", because the item is back or because an arm that covered the
slot went away), but the join treated it as "this shopper returned this product", matched by place and product name
after the takes had been merged across cameras, so it either undid nothing (put ignored, honest shopper reviewed) or
undid a real take (stolen item erased).

Checked before changing anything: on the TRAIN-seed clips the place of the take is fully back to the old picture in
82 of 85 pixel puts (share of the take's patch still changed under 0.05; 0.45 to 0.56 in the other three), true put-back or not. So the put is not a
misread; what was wrong is what it was taken to mean.

**What was changed (2 commits, e93c4bf and f72e3b4).**

1. **A put names the take it undoes.** Every take carries the names of its readings (`eids`); a put of the pixel
   comparison carries `undoes`, the name of that camera's own take of that slot. The names survive the merges (two
   cues in one camera, two cameras, one reach read twice). In the join a put takes only the reading it names out of
   the act. An act with no reading left is returned, to whoever was given the take, wherever the put was read and
   whoever stood nearest. An act another camera still reads stands, under the name a remaining reading gave it.
2. **A put with the item seen going in returns the whole act.** A held-item track that ends at the slot, coming from
   further away (`item_in`), is the pixel cue round 2 asked for. Scored against truth afterwards: 19 of 28 such puts
   are real put-backs on the TRAIN-seed clips and 4 of 6 on DEV, against 27 of 65 and 9 of 24 for "an item was seen
   in a hand near the put". The two thresholds were read off the TRAIN-seed clips, not DEV.
3. **A take whose slot was hidden for long is timed by the item.** When somebody stands in front of a slot and the
   item is first seen in a hand inside the hidden span, the take is 0.3 s before that sighting, not the moment the
   slot was first hidden (clip 7003: 6.3 s early, so the take went to the shopper who stood there before). The 0.3 s
   is the median lead on the TRAIN-seed clips (0.17 s, 9 in 10 under 0.46 s); over 147 takes there the mean time
   error goes from 0.23 s to 0.20 s.

**Variants measured on both sets** (thefts flagged / honest shoppers reviewed; from the stored events of this
round's code, `scripts/bench/whatif.py`):

| variant | DEV | TRAIN-seed |
|---|---|---|
| round 2 code | 17/20, 4/22 | 13/19, 4/36 |
| kept: a put undoes its reading, item seen going in returns the act | 17, 3 | 15, 3 |
| the same, and a reach is measured from its latest reading (`from_last`) | 17, 2 | 14, 3 |
| a put never returns an act other cameras still read (`put_returns` "none") | 17, 5 | not run without `from_last`; 15, 5 with it |
| any item seen in a hand near the put returns the act (`put_returns` "both") | 12, 2 | not run |
| a camera's own take and put pair is dropped before the cameras are merged | 16, 2 | 13, 3 |
| a track that starts at the door walking in is never a person lost inside (floor tracker) | 16, 3 (round 2 events) | 13, 4 |

`from_last` meets every bar on DEV (17 of 20, 2 of 22). It was not switched on: it loses one flagged theft on the
TRAIN-seed clips, so over both sets it trades one error for another (32 thefts and 6 reviews against 31 and 5). The
option is in the code (`one_act_per_reach(from_last=True)`), off.

**The three honest reviews left on DEV, and why this round does not reach them.**

- 7001 P007: the floor tracker handed the identity of a thief who was lost in an aisle to the next shopper through
  the door 7.4 s later, so the thief's item is on the honest shopper and the theft is missed. An identity fault. The
  door rule in the table clears the review but loses a flagged theft in 7005.
- 7005 P008: the cooler camera that read the take never reported the put-back; a second camera read both, at a slot
  of another product one door away, with no item track going in. Nothing in the pictures the pipeline reads says the
  first camera's reading is undone.
- 7006 P001: one take of a cooler drink read twice, 4.2 s apart, because the act's time came from a third reading
  that was later undone. `from_last` merges it.

**The three missed thefts on DEV** are unchanged: 7001 P001 (the identity fault above), 7005 P001 (took from the
counter, stood at the register, no receipt: the ledger scores one such item 0.27 to 0.30, below review, by design
for POS gaps), 7005 P007 (no shelf event: a passer-by covered more than half of the only camera's picture 0.7 s
after the take).

**Tests.** Full suite on the final code: 410 passed, 1 skipped, 0 failed (counted from the progress lines of
`out/bench/round3/tests_full.log`; pytest prints no closing line here, as in rounds 1 and 2). Three new tests.

**What limits the next round.** (1) Identity: 6 of 47 identities cover two shoppers on DEV, 26 of 77 on the
TRAIN-seed clips; it is behind one honest review and one missed theft on DEV. (2) A concealment cue from the item
cameras, still the only way to an alert tier. (3) Put-backs one camera reads and the other does not (7005).

## 2026-10-05: improvement round 2 (SIMULATED): a second tuning set, a tracker crash, and four rules that did not hold up

All numbers are simulated. TEST was not touched. No training, no new clips.

**Result: the DEV scorecard did not move, and the goal is still not met.** 17 of 20 thefts flagged for review (0.85),
4 of 22 honest shoppers reviewed (bar: about 2), pick recall 0.957, right SKU 0.910, 1.27 identities per shopper, no
alert tier. The pipeline behaves as in round 1 except for one crash fix.

**What was kept.**

1. **A crash in the floor tracker** (`FloorTracker._colours`). When clothing colour put a swap right, the other
   open meetings of those two people were dropped, and the loop then tried to drop them a second time
   (`ValueError: list.remove(x): x not in list`). No DEV clip has that situation. 2 of the 9 TRAIN-seed clips
   (4906, 4951) do: the pipeline died on them. A TEST clip could have hit it in the final run.
2. **A second tuning set: `make bench-train`.** The 9 TRAIN-seed clips of `make shelf-clips` are complete clip
   folders (people cameras, register feed, truth), so the benchmark can run and score them. They were never used to
   tune the join or the ledger (the per-camera shelf rules were tuned on 4900 to 4902). Their scenes are harder on
   purpose (a pair at the same shelf, swaps).
3. **`make bench-whatif`** (`scripts/bench/whatif.py`): tracking, association and the ledger again on the stored
   shelf events and person boxes of both sets with other settings, one line per set, in about a minute. It refuses
   the test split.

**The same pipeline on the TRAIN-seed clips** (`results/bench_train.md`: 653 s, 53 shoppers, 17 thieves, 36 honest,
106 picks, 19 stolen items):

| | DEV (6 clips) | TRAIN-seed (9 clips) |
|---|---|---|
| theft recall, alert or review | 0.85 (17/20) | 0.684 (13/19) |
| reviews on honest shoppers | 4 of 22 | 4 of 36 |
| pick recall | 0.957 | 0.915 |
| pick precision | 0.848 (79 PICK events) | 0.866 (112) |
| right SKU of paired picks | 0.910 | 0.938 |
| right slot of paired picks | 0.851 | 0.825 |
| right shopper of paired picks | 0.940 | 0.979 |
| identities per shopper | 1.27 | 1.453 |
| identities covering two shoppers | 6 of 47 | 26 of 77 |

Read this as: the 0.85 on DEV is partly the result of two rounds of rules chosen while looking at DEV. On clips
nobody tuned the join on, theft recall is 0.68. Identity is the visible difference (26 of 77 identities cover two
shoppers). Expect TEST to land between the two, not at the DEV number.

**Stage picked and what the cases showed.** The goal that fails is reviews on honest shoppers, so the cases read were
the 4 honest reviews on DEV and then the unpaid items of every review on both sets. Root cause in one sentence: a
false "unpaid" item is almost never a phantom take, it is a real take whose put-back or payment was not credited
(14 of 19 false unpaid items on DEV and TRAIN-seed sit on a real act, 5 on none), and the events that would credit it
(put events) cannot be told from the hand leaving the shelf: 41 of 92 put events with the item seen in the hand
match a real put-back (11 of 24 on DEV, 30 of 68 in the cached TRAIN-seed shelf events of `make shelf-eval`).
Evidence strength does not separate true from false unpaid items either (camera count, detector frames, cue, window
length were tabulated: no usable split).

**Tried on both sets and reverted** (thefts flagged / honest shoppers reviewed; code not kept):

| rule | DEV | TRAIN-seed |
|---|---|---|
| as it stands | 17/20, 4/22 | 13/19, 4/36 |
| a second reading inside the first one's time window is the same reach | 17, 4 | 13, 4 |
| a person out of view pays for the width of the guess of where they are (log term in the association cost) | 17, 4 | 13, 5 |
| both of the above | 17, 3 | 13, 5 |
| a put with the item seen in the hand returns the take from any slot that reach was read at | 16, 2 | 11, 3 |
| the same with the two rules above | 14, 2 | 11, 4 |
| any put does that (with the first rule on) | 11, 1 | not run |
| an ambiguous pick is not marked down once everyone who could have paid for it has left | 17, 5 | 15, 6 |

The fourth row meets every bar on DEV (0.80 recall, 2 of 22). It was not taken: 2 of its 7 uses on DEV were real
put-backs, it erases 1 stolen item on DEV and 2 on TRAIN-seed, and with the other two rules on it drops to 14. It is
the rule round 1 already rejected in another form. Every rule in the table moves an error from one column to the
other; none removes one on both sets.

**What cannot improve at the join or the ledger, with counts.**

- Put-backs: 41 of 92 item-in-hand put events are real (above). The time between a camera's take and its put of
  the same slot does not separate them on DEV (57 such pairs, 2 under 2 s, the rest from 2.1 s up, the real
  put-backs at 3.5 to 6.1 s in the middle of them).
- Concealment: still no cue, alert tier 0 of 20 (DEV) and 0 of 19 (TRAIN-seed).
- Two of the 4 honest reviews on DEV need a per-camera fix: one take read with the time its slot was first hidden
  (6.3 s before the item left, 7003), one put-back the cooler camera that read the take never reported (7005).

**Tests.** Full suite on the final code: 407 passed, 1 skipped, none failed (counted from the progress lines of
`out/bench/round2/tests_full.log`, which has no closing summary line), plus 1 new test for the tracker crash in
`tests/test_shelf_store.py`, which fails without the fix (that file: 10 passed).

## 2026-10-05: improvement round 1 on DEV (SIMULATED): receipts, puts, parties

All numbers are from the six simulated DEV clips (20 stolen items, 22 honest shoppers). TEST was not touched. The
shelf events and person boxes of the round 0 run were reused (`make bench-dev BENCH_ARGS=--keep`): nothing before
tracking changed. Round 0 is kept in `results/bench_dev_round0.md`.

| | round 0 | round 1 |
|---|---|---|
| theft recall, alert or review | 0.80 (16/20) | 0.85 (17/20) |
| theft recall, alert tier | 0/20 | 0/20 |
| reviews on honest shoppers | 4 of 22 | 4 of 22 |
| pick recall | 0.957 | 0.957 |
| pick precision | 0.744 (90 PICK events) | 0.848 (79) |
| right SKU of paired picks | 0.910 | 0.910 |
| identities per shopper | 1.27 | 1.27 |
| receipts credited to the true payer | 17 of 29 | 28 of 29 |

The goal is not met: 4 honest shoppers in 22 get a review (the bar is 1 in 10). They are review tier, there is no
alert tier at all.

**Stage picked.** The funnel's largest loss is "conceal or pay classified" (17 stolen items): no camera reports a
concealment. That stage does not decide the review tier, though. With the true concealment times fed in
(`results/assoc_dev_oracle_conceal.json`) theft recall at alert or review stays 0.80. What decided the misses and the
honest reviews was the ledger getting wrong inputs, so that is where the cases were read.

**Root cause, one sentence.** The register feed stamps a receipt 1.5 to 4.5 s after the payment, when the payer has
left the counter and the next shopper stands there, and the ledger gave the receipt to whoever stood at the counter
at the stamp (or to whoever held the same product), so 12 of 29 receipts went to the wrong shopper; the join then
passed every shelf event to the ledger although most puts and all slot-watch-only takes are false.

What was changed (each is a rule, no clip is named anywhere):

1. **Receipts go to who was served** (`LedgerConfig.pos_lag_s`, set to 1.5 to 4.5 s in `bree.shelf.store`, the
   delay written in `scripts/bench/render_clip.mjs`). Candidates are the people at the counter between stamp minus
   4.5 s and stamp minus 1.5 s; time is ranked before basket content; a receipt cannot match an item taken after its
   stamp. Receipts on the true payer: 17 to 26 of 29, thefts 16 to 17.
2. **Which shelf events reach the ledger** (`bree.events.shelf.confirm_puts`). A take from the slot watch alone is
   dropped (0 of 6 such events matched an act on the TRAIN-seed clips of `make shelf-eval`, all 6 on DEV were
   unpaired). A put counts only when the item was seen in the hand going in or it lands within 0.25 m of where that
   person took something still out (4 of 23 one-cue puts were true on the TRAIN-seed clips). A put that fits two
   people about equally goes to the one who took from that place. A put at the place of a take returns that item
   even when the two readings name different products. A count above 1 is one unit (0 of 5 right on DEV, the shelf
   stream had asked for this).
3. **One reach, measured along the floor** (`one_act_per_reach`): two readings of one reach at two shelf heights of
   one bay or cooler door were 1.1 m apart in 3D and stayed two takes. The merge is repeated until nothing changes,
   because two readings of one reach can first go to two people.
4. **A paid item nobody saw taken explains one unpaid pick** (`LedgerConfig.misread_factor` 0.5): vision named the
   product wrong. Concealed items are never discounted.
5. **A party walks together** (`bree.events.shelf.parties`, ENTER meta `party`): entering within 4 s is no longer
   enough, the two must stay within 1.5 m for half of their time on the floor. No DEV shopper pair does. Before,
   strangers were pooled and the scorer credited a pooled review to whichever of them had more boxes.

What-ifs on the stored shelf events and person boxes, in the order built ("pooled": strangers who enter within 4 s
are one party, as in round 0):

| step | thefts flagged, pooled | honest reviewed, pooled | thefts, not pooled | honest, not pooled | pick precision |
|---|---|---|---|---|---|
| round 0 | 16 | 4 | 15 (measured in round 0) | 7 (measured in round 0) | 0.744 |
| 1 receipts | 17 | 4 | not run | not run | 0.744 |
| + 2 puts and count (before the slot watch rule) | 16 | 3 | 17 | 5 | 0.761 |
| + 2 slot watch alone dropped | 16 | 3 | 17 | 4 | 0.817 |
| + 3 reach along the floor | 16 | 3 | 17 | 4 | 0.838 |
| + 4 misread factor | 15 | 2 | 17 | 3 | 0.838 |
| + 5 parties, merge repeated: the final setting | | | 17 | 4 | 0.848 |

The last step went from 3 to 4 honest reviews: the "not pooled" rows above still pooled people with the same
entry time, and one honest shopper (7005) had been inside such a pool.

Checked on the final setting: `misread_factor` 1.0 gives 17 thefts and 5 honest reviews. `pos_lag_s` (0, 0) gives 16
thefts and 2 honest reviews, which would read as meeting the bar. It was not taken: in that setting receipts go to
the wrong shopper (18 of 29 right in the same what-if one step earlier) and unclaimed receipts of other shoppers
pay for the false items of honest ones. Lag windows of 1 to 5 s and 2 to 4 s gave the same 17 thefts and 3 honest reviews as 1.5 to 4.5 (run one step
earlier, before the party rule).

Tried and reverted (each made DEV worse or was right too rarely):

- A person walking in at the door never takes over an identity lost further inside: fixed the 7001 case, but
  identities covering two shoppers went from 6 to 9 and thefts from 17 to 16.
- "Where they took it" measured along the floor, 0.25 to 1 m: thefts fell to 13 and then 11 (false puts erased
  stolen items).
- A both-cue put that names nothing in the basket returns the nearest item taken within 1 m: 16 thefts and 2 honest
  reviews, but only 2 of its 6 uses were real put-backs and 3 stolen items were erased. Not kept.
- Pick confidence 0.6 or 0.7 for one-cue takes, association groups of 3 or 4 s, reach window of 5 to 8 s: no change
  in flags (the longer reach window lowers pick recall to 0.943).

### What is left (DEV, round 1)

- **3 missed thefts.** 7005 P007: the pick was never seen by any camera's cues (lost at "hand or item detected").
  7001 P001: the identity that took the item was handed to the next shopper who came in 7 s later, so the item sits
  on an honest shopper. 7005 P001: a counter item seen by the pixel comparison alone, a 4 s stop at the counter with
  no receipt, scored 0.27 after the "register visit without receipt" discount.
- **4 honest shoppers reviewed**, each from a different vision error: the 7001 identity hand-over above; a take
  read 0.7 m from its true slot with the wrong product, so its correct put-back did not match (7003); a put-back
  read at the neighbouring cooler door (7005); a take read twice 4.2 s apart plus a take 5 m from a shopper who was
  out of view (7006).
- **Put-backs are the weakest input.** 10 of 43 put events on DEV matched an act (0.385 on the TRAIN-seed clips).
  Two of the four honest reviews are a real put-back the pipeline could not use. This is in `bree.shelf` (per
  camera), not in the join.
- **No alert tier.** Still no concealment cue.
- **Tests:** the full suite ran on the final code: 407 passed, 1 skipped, none failed (counted from the progress
  lines of `out/bench/round1/tests_full.log`; the log has no closing summary line). 9 tests are new
  (`tests/test_ledger.py`, `tests/test_association.py`).

## 2026-10-05: fixed benchmark and baseline (SIMULATED)

Everything here is simulated: 12 clips from the browser store simulator copy on the recommended-3d-45 layout.
Nothing is real footage. Command: `.venv/bin/python -m bree.sim.bench dev --runner bree.sim.bench:run_pipeline --name baseline`,
result in `results/bench_dev_baseline.md` and `.json`. The TEST split was not run except a 60 frame smoke check on
one clip; no TEST number is recorded.

DEV split: 6 clips (seeds 7001 to 7006), 421 s of simulated time, 37 shoppers (15 thieves, 22 honest), 70 true
picks (41 paid, 20 concealed, 9 put back; 46 at gondolas, 15 in the cooler, 9 at the counter), 18 to 20 cameras
per clip.

Baseline: the per-camera engine as committed at 19f51dc (sim_sku detector, camera nodes and hub, closed-world
identity, two-view slots).

| Metric | Baseline, DEV |
|---|---|
| Theft recall, alert tier | 0 of 20 stolen items |
| Theft recall, alert or review | 0 of 20 |
| Alerts, reviews | 0, 0 (so alert precision is undefined, and no false alert on the 22 honest shoppers) |
| Pick recall | 2 of 70 (2.9 percent) |
| Pick precision | 2 of 2 PICK events |
| Right SKU | 1 of the 2 paired picks, 1 of 70 true picks |
| Right slot | 1 of 2 paired picks |
| Time to alert | none (no alert) |
| Store-wide identities per real shopper | 4.0 (148 identities for 37 shoppers; 93 of them cover two shoppers; 1 shopper never tracked) |

Funnel, all 70 true picks, counted at the first stage each one fails:

| stage | reached | lost here | passed on its own |
|---|---|---|---|
| in view of a rendered item camera | 70 | 0 | 70 |
| frame reached the pipeline | 70 | 0 | 70 |
| hand or item detected | 70 | 21 | 49 |
| shelf event emitted | 49 | 47 | 2 |
| right slot | 2 | 1 | 1 |
| right SKU | 1 | 1 | 1 |
| associated to a shopper | 0 | 0 | 2 |
| right shopper | 0 | 0 | 2 |
| conceal or pay classified | 0 | 0 | 1 |
| ledger basket | 0 | 0 | 50 |
| alert | 0 | 0 | 50 |

Reading: the frames arrive and in 49 of 70 picks something is detected on the hand or the item (item box on 46,
right SKU on 44, person box on the picker on 41, wrist near the hand on 32), but only 2 PICK events come out. The
loss is at the shelf event stage (47 picks), then at detection (21, of which 20 at gondolas). The 50 in the last
two rows are paid and put-back picks for which no alert is the right answer; with zero alerts that says nothing.
None of the 20 concealed picks produced a PICK event.

Pipeline wall time per clip, on a loaded 10 core laptop, 3 clips at a time: 11 to 20 minutes (camera nodes and
hub 200 to 520 s, pipeline 480 to 815 s).

## 2026-10-05: shelf events without a person box (SIMULATED, TRAIN-seed clips)

Goal: rail, cooler and counter cameras report takes and puts per slot without a person box, so a pick no longer
needs a person and an item in one view. Code in `src/bree/shelf/` (`diff.py` pixel comparison, `slots.py` slot
watch, `hand.py` item in a hand, `events.py` fusion and the shared contract). Everything below is from simulated
clips rendered with the benchmark generator on TRAIN seeds. No DEV or TEST clip was used.

Clips: rules were tuned on 4900 to 4902. Held out are 4903 to 4906 plus 4950 and 4951, six clips. 4950 and 4951
were added after a review found the first four too favourable. The review changes (shift and brightness
handling, repeat drop) were made without looking at held-out results, and all nine clips were then run once.

Unit: one truth pick and one item camera that saw the picked item at 20 px or more. Found: that camera reported a
take within 3 s and 0.6 m. Right slot: the exact slot id. Slot or adjacent: the right slot, or the facing next to
it (0.1 m or less) with the same SKU. Shares are of all pairs, a miss counts as wrong. Precision: share of events
near any truth act. Strict precision: one event per act and camera. Repeats: further events of a camera near an
act it already matched.

Held out, six clips, `make shelf-eval`, `out/shelf/eval_heldout6.md`:

| camera kind | act | pairs | recall | right slot | slot or adjacent | right SKU | right count | events | precision | strict precision | repeats | time error median s | p90 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shelf | take | 62 | 0.919 | 0.855 | 0.903 | 0.903 | 0.919 | 87 | 0.828 | 0.724 | 0 | 0.07 | 0.39 |
| cooler | take | 19 | 0.895 | 0.737 | 0.895 | 0.895 | 0.895 | 43 | 0.605 | 0.442 | 7 | 0.07 | 0.33 |
| checkout | take | 13 | 1.000 | 1.000 | 1.000 | 1.000 | 0.923 | 23 | 0.652 | 0.565 | 1 | 0.10 | 0.12 |
| all | take | 94 | 0.926 | 0.851 | 0.915 | 0.915 | 0.915 | 153 | 0.739 | 0.621 | 8 | 0.07 | 0.35 |
| shelf | put | 12 | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 | 20 | 0.500 | 0.500 | 0 | 0.23 | 0.44 |
| cooler | put | 4 | 1.000 | 0.750 | 0.750 | 1.000 | 1.000 | 21 | 0.238 | 0.238 | 0 | 0.38 | 1.05 |
| checkout | put | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 11 | 0.455 | 0.455 | 0 | 0.27 | 0.45 |
| all | put | 21 | 0.857 | 0.810 | 0.810 | 0.857 | 0.857 | 52 | 0.385 | 0.385 | 0 | 0.23 | 0.58 |

Views fused per act (`fuse_views`): 70 picks, recall 0.957, right slot 0.871, right SKU 0.943. 167 fused events,
97 match a truth act.

By clip set (takes):

| run | pairs | recall | right slot | slot or adjacent | precision | strict precision |
|---|---|---|---|---|---|---|
| held out, six clips | 94 | 0.926 | 0.851 | 0.915 | 0.739 | 0.621 |
| of which 4903 to 4906 | 59 | 0.949 | 0.915 | 0.949 | 0.753 | 0.645 |
| of which 4950 and 4951 | 35 | 0.886 | 0.743 | 0.857 | 0.717 | 0.583 |
| tuning clips 4900 to 4902 | 41 | 0.829 | 0.805 | 0.829 | 0.763 | 0.513 |
| all nine clips | 135 | 0.896 | 0.837 | 0.889 | 0.747 | 0.585 |

Acceptance (pick recall and right slot at least 0.85 on held-out TRAIN-seed clips), stated plainly:

- Recall is met on the six held-out clips: 0.926 over 94 pairs.
- Right slot sits exactly at the bar: 0.851 over 94 pairs (80 right). It is not a comfortable pass. On the two
  newest clips alone it is 0.743, on all nine clips 0.837.
- Cooler cameras are under the bar on right slot: 0.737 over 19 pairs held out, 0.724 over 29 on all nine.
- Of the 7 wrong slots held out, 6 are the neighbouring facing with the same SKU, seen at 20 to 52 px. For the
  ledger (which counts SKUs) those are right: right SKU is 0.915. For slot-level stock they are wrong.
- The neighbour-facing error was not fixed. The slot is still chosen by overlap of the changed patch with each
  slot's visible pixels.

What else the numbers say:

- Precision is the weak side. Strictly counted, about 4 take events in 10 and 6 put events in 10 match no act.
  An association step should not turn an unconfirmed put into a PUT_BACK on its own.
- Repeats. One camera no longer reports the same or the neighbouring facing twice within 4 s. The repeats that
  remain (8 held out, 13 on the tuning clips) are other slots 0.14 to 0.4 m from the act (checked on the tuning
  clips), most in clip 4902 where the camera sways. They would be phantom picks in the ledger; `fuse_views`
  does not merge events of one camera.
- Idle cameras (rendered item cameras that see no act): 18 cameras, 22.4 camera minutes, 5 events, 0.22 events
  per camera per minute held out.
- Limit of this measure: the renderer keeps mostly item cameras that see a pick (10 of 12 in clip 4903), and
  about 30 item cameras of the layout are never rendered. Store-wide false event counts will be higher than
  these tables suggest. Going by the idle rate, 30 more cameras would add about 7 false events per minute, but
  that rate rests on 5 events. A clip rendered with every item camera is needed (`render_clip.mjs
  --max-item-cams 100 --spare-cams 100`, then `eval_shelf.py <seed> --tag _allcams`). I started one and stopped
  it: with the other jobs on the laptop it would have taken over an hour. Request to the benchmark agent.
- Unit count: every pick in these clips is one unit. 5 of 320 events on the nine clips carry a count above 1.
  Counts above 1 are covered by a unit test only. Downstream should treat count above 1 as unconfirmed.
- Timing: the take time is within 0.1 s of the truth for the median found pick, within 0.35 s for 9 in 10.
- Tuning clips score lower than held out. Clips differ a lot one to the next; nine clips is still a small set.

Shift and light (`make shelf-robust`, `out/shelf/robust_4903.md`; clip 4903, pixel comparison alone, the
disturbance applied to the decoded frames of every item camera):

| disturbance | pairs | recall | right slot | take events | precision | before this change |
|---|---|---|---|---|---|---|
| none | 13 | 0.923 | 0.923 | 23 | 0.696 | 24 events, precision 0.667 |
| picture shifted 3 px from frame 50 | 13 | 0.923 | 0.846 | 25 | 0.560 | 148 events, precision 0.128 (reviewer's run) |
| picture shifted 8 px from frame 50 | 13 | 0.923 | 0.923 | 26 | 0.615 | not measured |
| dimmed to 30 percent over 30 s | 13 | 0.923 | 0.923 | 24 | 0.708 | recall 0.615, 94 events (reviewer's run) |
| 10 percent brightness step | 13 | 0.923 | 0.923 | 22 | 0.727 | 12 of 13 found (reviewer's run) |
| brightened by 60 percent | 13 | 0.923 | 0.923 | 48 | 0.312 | not measured |

- A shift of the whole picture is estimated against the reference (phase correlation) and undone before
  comparing. Brightness is undone for gains from 0.25 to 4 (it was 0.8 to 1.25, with a silent fallback).
- Strong brightening is still weak: picks are found, but false takes double (48 against 23), because bright
  areas clip at white. The camera reports `brightness_changed`, so the association step can discount it.
- When more than half the picture differs from the reference the camera reads nothing and says `unreliable`;
  after 1 s without motion it takes a new reference and says `reference_reset`. On the nine clips this window
  opened 87 times (not inspected one by one; a shopper close to the camera is the likely cause), closed by
  itself 85 times (`reliable`) and by a new reference twice.
- A swaying camera is not fixed by this: clip 4902 still gives 67 events.

Not done:

- `point_3d` is the planogram slot face, not a two-view triangulation. `point_sigma_m` is now half the slot
  width instead of null.
- `evidence.before` and `.after` are null unless an evidence folder is passed (`run_clip(evidence_dir=...)`,
  `--evidence DIR`). The evaluation runs did not pass one.
- The hand cue is a skin blob or the detector's hand class; no better hand detector was dropped in.

Wall time: 493 s for 38,010 camera frames (clips 4903 to 4906, 3 processes, laptop shared with other jobs).
No "NMS time limit exceeded" warnings in these runs.

## 2026-10-05 Hand and held-item detector (stream hand-detector, SIMULATED)

Everything in this section is on simulated frames from the browser store simulator: one store model, invented
package art, block-shaped hands. None of it is accuracy on real footage.

### What was built

- `scripts/train/sim/hands.js` + `scripts/train/render_hands.mjs`: frames of the benchmark's scripted behaviour
  (reach, look, hold, put back, conceal, counter, two shoppers at one shelf) in randomised scenes, rendered on the
  item cameras that see the hand, at the cameras' real resolutions (2688x1520, 1520x2688, 1920x1080). Each frame has
  every item box with its SKU, and a box for every visible hand with `reaching` and `holding`.
- 42 training scenes (seeds 2000 to 2041, 7,894 frames) and 10 held-out scenes (seeds 3000 to 3009, 1,936 frames),
  all inside the benchmark TRAIN range, none shared with the first detector, dev or test.
- `bree.train.dataset`: hand class, held-out seed range, crops around held items and reaching hands, frame stride,
  parallel build. `bree.train.train`: fine-tuning with an added class keeps the old classes' outputs.
- `bree.train.backend`: weights picked by version name, `SkuDetector.detect` returns items and hands apart, optional
  second look centred on the hands, `BREE_SKU_ROI=full`. `sim_sku.pt` is unchanged and still the default.
- `bree.train.eval_hands`: the evaluation below. `tests/test_hand_detector.py`.

### Results on the held-out frames

484 whole frames (every fourth of the held-out render), tiled inference on the whole frame at confidence 0.25, no
person box. Command: `make hands-eval`. Full tables: `results/hand_detector.md`.

| measure | sim_sku (old) | sim_sku_hands_v2 | sim_sku_hands_v3 | v3 + second look |
|---|---|---|---|---|
| item in a hand, 20 px or more: right SKU (target 85%) | 67.0% (235 of 351) | 74.1% | 87.5% (307 of 351) | 88.3% (310 of 351) |
| item in a hand, 20 px or more: found | 75.2% | 78.9% | 90.6% | 91.7% |
| hand on a reaching arm found, IoU 0.5 (target 90%) | 0% (no hand class) | 56.8% | 76.4% (230 of 301) | 79.1% (238 of 301) |
| a hand box holds the true hand centre, reaching arm | 0% | 65.5% | 83.4% | 84.7% |
| hand boxes that are a hand | n/a | 80.7% | 81.2% | 80.2% |
| item on the counter, 20 px or more: right SKU | 65.9% | 70.5% | 76.1% | 77.3% |
| shelf item, clear, 20 px or more: right SKU | 97.9% | 98.6% | 99.0% | 99.0% |
| shelf item, partly hidden: right SKU | 95.7% | 97.0% | 97.8% | 98.0% |
| item behind the front one: right SKU | 90.2% | 93.2% | 95.3% | 95.5% |

Item in a hand, right SKU by pixels across a 6.6 cm can:

| px | items | sim_sku (old) | sim_sku_hands_v3 |
|---|---|---|---|
| 30-40 | 17 | 35.3% | 70.6% |
| 40-60 | 122 | 55.7% | 87.7% |
| 60-100 | 148 | 83.8% | 93.9% |
| 100+ | 61 | 59.0% | 78.7% |

Held-out tiles, v3: mAP50 0.973, mAP50-95 0.872 over the 34 classes; the lowest SKU class is 0.956 AP50; the
hand class is 0.848 AP50 (0.645 AP50-95). Per class: `hand_detector.md`.

Speed, v3, measured while other jobs were running on the machine: 11.8 ms per 640 px tile and 51 ms per 4MP frame
(15 tiles) on MPS; 65.5 ms per tile and 544 ms per frame on CPU. The same as the old weights (51 ms and 556 ms).

Near-identical variants held in a hand, v3: 3 of 123 found bars given a sibling's SKU (cocoa_crest family), 3 of 34
for the relieva pair, 0 in the other eight families that occur.

### Acceptance

- Right SKU for an item in a hand at 20 px or more, target 0.85: **met**, 0.875 (0.883 with the second look).
  The old weights score 0.670 on the same frames.
- Hand recall on reaching arms, target 0.9: **not met**, 0.764 (0.791 with the second look).

What limits the hand recall, measured:

1. Size. By the short side of the hand box, v3 finds 94.6% of reaching hands of 60 px or more (175 of 185), 64.7% at
   30 to 60 px (44 of 68), 30.6% at 15 to 30 px (11 of 36) and none under 15 px (0 of 12). The 48 hands under 30 px are
   16% of the reaching hands; they are far from the camera or mostly behind the item or the shelf (a hand counts as a
   label from 30 visible pixels).
2. Training time. The hand class starts from zero. 15.5 minutes of training gave 56.8%, 44 more minutes gave 76.4%, and
   the val recall of the class was still rising (0.49 to 0.66). A further 3 epoch run (v4) was started and stopped at
   79% of its second epoch, because other jobs on the shared machine had slowed it to between 2 and 12 seconds a step.
   It was not evaluated and no v4 weights were written.
3. Confidence threshold. At confidence 0.1 instead of 0.25, v3 finds 81.1% of reaching hands at IoU 0.5 and a hand box
   holds the true hand centre for 89.4%, while the share of hand boxes that are a hand falls from 81.2% to 64.9%
   (`eval_hands --weights sim_sku_hands_v3 --quick --conf 0.1`, not saved in the repo).

What limits the held item: how much of it shows. v3 has the right SKU for 96.5% of held items that are 80% or more
visible, 92.4% at 50 to 80% and 72.9% at 25 to 50% visible. Close-up items (100 px or more across a can, often larger
than the 128 px tile overlap) are at 78.7%.

### Not done

- The new weights were not run through the pipeline or the benchmark clips; that is the integrator's and the
  benchmark's step. Dev and test clips were not opened by this stream.
- No frames of the first dataset were re-rendered with hand labels, so the 5,205 frames of seeds 1000 to 1239 are not
  part of the new training set.
- The tile mAP of the old weights on the new tiles is not reported (its class list has no hand class).

## 2026-10-05: association stream (store-wide people, who took it, shelf events to the ledger)

**Bottom line.** Shelf events can now be attached to a shopper without a person box in the item camera, and identity no
longer falls apart. On the simulated clip of the first end to end run (`data/synth/clip_5001_door`, TRACK-2 and TRACK-3)
the 4 shoppers get 4 identities, each born at the door and each seen leaving, plus one identity for the clerk (was 13
people, 1 exit seen). On scripted scenes (SYNTHETIC) the right shopper is named for 450 of 462 shelf events (0.974)
and none of the wrong ones is given without the uncertain mark. Fed the TRUE picks of the 6 DEV clips as shelf events
(SIMULATED, an upper bound for this stage, not a pipeline result), the chain puts 16 of 20 stolen items in front of a
reviewer with 1 review on an honest shopper; the recorded baseline found 0 of 20. Alert tier needs a conceal cue from
an item camera: the overhead pose does not show concealment in this simulator (measured below).

### Why identity fragmented (20, then 13 people for 4 shoppers)

Read off the old run's logs (`data/synth/clip_5001_door/e2e/sim_eval/pipeline`):

| cause | evidence | fix |
|---|---|---|
| Per-camera track ids were the unit of identity. Every new ByteTrack id was a new placement decision. | TRACK-2 made 22 track ids and TRACK-3 18 for 4 shoppers; 43 hand-offs, 40 uncertain marks | One tracker in floor metres over all people cameras (`bree.track.floor`). No per-camera ids. |
| A track that appeared while its person's other track was still held counted as an entry nobody saw. | 9 of 13 people were "missed entry" births: TRACK-2 4, TRACK-3 3, rail cameras 2 | Births only at the door or in the first second. A new track elsewhere takes back a lost identity, or waits. |
| Rail cameras took part in identity with floor points from a view that looks along the aisle. | 2 births from G1R-rail-3 and G2L-rail-4 | Identity comes from overhead and entrance cameras only. |
| Floor position from the bottom of the box. | DEV 7001: median error per camera 0.14 to 0.63 m (p90 up to 3.7 m); hips at 0.93 m: 0.09 to 0.17 m | Placed by hip keypoints, then shoulders; a box without them can keep a track going but not start one. |
| A second box of one person became a person. | the old rule waited 2 s and then made a person; 2 second-box frames on the door clip now | A new track on top of a tracked person is not confirmed; an inside birth needs steady detection for 1.5 s. |
| Someone leaving and someone entering right behind shared an identity. | DEV 7003: a shopper left at 28.8 s, the next entered at 29.4 s and took the id | Last seen in the doorway or past it: left at once, never handed back. |
| Exits were not seen, so nobody was ever reconciled. | 1 exit of 4 | 4 of 4 on the door clip; 39 exits counted for 37 true on DEV. |

### Identity (SIMULATED clips; person boxes and keypoints from the pipeline's frame logs, tracker from pixels only)

| | before | now |
|---|---|---|
| door clip: people made for 4 shoppers | 13 (4 at the door, 9 inside) | 4 shoppers + the clerk: 1.0 per shopper (1.25 counting the clerk) |
| door clip: exits seen | 1 of 4 | 4 of 4 |
| DEV: identities per shopper (37 shoppers) | 4.0 (148 ids; baseline run) | 1.08 (34 of 37 have exactly one, none untracked) |
| DEV: identities covering two shoppers | 93 | 20, of which 2 are not marked uncertain |
| DEV: floor position error, median | 0.41 m (box placement, first version of this tracker) | 0.064 m; 81% of shopper-frames covered |

What is left: 20 identities on DEV still cover two shoppers at some point. The simulated shoppers walk through each
other, and no tracker that uses position only can keep two people apart who pass through one point.
`FloorConfig.encounter_m` (0.25 m) marks both identities uncertain when two come that close: 18 of the 20 mixed
identities carry the mark. The price is that 29 shopper identities on these clips (47 identities were made in all, 7 of them on a clerk) are marked uncertain and
so capped at review. With `encounter_m=0`, 5 are marked and 17 mixed identities go unmarked.

### Who took it

| test | events | right shopper | wrong and sure | marked uncertain |
|---|---|---|---|---|
| Scripted scenes, scripted tracks (SYNTHETIC, 200 scenes) | 462 | 450 (0.974) | 0 | 114 (12 of them wrong) |
| of those: second shopper at the same shelf, 0.7 to 1.3 m along | 112 | 110 | 0 | 22 |
| of those: shoulder to shoulder, 0.15 to 0.35 m (nobody could tell) | 87 | 79 | 0 | 80 |
| of those: across the gondola / passer-by / alone | 263 | 261 | 0 | 12 |
| Scripted scenes through the floor tracker (SYNTHETIC, 200 scenes) | 462 | 446 (0.965) | 0 | 229 |
| DEV clips, true picks as shelf events, tracks from pixels (SIMULATED) | 70 | 66 (0.943) | 0 | 28 (24 of them right) |
| same with `encounter_m=0` | 70 | 66 | 0 | 12 (8 of them right) |

In the scripted scenes through the tracker, 780 identities were made for 753 shoppers (1.04); 90 cover two shoppers at
some point and 3 of those are not marked.

### To alerts, with the TRUE picks and put-backs as shelf events (SIMULATED DEV, 15 thieves, 20 stolen items)

This isolates everything after the shelf-events stream. It is not an end to end result.

| | alert | alert or review | alerts (false) | reviews (on honest shoppers) |
|---|---|---|---|---|
| recorded baseline (per-camera engine) | 0 of 20 | 0 of 20 | 0 | 0 |
| no conceal cue | 0 | 16 of 20 | 0 (0) | 12 (1) |
| true conceal times as cues | 6 | 16 of 20 | 4 (0) | 8 (1) |
| true conceal times as cues, `encounter_m=0` | 13 | 16 of 20 | 9 (0) | 3 (1) |

All 4 missed thefts were attached to the right shopper at the pick, without the uncertain mark. Three were flagged for
review anyway, but under an identity that had become another shopper by the exit (its basket holds the picks of two
shoppers), so the scorer counts the review for that other shopper. The fourth identity was never seen leaving, so it
was never reconciled. Identity after the pick is what is left to fix, not the association.

### Concealment is not visible from above in this simulator

DEV 7001 to 7003, overhead and entrance cameras. Wrist to hip distance over torso length (10th / 50th / 90th
percentile): concealing 0.11 / 0.18 / 0.25, walking with an item 0.08 / 0.20 / 0.36. The true hand of a walking shopper
is in the same place with or without an item (0.26 m from the body axis, 0.92 m high). So the conceal cue has to come
from an item camera; `store_events(conceal=...)` takes it and the ledger raises the tier as before.

### Limits

Simulated and synthetic only. The DEV numbers are from the clips used for tuning; TEST was not touched. The person
boxes came from the baseline run's frame logs (the repo's detector and crop pose after ByteTrack); `bree.track.people`
makes the same boxes without the tracker (738 against 692 on the first 260 frames of 7001 TRACK-1) but a full run with
it was not timed. The stand distance in the simulator is exact (0.42 m gondola, 0.54 m cooler, 0.56 m counter, 0
along the shelf); the association uses 0.5 m with 0.25 and 0.3 m spreads so it does not depend on that. Wrist lines of
sight are used when an overhead camera has the wrists; their effect on its own was not measured.

Reproduce: `make assoc-test assoc-scripted assoc-door assoc-dev`; the variants with
`python -m bree.track.floor_bench dev --chain [--oracle-conceal] [--encounter-m 0]`. Results as run:
`results/`.

## 2026-10-05: fuel drive-off module (plates stream), SYNTHETIC results only

The live site lists drive-off capture. This is the first version of it: a plate detector and text reader behind
an interface, a pump-zone tracker, the drive-off rule, plate retention, and legal notes. All numbers below come
from `scripts/plates/bench.py` (seed 20261005; full run 779 s for the reader table, then a `--scenarios-only`
rerun of 215 s for the vote calibration and the timelines after the verifier fixes) on plates drawn with system fonts on crude car shapes and
on scripted timelines. Nothing was measured on real footage, because there is none. The full tables are in
`results/plates_bench.md`.

### What was built

- `src/bree/plates/reader.py`: open plate detector (open-image-models 0.6.0, YOLOv9-t, ONNX) and open text
  reader (fast-plate-ocr 1.1.0, ONNX), both MIT packages that downloaded with no sign-up, plus an OpenCV-only
  plate localiser. One call: `make_reader().read(frame, roi=vehicle_box)`. `vote()` merges reads over frames.
- `src/bree/plates/pump.py`: `DriveOffMonitor`. Vehicle boxes in, dispenser sales and payments in, review
  alerts out through the existing `Alert`, `AlertSink` and `ReviewStore`. Retractions on late payment.
- `src/bree/plates/retention.py`: `PlateStore`. 72 hours by default, 30 days once confirmed, deleted at once on
  "not theft" or late payment, logged lookups, automatic purge.
- `README.md`: usage, retention table, licences with open questions, per-state legal notes.
- `tests/test_plates.py`: 45 tests (44 need no model weights).

### Plate read accuracy (SYNTHETIC), whole plate right

Text reader alone on a plate crop, 150 crops per cell:

| plate width px | day | day + motion blur | night | night, heavy noise |
|---|---|---|---|---|
| 32 | 3.3% | 4.0% | 0.0% | 0.0% |
| 48 | 62.0% | 72.0% | 54.0% | 0.0% |
| 64 | 88.7% | 71.3% | 82.0% | 0.0% |
| 96 | 96.7% | 88.7% | 90.7% | 0.0% |
| 160 | 100.0% | 98.7% | 98.0% | 16.7% |

Character accuracy in the same cells: 49% at 32 px in daylight, 91% at 48 px, 98% at 64 px, 99% at 96 px,
100% at 160 px; at night 39%, 89%, 95%, 99%, 100%. Voting over 5 frames lifts 48 px from 62% to 87% whole plate
in daylight and from 54% to 97% at night (30 plates per cell).

Detector plus reader inside the vehicle box, 30 frames per cell, whole plate right (open / OpenCV / both):

| plate width px | day | day + motion blur | night | night, heavy noise |
|---|---|---|---|---|
| 48 | 43% / 47% / 47% | 43% / 53% / 53% | 33% / 20% / 30% | 0% / 0% / 0% |
| 64 | 57% / 73% / 73% | 53% / 77% / 77% | 77% / 23% / 67% | 0% / 0% / 0% |
| 96 | 57% / 87% / 87% | 47% / 83% / 83% | 100% / 67% / 100% | 0% / 0% / 0% |
| 160 | 67% / 73% / 73% | 67% / 80% / 80% | 93% / 57% / 87% | 13% / 0% / 13% |

What the tables say:
- **Plate width decides.** Under 48 px nothing is readable; 64 px is the lowest width where a single frame is
  right most of the time; 96 px and up is comfortable. A US plate is 12 inches wide, so 96 px on the plate
  means about 8 px per inch at the pump. Camera placement for the forecourt should be planned from that number.
- **The heavy-noise night level fails at every width** (brightness at 10%, noise sigma 16 grey levels: the
  noise is two thirds of the whole signal). The ordinary night level (18%, sigma 10) is close to daylight from
  64 px up. A forecourt is lit, so the ordinary level is the likelier one, but that is an assumption.
- **The open detector misses a third to a half of the drawn daytime cars** and almost none at night from 64 px up. With 30 frames per
  cell the margin is wide (about plus or minus 18 points). This may be an artefact of the crude drawings.
- Text reader speed: 19.9 ms per crop on the laptop CPU with other jobs running.

### Stored plates: how often a confident vote is right (SYNTHETIC, seed 20261006)

A plate is stored only at vote confidence 0.9 or more. On 360 drawn plates (48 to 96 px, day, motion, night, 5
frames each): votes at 0.9 and up were right for 313 of 319 (98%), 0.8 to 0.9 for 9 of 20 (45%), under 0.8 for
4 of 21 (19%). So about 1 stored plate in 50 would still be wrong on this synthetic data.

### Drive-off rule on 18 scripted timelines (SYNTHETIC)

Right alerts and retractions in 18 of 18: paid inside, prepaid, drive-off, moved the car then paid inside the
grace period, late payment (alert then retraction), drive through, two pumps at once, two cars in a row at one
pump, sale reported late, camera blocked for 15 s, camera blocked past the grace period (alert then retraction,
the known limit), hidden for 6 s while fuel is flowing then pays, hidden while fuel is flowing then drives
off, track id change while fuel is flowing, drive-off with an unreadable plate followed 10 s later by a paying
car in the same spot (one alert, the second car's plate not stored), no vehicle seen, night drive-off, night
paid. Every alert was review tier.

Plates on the 6 readable drive-offs that were not retracted: with both detectors 5 read exactly right, 1 not
read, 0 stored wrong; with the open detector alone 3 right, 3 not read, 0 wrong. Retracted alerts left 0 plates
in the store. 18 timelines is a check that the rule does what it says, not a rate.

A verifier found two faults in the first version of the rule, both fixed and covered by the four new
timelines and by unit tests: a paying customer's plate could be stored on the previous car's drive-off alert,
and a short occlusion or track id change late in fuelling raised an alert on a car still at the pump.
Remaining limit: an unreadable drive-off plate followed within 60 s by a car that takes the spot and buys no
fuel delays the alert until that car leaves (the alert names the gap and stores no plate).

### What this does not show

- Nothing about real plates (embossed characters, frames, dirt, state designs, temporary tags, glare,
  headlights at night) or real forecourt cameras.
- Vehicle detection and tracking at a pump were not measured; the bench uses scripted vehicle boxes.
- No forecourt controller is connected. `FuelSale` and `on_payment` are the interface; the adapter for a real
  controller or POS feed is not written.
- The licence of the detector weights is an open question for counsel (readme.md). The legal notes are pointers
  compiled from the NCSL table and the EFF list, not legal advice, and the statutes were not read in full.

# Audit fixes (2026-10-05)

An independent audit re-ran the commands behind this report and probed the code. It reported 30 findings, 5 major and 25 minor. This is what was done about each. The corrected text and numbers are in the sections below; reasons are in DECISIONS "Audit fixes". Figures marked "audit" were measured by the audit and not re-run for this section.

**Wrong or unsupported claims, corrected.**
- **"The SKU detector found the item 69 times of 69" was a misreading of the pick funnel.** The columns are independent counts. Recounted on the stored run: of the 69 item boxes in a frame with a tracked person, the item was found on 58 (84%), all 58 with the right SKU; 11 more finds were in frames with a person detection and no track. The funnel now carries the two intersection columns and "person box" is renamed "tracked person" (`scripts/e2e_sim_chain.py --refunnel`).
- **The clip did not re-render to the same pixels, and now does.** The simulator drew the sleeve length of the clerk behind the counter from an unseeded random number, once per page load; the two overhead cameras see the clerk's forearm (152 differing pixels in the TRACK-2 frames checked, 373 in TRACK-3). The render overlay now sets it per seed. A full re-render of the recorded clip is byte-identical to the stored one in all 4,240 frames.
- **"In a hand" was "in a hand or on the counter"** wherever the 50.0% to 83.6% range is quoted; the 60.0% bucket has 5 items.
- **20 px is "not contradicted in simulation", not "validated".** The curve has no cliff at any size (100.0% at 0 to 10 px on 93 items), so it does not single out 20 px. The item's own box is the measure that shows a floor (81.7% at 10 to 15 px, 50.0% under 10 px).
- **Closed world plus re-ID does not win "on every count".** It wins every handoff, merge, split and exit count and is worse on two rows (still counted inside at the end, visits marked uncertain).
- **The MERL test split was run twice, not once** (before and after brightness normalisation; recall 425 of 425 both times).
- **The second theft was scored without its best camera** (COOLER-rail-1 is not in the clip).
- **The pipeline's error model and the simulator's agree within about 10%, not exactly** (0.891 to 0.999, audit), and the first-order prediction is about 4% low (audit).
- **Training log:** 11 to 14 minutes for the four normal epochs, not 11 to 12.
- **Timing figures depend on machine load.** Marked where they appear. Re-timed on 2026-10-05 with a system job running (load average 3 to 5): detector on MPS 8.2 to 11.8 ms per tile, on CPU 77.1 to 113.5 ms per tile (three runs each); node trigger 0.62 to 0.91 ms per frame.

**Code defects, fixed, each with a test.**
- **Tailgating at the door swapped identities** (store-wide closed world): someone walking in 0.2 or 0.4 s after someone walked out got the leaver's identity and basket, and the leaver's exit was dropped. A new track at the door now takes an existing identity only when another camera has that person in its latest frame.
- **Hub bursts (raw frames, faces not pixelated) were never deleted.** They are now deleted 24 hours after they were received (`--retain-hours`).
- **Purged keypoints stayed readable in the review database file.** The database now zeroes what it deletes and is vacuumed after a purge.
- **An exited identity kept receiving events** if its camera track came back. The track is now placed again as a new person or a hand-off.
- **Floor positions from a box cut by the frame bottom, or from a pixel above the horizon,** are no longer used: a known person keeps their last good position, a track with no floor position is not placed. On the simulated clip the identity fixes together took the people created for 4 shoppers from 20 to 13; which fix did how much was not separated.
- **Pre-roll was a frame count, not a time span;** it is now trimmed by time. **Bursts with and without a hub time were merged on one time axis;** frames without one are now dropped and counted, and two hub-time cameras need a shared `t0`.
- **`make e2e-sim` rendered a different clip and overwrote the tracked result.** It now reads the recorded run's clip (`make sku-clip-door`) and writes to `results/` only when asked.
- **`make sku-data` with one seed crashed with a KeyError;** it now says what is missing, and the README has a smoke run. **`ingest` of a missing file printed a traceback;** now one line. **`make sim-eval` printed nothing for many minutes;** it now prints a line per stage and per 100 frames.
- **The reviewer-page browser test depended on the clock** and failed once under load; it no longer does.
- **`motmetrics` was not a declared dependency and MOT16 was not fetched;** both fixed. **An absolute home path** was in a tracked result file; removed. **The simulator copy's version** is now recorded in every clip and dataset (the recorded frames came from commit a4f479f; the current simulator renders the same pixels on a 1,200 frame sample).

**Because the identity code changed, three results were re-run and replaced:** the store-wide identity tables (`results/calib_bench.json`: closed-world columns moved by one to four counts), and both end to end runs (`results/e2e_sim_chain*.json`: 13 enter events and 9 missed-entry people where there were 20 and 16; thefts caught, alerts, PICK events and the pick funnel are unchanged at 0 of 2, 0, 0 and the table below).

**Not changed, and why.** The triangulation error prediction is left about 4% low (inside the tested band; documented). Four edge slots where the simulator and the pipeline disagree on visibility by 2.4 to 3.0 px (audit) are noted only. The checkout cameras cut a standing shopper's feet in about half of positions (audit geometry check); identity should come from the overhead cameras, and the code now refuses to move a person on a cut box, but a new track first seen that way is still placed.

# Integration summary (2026-10-05)

Four streams were built in parallel and merged here. What each added, what was measured, and on what kind of data:

| stream | what exists now | measured on | headline | where |
|---|---|---|---|---|
| SKU detector | a detector that names the SKU, trained on simulator renders; `--backend sim_sku` | SIMULATED frames, held-out scenes of one store model | right SKU 97.3% for a clearly visible shelf item at 20 to 25 px across a can; 50.0% to 83.6% for items in a hand or on the counter (counted together); nothing contradicts the 20 px default | next section but one |
| calibration | camera calibration from clicked marks, 3D slot of a pick, store-wide closed-world identity | SYNTHETIC geometry of the 45 camera layout | all 45 cameras calibrate (0.5 cm, 0.08 degrees median at 2 px clicks); exact slot 56.0% and right SKU 98.7% from two views at the default noise | "Calibration, 3D slots and store-wide identity" |
| camera node | zone trigger on a Pi node, full-res bursts to a hub, Pi install kit | real lab video (MERL) replayed on a Mac; toy clips; nothing on a Pi | 425 of 425 reaches triggered; 22.7% of bytes saved with a shopper at the shelf the whole time | "Camera node first pass" |
| human review | review store, reviewer page, label export, metrics, weekly owner report, retention | SYNTHETIC alerts and random-number reviewers | the loop runs and adds up (61 alerts in, 128 label examples out, 0 left after retention) | "Human review feedback loop" |
| all together | one command from simulator views to a scorecard | one SIMULATED clip | 0 of 2 thefts caught, 0 alerts: the picks are lost before recognition | next section |

**What integration changed** (details in DECISIONS "Integration of the four streams"): the sim-trained detector is a backend choice everywhere; `make sim-eval` takes `--closed-world` and `--slots`; a node camera's bursts can be read back as one frame stream, so node cameras share a ledger and an identity pool with ordinary streams; the store YAML's `camera.calibration` is now loaded; the review store's label export is what the detector fine-tune script reads; the camera node numbers were re-measured with the final trigger code; the sim-eval adapter gives rail cameras their shelf zone. The triangulation and the simulator's 3D metric use one error formula (their outputs agree within about 10%, see the calibration section). The minimum pixel threshold stays at 20 px across a 6.6 cm can.

**Tests (re-run 2026-10-05 after the audit fixes).** `make test`: 313 passed in 658 s, with `BREE_PLAYWRIGHT` set and the sim-trained weights in place; without them the reviewer-page test and the detector weights test are skipped. `make test-fast` (no model weights): 300 passed, 1 skipped, 12 deselected in 55 s with `BREE_PLAYWRIGHT` unset; with it set the skipped browser test runs and the count is 301 passed. That browser test failed once in the audit on a loaded machine because its first key press arrived after the page's 0.4 s guard; the press is now made from inside the page, so it no longer depends on the clock. Each new area has its own file: `test_calib.py`, `test_multicam_closed_world.py`, `test_calib_bench.py`, `test_edge_node.py`, `test_review.py`, `test_detector_train.py`.

**The most important finding.** The recommended camera layout and the pipeline do not fit together yet. The layout gives people to the overhead cameras and labels to the rail cameras; the pipeline needs the person, the hand and the shelf edge in one camera. On the simulated clip that cost every pick. The next piece of work is a pick decided across cameras in 3D, not more accuracy in any one stage.

**Still open, in order.**
1. A cross-camera pick: person from the overhead cameras, hand and item from the item cameras, joined in 3D. Without it the recommended layout produces no picks in simulation.
2. Real frames. No stage here has seen the pilot store. The detector needs real labelled frames (`make sku-finetune`), the calibration tool has not been used on a real camera, the node has not run on a Pi, the review page has not had a real reviewer.
3. Items in a hand or on the counter (the results file counts the two together) are the detector's weak class (50.0% to 83.6% right SKU, simulated); more in-hand renders, then real ones.
4. The pipeline does not write alerts to the review store live, does not save a clean evidence frame at an alert (so real alerts give no detector labels), and writes no wall-clock time on alerts.
5. The ledger does not use the 3D slot. Store-wide closed world and re-ID stay off by default; re-ID waits on legal advice.
6. Node to hub is plain HTTP with a shared token: no TLS. JPEG bursts are 15 times larger than an H.264 stream of the same clips.
7. Isaac Sim has still not been run (no GPU). The simulator used here is the browser one.

# End to end on simulated data (2026-10-05)

**Bottom line.** The whole chain now runs as one command on simulator views, every stage through the repo's real code: simulator clip, camera nodes with the zone trigger, hub over HTTP, one pipeline run over all cameras with the sim-trained SKU detector, store-wide closed-world identity and the 3D slot step, alerts, review store, owner report, and the `make sim-eval` scorer. **The scorecard on the one clip run is zero: 0 of 2 simulated thefts caught, 0 alerts, 0 false alerts, 0 of 7 picks turned into a PICK event.** The chain did not break; it found nothing to alert on, and the run shows where the picks are lost. This is SIMULATED data (one 53 second clip from the browser simulator, 4 shoppers, simulated figures and invented products), so it is a test of the plumbing and of how the pieces fit, not a detection rate. Source: `results/e2e_sim_chain.json` and `.md` (`make sku-clip-door`, then `make e2e-sim E2E_RESULTS=e2e_sim_chain`; plain `make e2e-sim` writes next to the clip and leaves `results/` alone).

**Reproduce.**
```bash
export BREE_PLAYWRIGHT=<path to node_modules/playwright>     # Chrome, node 18+, network for three.js
node scripts/train/render_synth.mjs --clip --seed 5001 --out data/synth/clip_5001_door \
    --layout ~/bree/software/shared/layouts/recommended-47.json \
    --cams REGISTER-top,TRACK-2,TRACK-3,G4L-rail-1,G2L-rail-4,G2L-rail-3,G1R-rail-3,G3R-rail-1
.venv/bin/python scripts/e2e_sim_chain.py --sim-out data/synth/clip_5001_door --results-name e2e_sim_chain
```
`make sku-clip-door && make e2e-sim E2E_RESULTS=e2e_sim_chain` is the same two commands. The camera list is the register camera, the overhead camera that sees the door (TRACK-2), a second overhead, and five rail cameras: for 6 of the 7 picks the rail camera the simulator ranks best for that pick. The seventh is the second theft (P004, cocoa_crest, t = 19.87 s): the simulator ranks COOLER-rail-1 best for it (64.8 px across a can) and that camera is not in the clip; the clip has its second camera, G3R-rail-1 (28 px). So that theft was scored without its best view (`events.jsonl` in the clip folder). `make sku-clip` (no `-door`) picks cameras by itself and leaves the door camera out, which is why the list is given here.

**Does a re-render give the same clip? Yes, since 2026-10-05.** The audit re-rendered this seed and got different pixels in every frame of TRACK-2 and TRACK-3, and identity counts that were off by about one event. The cause was one unseeded draw: the simulator builds the clerk behind the counter once per page load and takes the sleeve length from `Math.random`, and the overhead cameras see the clerk's forearm. `scripts/train/sim/synth.js` now rebuilds the clerk for each seed from its own generator, without touching the two random streams that set the scene and the shoppers. Checked: a full re-render (530 frames on 8 cameras) is byte-identical to the stored clip in 4,240 of 4,240 frames, with identical `events.jsonl` and `truth_frames.jsonl`. The training frames were rendered before this, so in them the clerk's sleeve varies by render session, not by seed.

**Which simulator rendered it.** The private copy in `out/sim-copy` is commit a4f479f of `~/bree` (2026-10-03), found by comparing files; it is only made on first use, and the simulator has changed since (camera metrics, harness, page). The version is now written into `clip.json` and the dataset's `meta.json`. A fresh copy of the current simulator (07b2aa6) rendered the first 150 frames of this clip on the 8 cameras byte-identical to the stored ones (1,200 of 1,200 frames), so on that sample the detector saw what the current simulator draws. The clip is 530 frames at 10 fps on 8 of the 45 layout cameras, 7 picks by 4 shoppers, 2 concealed and 5 paid, 4 receipts.

**Stage 1 and 2: camera nodes and hub (SIMULATED views, real node and hub code, HTTP on localhost).** The five item cameras ran as nodes, as in the hardware plan; the register and overhead cameras streamed every frame.

| camera | kind | path | frames | triggers | bursts | frames that reached the hub | burst MB | every frame at the same mean size, MB (estimate) | trigger ms per frame |
|---|---|---|---|---|---|---|---|---|---|
| G1R-rail-3 | shelf | node, then hub | 530 | 2 | 4 | 197 (37.2%) | 89.9 | 241.7 | 3.370 |
| G2L-rail-3 | shelf | node, then hub | 530 | 3 | 5 | 234 (44.2%) | 110.3 | 249.9 | 3.521 |
| G2L-rail-4 | shelf | node, then hub | 530 | 4 | 5 | 221 (41.7%) | 95.9 | 230.0 | 0.801 |
| G3R-rail-1 | shelf | node, then hub | 530 | 4 | 6 | 281 (53.0%) | 154.0 | 290.4 | 3.052 |
| G4L-rail-1 | shelf | node, then hub | 530 | 3 | 5 | 231 (43.6%) | 112.5 | 258.1 | 3.509 |
| REGISTER-top | checkout | continuous stream | 530 | | | 530 (100%) | | | |
| TRACK-2 | overhead | continuous stream | 530 | | | 530 (100%) | | | |
| TRACK-3 | overhead | continuous stream | 530 | | | 530 (100%) | | | |

The nodes sent 1,164 of 2,650 full-resolution frames (43.9%), 562.6 MB against an estimated 1270.1 MB for every frame, with 0 hub errors. The trigger time per frame is from the run recorded on 2026-10-05 with other jobs on the machine (load average 3 to 5); the first recorded run, on a quieter machine, read 0.645 to 1.768 ms for the same frames. It depends on machine load, the counts do not: frames, triggers, bursts and bytes are identical in both runs. Wall time was 1,093.0 s in this run and 513.7 s in the first. A busy 53 second clip with four shoppers is close to the worst case for the trigger; the saving comes from the hours when nobody is at a shelf. **No pick was lost at this stage:** all 296 ground-truth boxes of an item in a hand on the item cameras are in frames that reached the pipeline.

**Stage 3 to 7: the scorecard (make sim-eval scorer).** Both columns were re-run on 2026-10-05 with the audit fixes in (the door rule, the horizon check). Thefts, alerts, PICK events, product boxes and the pick funnel are unchanged from the first recorded run; the identity rows moved and are the new values.

| metric | default settings (the recorded run) | person threshold 0.15 (probe, see below) |
|---|---|---|
| theft recall, alert tier | 0.0% (0 of 2) | 0.0% (0 of 2) |
| theft recall, alert or review | 0.0% | 0.0% |
| alerts / false alerts / review-tier | 0 / 0 / 0 | 0 / 0 / 0 |
| alert precision, SKU-correct rate, time to alert | no alerts, so none | no alerts, so none |
| PICK events from 7 true picks | 0 | 0 |
| product boxes seen by the pipeline | 5,370 in 2,754 frames | 7,333 in 2,754 frames |
| pipeline events | 13 enter, 1 exit | 13 enter, 1 exit |
| people created for 4 shoppers (at the door / inside as "missed entry" / warm-up) | 4 / 9 / 0 | 4 / 8 / 1 |
| cross-camera handoffs to someone seen now / to someone lost | 30 / 13 | 32 / 12 |
| identity uncertain marks / exits seen | 40 / 1 | 40 / 1 |
| frames where a track had no floor position (foot pixel at or above the horizon; not placed) | 52 | 48 |
| 3D slots on picks | none (no PICK) | none (no PICK) |
| review store: alerts ingested, decisions | 0, none | 0, none |
| owner report, week of 2026-10-05: alerts sent | 0 | 0 |

**Where the picks are lost (ground-truth boxes of an item in a hand, per camera kind).** Each column is its own count over the same item boxes. They are not nested steps: the two "and tracked person" columns are the intersections. "Tracked person" means a track in the pipeline's frame log. The SKU detector looks around raw person detections, which the frame log does not keep, so it can find a product in a frame that has no track.

| run | item boxes | frame reached pipeline | tracked person in frame | product box on item | right SKU | product box on item and tracked person | right SKU and tracked person | item inside a shelf zone of this camera |
|---|---|---|---|---|---|---|---|---|
| default, shelf cameras | 296 | 296 | 69 | 69 | 69 | 58 | 58 | 203 |
| default, overhead cameras | 602 | 602 | 578 | 2 | 0 | 2 | 0 | 0 |
| default, checkout cameras | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| person threshold 0.15, shelf cameras | 296 | 296 | 79 | 83 | 83 | 63 | 63 | 203 |
| person threshold 0.15, overhead cameras | 602 | 602 | 602 | 2 | 0 | 2 | 0 | 0 |
| person threshold 0.15, checkout cameras | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |

1. **A person is rarely tracked on the item cameras.** Of the 296 in-hand item boxes on the item cameras, 69 are in a frame with a tracked person (23%). The person detector itself fired more often than that: in 11 more of those frames the SKU detector found the item, which it can only do around a person detection, so a detection was there in at least 80 of the 296 (27%) and did not become a track. The rail cameras sit at shelf height and look along the aisle, so a shopper fills the frame and is cut by its edges, and the simulator's figures are smooth mannequins; COCO weights score them around the 0.3 threshold (0.24 to 0.58 on five pick-moment frames checked by hand). Lowering the threshold to 0.15, chosen from those five frames and so not a clean setting, moved the tracked count to 79 of 296 (and at least 99 with a detection: 79 plus 20 finds without a track). Whether real people in a real rail view do better is not known from this.
2. **The SKU detector is right when it finds the item, and it finds most of them when it gets to look.** It only runs around person detections. Of the 69 item boxes in a frame with a tracked person it found the item on 58 (84%), all 58 with the right SKU; it missed 11. It also found 11 items in frames with a detection but no track, again all with the right SKU, which is why the "product box" column also reads 69: the two 69s are a coincidence of totals, not the same boxes. At threshold 0.15: 63 of 79 (80%) with a tracked person, plus 20 without, all 83 with the right SKU. (Corrected 2026-10-05: this point first read the columns as nested and said 69 of 69.)
3. **The overhead cameras see the people and cannot see the items.** A tracked person was in 578 of 602 overhead frames, a product box on the item in 2: from the ceiling an item is a few pixels.
4. **The shelf zone as an image polygon does not fit an along-the-aisle view.** The event engine counts an item as taken once it is outside the shelf polygon. From a rail camera the shelf face fills most of the image, and 203 of the 296 in-hand item boxes (69%) lie inside that camera's shelf polygon, so by the engine's rule most of them would not count as taken while they are there, whatever the detector does.
5. **Identity on these views is poor.** 4 shoppers became 13 people, 9 of them created inside the store because a track appeared with nobody unaccounted for, and 1 of the 4 exits was seen. (Before the audit fixes the same clip gave 20 people, 16 of them inside. One cause was tracks whose foot pixel lies at or above the horizon: they were given a floor position that does not exist. They are now left unplaced, 52 track-frames in this run. How much of the drop is that and how much the new door rule was not separated.) No exit means no reconciliation, so even a detected pick would mostly not have reached an alert.

**What this means.** The pipeline was built and measured on views that show the person, the hand and the shelf edge in one camera (MERL's overhead view, the toy clips). The recommended layout splits those jobs: overhead cameras see people, rail cameras see labels. Nothing yet joins a person tracked from above to an item seen from a rail camera. That join is the missing piece, and the parts exist: store-wide identity gives one person id across cameras, and the calibration work gives the 3D position of a hand and of a shelf face. A pick decided in 3D (the hand crosses the shelf face plane, the item camera names what left) would replace both the per-camera person requirement and the image-polygon zone. Not built.

**Integration fixes found by this run.** The sim-eval adapter dropped any shelf zone with a corner behind the camera, so the rail cameras had no shelf zone at all; it now keeps the part in front. The door zone went to the item camera that saw it largest, which as a node only sends frames on a trigger; door and register zones now go to a people camera when one sees them. `summary.json` had no store-wide identity counts. All three are fixed; `tests/test_sim_eval.py` has a test for the rail camera zone and the fixture test covers the door rule.

**Limits.** One clip, 53 seconds, 4 shoppers, 2 thefts: every count above is tiny. Simulated figures, simulated products, a detector trained on the same simulator. The second theft's best item camera (COOLER-rail-1) is not in the clip; it was scored from its second camera (G3R-rail-1). Zero alerts means the review store, owner report and slot scoring ran on empty input here; they are exercised with data by their own tests and by `make review-demo` (SYNTHETIC). The node read an mp4 of the rendered frames, not a sensor.

# SKU detector trained on simulated frames (2026-10-04)

**Bottom line.** A product detector that names the SKU now exists, trained only on frames rendered from the browser store simulator, and it is a first-class backend (`--backend sim_sku`, `make sim-eval SIM_BACKEND=sim_sku`). **Every number in this section is on SIMULATED frames from one store model with invented package art. None of it is accuracy on real footage.** On held-out simulated scenes it finds a clearly visible shelf item with the right SKU 97.3% of the time at 20 to 25 pixels across a can and 96.2% at 15 to 20, so nothing on this data contradicts the simulator's 20 pixel threshold or supports raising it (the curve shows no cliff at any size, so it does not single out 20 either; see below). Two things are weak: items in a hand or on the counter (the evaluation counts the two together: 50.0% to 83.6% right SKU by bucket, few examples, and one bucket has 5 items), which is the case the pick logic depends on, and items larger than a tile's overlap, which the tiling cuts. Source: `results/sku_detector.json`, `results/sku_detector.md`, `results/px_vs_accuracy.png` (`make sku-eval`, run at integration).

**What was built** (`src/bree/train/`, `scripts/train/`). `render_synth.mjs` drives a private copy of the browser simulator headless (the original is never edited) with an overlay that randomises lights, exposure, colour cast, clothing, slot fill, item pose, same-size SKU swaps and camera pose, and reads ground-truth boxes from an instance-id render, so a box is what is actually visible. `dataset.py` cuts each 4MP frame into 640 px tiles at native resolution (a 25 px can stays 25 px) after adding blur, noise, white balance, gain and JPEG compression. `train.py` fine-tunes YOLO26 nano. `backend.py` runs it tiled in the pipeline, around people only. `evaluate.py` scores it. `finetune_real.py` is the path to real labelled frames, and it reads the review store's label export (`--manifest`).

**Data and training.** 151 scene seeds for training (3,299 frames, 13,596 tiles, 170,667 boxes), 44 for validation, 45 for the test (965 frames, 3,991 tiles, 49,424 boxes). The split is by scene seed, so no scene, shopper or lighting draw is shared; it is still one store and one set of package art. 33 SKUs. Of the training boxes 2,079 are items in a hand and 808 on the counter; the rest sit on shelves. yolo26n.pt fine-tuned for 6 epochs at batch 32 on mps. The recorded wall time is 320.8 minutes, but the first epoch took 4 hours 4 minutes because the machine stalled, and the fourth took 27 minutes while other jobs ran (the training log shows 11 to 14 minutes for each of the other four: 11.6, 11.2, 11.3 and 13.5, the last one including the final validation), so that figure is not a cost. No validation pass ran during training; the kept weights are the last epoch's. Val split at the end: mAP50 0.973, mAP50-95 0.884.

**Test tiles (SIMULATED).** mAP50 0.972, mAP50-95 0.882 over 3,991 tiles and 49,424 boxes. Lowest per SKU: orchard_peach 0.903 and orchard_lemon 0.916; every other SKU is at 0.953 or above.

**Pixels against accuracy (SIMULATED; whole test frames through the same tiled inference the pipeline uses; 965 frames, 84,873 labelled items, confidence 0.25, IoU 0.5, box precision 90.9%).** "Pixels across a can" is the simulator's effective-pixel metric without its lens edge term. Each cell is the share of ground-truth items found with the right SKU.

| pixels across a 6.6 cm can | 0-10 | 10-15 | 15-20 | 20-25 | 25-30 | 30-40 | 40-60 | 60-100 | 100+ |
|---|---|---|---|---|---|---|---|---|---|
| shelf, front item, 80%+ visible: right SKU (items) | 100.0% (93) | 94.3% (124) | 96.2% (1,238) | 97.3% (2,048) | 98.4% (2,357) | 98.3% (12,070) | 91.8% (5,743) | 91.8% (1,958) | 92.2% (64) |
| shelf, front item, 25 to 80% visible: right SKU (items) | 96.6% (267) | 92.3% (804) | 91.6% (4,060) | 94.7% (5,788) | 94.4% (6,006) | 95.6% (10,353) | 89.8% (9,268) | 83.7% (2,922) | 90.1% (81) |
| shelf, item behind the front one: right SKU (items) | 95.7% (392) | 84.3% (261) | 87.5% (1,299) | 91.8% (1,996) | 91.6% (2,214) | 91.3% (7,277) | 86.9% (4,589) | 79.9% (1,160) | 75.9% (29) |
| in a hand or on the counter: right SKU (items) | none | none | none | none | none | 60.0% (5) | 58.3% (144) | 83.6% (165) | 50.0% (98) |
| all items: found / right SKU | 97.6% / 96.5% | 92.5% / 90.8% | 94.0% / 91.7% | 96.4% / 94.6% | 95.8% / 94.7% | 96.4% / 95.6% | 90.2% / 89.5% | 85.9% / 85.5% | 79.8% / 74.6% |

| short side of the item's own box, pixels | 0-10 | 10-15 | 15-20 | 20-25 | 25-30 | 30-40 | 40-60 | 60-100 | 100+ |
|---|---|---|---|---|---|---|---|---|---|
| all items: right SKU (items) | 50.0% (96) | 81.7% (1,071) | 92.4% (2,790) | 93.5% (3,073) | 93.6% (3,316) | 96.4% (16,130) | 96.2% (23,504) | 87.9% (18,612) | 90.6% (16,281) |

**What the curve says.**
- **The 20 px threshold is not contradicted in simulation. The curve does not single it out.** For a clearly visible front item the right-SKU rate is 100.0% at 0 to 10 px (93 items), 94.3% at 10 to 15 (124 items), 96.2% at 15 to 20, 97.3% at 20 to 25, 98.4% at 25 to 30 and 98.3% at 30 to 40. There is no cliff between 20 and 40, which is the range the camera layout report worried about, and none below 20 either: this measure finds no lower limit down to its smallest bucket. **Documented default: 20 px across a 6.6 cm can** (the simulator's `pxMin`), unchanged.
- **The floor shows in the item's own box, and that is the measure to use for a lower limit.** When the short side of the box is 10 to 15 px the right-SKU rate is 81.7%, and under 10 px it is 50.0%. From 15 px it is 92.4% or better up to 60 px. The two measures differ because "across a can" normalises to a 6.6 cm width and includes the viewing angle, while most items are wider than a can.
- **Accuracy drops again for large, close items** (91.8% at 40 to 60 px for clear front items, 74.6% over all items at 100 px and more). These are found less often, not confused: an item larger than the 128 px tile overlap that no tile holds whole is cut. A downscaled whole-frame pass next to the tiles is the fix, not built.
- **Items in a hand or on the counter are the weak class** (`in_hand_or_on_counter` in the results file; the two are not split): 60.0% at 30-40 px (5 items, too few to read), 58.3% at 40-60 px (144 items), 83.6% at 60-100 px (165 items), 50.0% at 100+ px (98 items). The pick logic reads exactly these. They are 1.2% of the training boxes; more in-hand renders is the cheap next step.
- **Near-identical variants** (same shape and size, only the label art differs, 11 families): the detector gives a sibling's SKU in 0.2% to 7.5% of found items. Worst: orchard_lemon / orchard_peach 7.5% (284 of 3,770); torqueline_5w30 / torqueline_10w30 2.0% (83 of 4,153); cocoa_crest / peanut_pilot / caramel_crest 1.0% (172 of 17,366); fizzo_cola / fizzo_zero / fizzo_cherry / lumen_citrus 0.8% (92 of 11,697).
- **Speed (this Mac, random-pixel frames, measured with the machine otherwise idle):** 8.2 ms per 640 px tile on MPS and 39.8 ms on CPU; a whole 1520x2688 frame is 15 tiles, 48.2 ms on MPS (20.73 frames per second) and 300.3 ms on CPU (3.33). In the pipeline only tiles around people are run. **The CPU figures are approximate.** They move a lot with what else the machine is doing: the audit measured 45.7 to 51.3 ms per tile on a quiet machine, and three re-runs on 2026-10-05 with a system job running (load average 3 to 5) gave 77.1 to 113.5 ms per tile and 1.27 to 1.32 frames per second. Read CPU as roughly 40 to 115 ms per tile, 1.3 to 3.3 frames per second. MPS held up better: 8.2 to 11.8 ms per tile, 12.17 to 20.61 frames per second in the same three runs.

**Limits.** One store model, one planogram, one set of invented package art, and the detector is trained and tested on the same renderer: the test split measures held-out scenes, not a held-out store, and says nothing about real packaging, real lenses or real light. Real labels seen 60 degrees off the shelf normal are the untested case. On real footage this detector finds nothing useful until it is fine-tuned on real labelled frames (`make sku-finetune`), which has not been run. Six epochs without a validation pass is a short run; nothing was tuned.

# Calibration, 3D slots and store-wide identity (2026-10-04)

**Bottom line.** Three things are built and tested, all off until a store is configured for them, and **every number here is SYNTHETIC geometry** (the 45 camera layout's camera poses and the store's slots, seeded random draws, no footage and no rendering): (1) each camera can be calibrated to the floor plan from hand-clicked marks (`scripts/calibrate.py`, `make calibrate`, `make calib-check`); (2) every PICK can carry the 3D slot it came from, from two views of the hand when two calibrated cameras saw it, else from one; (3) closed-world identity now runs across cameras as one pool for the store. With 2 px clicks and a known lens all 45 layout cameras calibrate, with a median pose error of 0.5 cm and 0.08 degrees. The exact slot is the hard part: under the default noise model (5 px on the hand, 2 cm and 0.2 degrees of calibration) two views name the exact slot 56.0% of the time and the right SKU 98.7% of the time, because neighbouring slots usually hold the same SKU. Store-wide closed world with re-ID beats the current handoff on every handoff, merge, split and exit count, and costs something on two rows the current handoff leaves at zero (people still counted inside at the end, 61 and 66 against 0, and visits marked uncertain, 239 and 237 against 0); without re-ID it trades splits for wrong joins it marks as uncertain, the same shape as the single-camera result. Source: `results/calib_bench.json` (`make calib-bench`, re-run on 2026-10-05 after the door rule changed, 181 s: the calibration and slot tables are identical to the earlier file, the closed-world columns of the identity tables moved by one to four counts). Design: DECISIONS.md "Calibration, 3D slots and store-wide identity".

**Calibration from hand-clicked marks (SYNTHETIC marks: each of the 45 layout cameras calibrated from up to 12 marks it can see, Gaussian click noise).**

| click noise, lens | cameras calibrated | reprojection RMS px (median / max) | position error cm (median / p90 / max) | rotation error degrees (median / p90 / max) | focal length error % (median / max) |
|---|---|---|---|---|---|
| 1 px, lens known | 45 of 45 | 1.2 / 1.6 | 0.2 / 0.9 / 1.8 | 0.04 / 0.12 / 0.19 | 0.00 / 0.00 |
| 1 px, focal length fitted | 45 of 45 | 1.1 / 1.6 | 0.8 / 1.9 / 4.0 | 0.05 / 0.13 / 0.27 | 0.18 / 0.90 |
| 2 px, lens known | 45 of 45 | 2.3 / 3.2 | 0.5 / 1.9 / 3.6 | 0.08 / 0.24 / 0.39 | 0.00 / 0.00 |
| 2 px, focal length fitted | 45 of 45 | 2.3 / 3.2 | 1.6 / 3.8 / 8.1 | 0.11 / 0.26 / 0.55 | 0.37 / 1.81 |
| 5 px, lens known | 45 of 45 | 5.8 / 8.0 | 1.2 / 4.7 / 9.1 | 0.20 / 0.58 / 0.98 | 0.00 / 0.00 |
| 5 px, focal length fitted | 45 of 45 | 5.7 / 7.9 | 4.0 / 9.5 / 20.5 | 0.26 / 0.65 / 1.40 | 0.93 / 4.55 |

Leave-one-out floor error at 2 px clicks, for the 19 cameras that see 5 or more floor marks: median 7.9 cm, p90 1.19 m, max 20.2 m. The large values are item cameras that see only a short strip of floor: a floor-only fit there is close to degenerate, which is why the full pose (floor plus raised marks) is the default when the lens is known.

**3D slot of a pick (SYNTHETIC reaches, one per slot; 2349 slots, 41 item cameras; fixtures block lines of sight, bodies do not).** Slots seen by no item camera: 5, by one: 785, by two or more: 1559.

| noise model (hand keypoint, calibration) | two views: exact slot | two views: right SKU | two views, plain nearest slot (no error weighting) | one view, hand only, same slots: exact | one view, item seen on the shelf, same slots: exact | as deployed, all seen slots: exact / right SKU | two view 3D error cm (median / p90) |
|---|---|---|---|---|---|---|---|
| pixel 0.29 px, pointing 0.1 deg (the simulator's defaults) | 79.8% | 99.7% | 79.9% | 60.0% | 98.9% | 75.8% / 99.4% | 6.2 / 10.1 |
| pixel 2 px, perfect calibration | 84.0% | 100.0% | 84.2% | 60.2% | 99.3% | 79.1% / 99.5% | 6.0 / 9.5 |
| pixel 5 px, perfect calibration | 72.8% | 99.9% | 69.1% | 59.5% | 99.1% | 71.1% / 99.5% | 7.0 / 12.8 |
| pixel 5 px, calibration 1 cm / 0.1 deg | 66.7% | 99.3% | 56.1% | 54.1% | 87.0% | 66.2% / 98.9% | 8.1 / 19.4 |
| pixel 5 px, calibration 2 cm / 0.2 deg (default model) | 56.0% | 98.7% | 38.6% | 46.2% | 63.8% | 56.7% / 98.3% | 10.6 / 29.5 |
| pixel 5 px, calibration 5 cm / 0.5 deg | 32.8% | 95.6% | 14.2% | 26.8% | 27.4% | 34.3% / 94.0% | 19.7 / 63.3 |
| pixel 10 px, calibration 2 cm / 0.2 deg | 53.4% | 98.5% | 37.6% | 43.9% | 62.3% | 53.7% / 98.0% | 11.2 / 34.9 |
| pixel 20 px, calibration 2 cm / 0.2 deg | 47.8% | 97.9% | 29.2% | 38.9% | 54.5% | 46.8% / 96.5% | 14.3 / 52.0 |
| pixel 5 px, cameras calibrated by the tool (2 px clicks, 12 marks) | 70.2% | 99.7% | 69.7% | 57.5% | 98.7% | 68.5% / 99.5% | 6.8 / 13.5 |

**Store-wide identity, 2 overhead cameras (TRACK-1, TRACK-2; door cameras TRACK-1, TRACK-2). SYNTHETIC shoppers, 200 episodes, identity logic only, appearance features drawn, not extracted from pixels.**

| | current (floor-plan handoff) | re-ID | closed world | closed world + re-ID |
|---|---|---|---|---|
| shoppers / camera tracks | 683 / 6,123 | 683 / 6,123 | 683 / 6,123 | 683 / 6,123 |
| handoffs right, another camera sees them now | 2,044 of 3,347 (61.1%) | 3,047 of 3,347 (91.0%) | 3,150 of 3,347 (94.1%) | 3,296 of 3,347 (98.5%) |
| handoffs right, after a gap nobody saw | 451 of 2,093 (21.6%) | 1,854 of 2,093 (88.6%) | 1,362 of 2,093 (65.1%) | 1,930 of 2,093 (92.2%) |
| handoff success, all | 45.9% | 90.1% | 82.9% | 96.1% |
| false merges | 776 | 246 | 902 | 190 |
| of those, silent (not marked uncertain) | 776 | 246 | 62 | 3 |
| splits | 2,815 | 435 | 698 | 173 |
| visits with one clean identity | 0.1% | 41.4% | 21.5% | 75.7% |
| visits marked uncertain (alerts capped at review) | 0 | 0 | 672 | 239 |
| exits on the right person / the wrong person | 612 / 53 | 659 / 22 | 641 / 0 | 642 / 0 |
| still counted inside at the end | 0 | 0 | 62 | 61 |

**Store-wide identity, 4 overhead cameras (TRACK-1, TRACK-2, TRACK-3, TRACK-4; door cameras TRACK-1, TRACK-2). SYNTHETIC shoppers, 200 episodes, identity logic only, appearance features drawn, not extracted from pixels.**

| | current (floor-plan handoff) | re-ID | closed world | closed world + re-ID |
|---|---|---|---|---|
| shoppers / camera tracks | 703 / 8,505 | 703 / 8,505 | 703 / 8,505 | 703 / 8,505 |
| handoffs right, another camera sees them now | 3,589 of 5,830 (61.6%) | 5,251 of 5,830 (90.1%) | 5,503 of 5,830 (94.4%) | 5,762 of 5,830 (98.8%) |
| handoffs right, after a gap nobody saw | 367 of 1,972 (18.6%) | 1,741 of 1,972 (88.3%) | 1,242 of 1,972 (63.0%) | 1,814 of 1,972 (92.0%) |
| handoff success, all | 50.7% | 89.6% | 86.5% | 97.1% |
| false merges | 1,370 | 426 | 1,008 | 194 |
| of those, silent (not marked uncertain) | 1,370 | 426 | 69 | 16 |
| splits | 3,500 | 566 | 804 | 179 |
| visits with one clean identity | 0.0% | 31.3% | 19.2% | 76.4% |
| visits marked uncertain (alerts capped at review) | 0 | 0 | 700 | 237 |
| exits on the right person / the wrong person | 602 / 80 | 669 / 32 | 659 / 0 | 660 / 0 |
| still counted inside at the end | 0 | 0 | 71 | 66 |

**How to read the slot table.** "Two views" rows are over the 1,559 slots two or more item cameras see. "One view, hand only" is the fallback when a single camera saw the hand: the line of sight meets the shelf face, and a hand held in front of the shelf lands on the wrong slot along that line. "One view, item seen on the shelf" is the case where the camera that raised the PICK first saw the item sitting in its slot; it beats two views of the hand whenever calibration is good, because the item is on the shelf face and the hand is not. "As deployed" is what the pipeline does: the best source available for each slot. The 3D error column is the distance from the estimated hand to the true slot face, so it includes how far the hand is from the shelf. It is not the simulator's triangulation error of an exactly known point (about 1 cm in `~/bree/research/camera-layouts-3d.md`).

**What the numbers say.**
- **Calibration is not the bottleneck if the lens is known.** 2 px clicks give 0.08 degrees median and 0.24 degrees at the 90th percentile, close to the simulator's 0.1 degree placeholder at the median and to this repo's 0.2 degree default at the tail. Fitting the focal length as well roughly triples the position error (0.5 to 1.6 cm median), so write the lens into the YAML.
- **Exact slot needs good calibration more than good pixels.** Going from 5 px to 20 px on the hand at fixed calibration costs 8 points (56.0% to 47.8%). Going from perfect calibration to 2 cm and 0.2 degrees at 5 px costs 17 points (72.8% to 56.0%), and 5 cm and 0.5 degrees leaves 32.8%. Cameras calibrated by the tool itself (2 px clicks) give 70.2%.
- **Weighting by the error model earns its keep once calibration is imperfect:** 56.0% against 38.6% for the plain nearest slot at the default model. With near-perfect calibration the two are the same.
- **The right SKU is much easier than the right slot** (98.7% at the default model), which is what the ledger needs. The exact slot matters for planogram audits and for telling two SKUs on neighbouring facings apart.
- **Store-wide closed world.** With 4 overhead cameras, handoff success goes from 50.7% (current) to 86.5% (closed world) and 97.1% (closed world plus re-ID); exits credited to the wrong person go from 80 to 0. Without re-ID, false merges fall with 4 cameras (1,370 to 1,008) and rise with 2 (776 to 902), but silent ones fall to 62 and 69; nearly every visit is then marked uncertain (672 of 683, 700 of 703), so alerts are capped at review. With re-ID 237 to 239 visits are marked.

**Decision: store-wide closed world stays off by default.** Same rule as the single-camera decision: on only if clearly better on store data without more false merges. Without re-ID it marks almost every visit uncertain and raises raw false merges with 2 cameras; with re-ID it wins on every handoff, merge, split and exit count (at the price of the two rows named above), and re-ID is still waiting on legal advice. All of it is synthetic. To turn it on: `rules: {closed_world: true}` in the camera YAMLs; `multicam: {closed_world: false}` keeps the plain handoff.

**Shared definitions with the simulator.** The triangulation error model is the simulator's two-camera formula (`pixel_px` = `pxSigma`, `cam_rot_deg` = `calibDeg`), checked at the image centre in `tests/test_calib.py::test_error_model_is_the_simulators`. The outputs agree within about 10%, not exactly: on the simulator's per-slot output for the 45 camera layout (471 slots it calls localisable, same camera pair) the pipeline's `rms_m` divided by the simulator's error has median 0.960, minimum 0.891, maximum 0.999. The simulator is the more cautious one (lens edge loss 0.2, range in place of depth). The prediction is also first order: an independent Monte Carlo measured 1.043 times the predicted error at the median, up to 1.091 at the pipeline defaults and 1.121 at the simulator's. Both sets of figures are from the audit of 2026-10-05, SYNTHETIC geometry, not re-run here; the camera model projects layout cameras to the same pixels as the sim-eval adapter (`test_layout_camera_projects_like_the_simulator_adapter`). The defaults differ and say so: simulator 0.29 px and 0.1 degree for a shelf point, pipeline 5 px, 2 cm and 0.2 degrees for a hand keypoint. The simulator re-run at the pipeline's values is in `~/bree/research/camera-layouts-3d.md`, addendum 2026-10-04.

**Integration fixes.** `load_store_config` now loads the `camera.calibration` block that `scripts/calibrate.py` writes (it was written but never read, so a calibrated YAML fell back to floor marks). `summary.json` `identity.store` and `engine_log.txt` `[store]` lines carry the store-wide closed-world counts and log. `make sim-eval SIM_ARGS="--closed-world --slots"` runs both on simulator output and scores the slot of each pick against the simulator's slot.

**Tests.** `tests/test_calib.py` (22), `tests/test_multicam_closed_world.py` (18), `tests/test_calib_bench.py` (4, a seeded guard against `tests/fixtures/calib_guard.json`).

**Limits.** Pinhole model, no lens distortion. Bodies are points and do not block views in the bench. Appearance features are drawn, not extracted. Nothing has been calibrated from a real frame. The slot of a pick is not used by the ledger yet. No real multi-camera footage with identity labels exists here.

# Camera node first pass (2026-10-04): trigger on the node, full-res bursts to the hub

What it is: each Pi node runs a cheap "something hand-sized moved into a shelf zone" check on a 320 pixel
wide stream and sends full-resolution frames only around a trigger (2 s before, while active, 1.5 s
after). Code in `src/bree/edge/` (`trigger.py`, `capture.py`, `uplink.py`, `node.py`, `hub.py`), Pi kit in
`deploy/pi/`. Nothing here has run on a Pi or seen a Camera Module 3: every number below is from recorded
clips replayed through the node code on a Mac (Apple M1 Max), with a file standing in for the camera.

Reproduce (results in `results/edge_measure_*.json`; `make edge-measure` runs the first three). Every file was re-run at integration with the final trigger code:

```bash
PYTHONPATH=src .venv/bin/python scripts/edge/measure.py merl --split test --out results/edge_measure_merl_test.json
PYTHONPATH=src .venv/bin/python scripts/edge/measure.py toy --fps 15 --out results/edge_measure_toy.json
PYTHONPATH=src .venv/bin/python scripts/edge/measure.py cpu --out results/edge_measure_cpu.json
PYTHONPATH=src .venv/bin/python scripts/edge/measure.py merl --split train --every 6 --sweep --out results/edge_measure_merl_train_sweep.json
PYTHONPATH=src .venv/bin/python scripts/edge/measure.py synthetic --out results/edge_measure_synthetic.json
PYTHONPATH=src .venv/bin/python scripts/edge/measure.py table      # the tables below
.venv/bin/python -m pytest tests/test_edge_node.py                 # 32 tests
```

Data and truth:
- **MERL Shopping, test split**: real overhead lab video, 920x680, one shopper at one shelf, hand actions
  labelled per frame. MERL has no "pick" label. Every pick starts with a reach, so recall is counted on the
  425 labelled "reach to shelf" instances. A trigger is false when it touches no labelled reach, retract
  or hand-in-shelf (within 0.5 s).
- **Toy clips**: the repo's synthetic flat-colour drawings. Truth is the renderer's own script: the
  moment each item is taken or put back.
- The trigger settings were chosen on 10 MERL training videos. The test split was then run twice: once with the first trigger (455 triggers, 81 false) and again after brightness normalisation was added (429 and 77, the numbers below). That change was motivated by the synthetic flicker scene, not by test results, and recall is 425 of 425 in both runs; it is still a test set seen twice.
- The browser store sim was not used: its shoppers are cylinders without arms, so there is no hand to
  trigger on.

| | MERL test split (real lab video, 10 fps) | toy clips (SYNTHETIC, 15 fps) |
|---|---|---|
| clips, length | 28, 64.3 min | 8, 4.1 min |
| reaches (MERL) / takes and put-backs (toy) | 425 | 12 |
| trigger recall | 100.0 % (425 of 425) | 100.0 % (12 of 12) |
| fully inside a full-res burst | 99.8 % | 100.0 % |
| triggers | 429 | 31 |
| false triggers | 77 = 72 per hour | 16 = 237 per hour |
| always-on control (a trigger active on every frame) | 100.0 % recall, 0 false triggers, 0 % saved | 100.0 % recall, 0 false triggers, 0 % saved |
| frames within 0.5 s of labelled hand activity | 46.2 % | 12.7 % |
| active frames outside labelled hand activity | 32.4 % | 79.2 % |
| reach frames with the trigger active | 81.3 % | 91.0 % |
| share of time the trigger is active | 55.4 % | 51.0 % |
| share of frames sent at full resolution | 77.1 % | 80.2 % |
| full-res JPEG stream, every frame (baseline) | 4487.4 MB = 9.31 Mbit/s | 130.6 MB = 4.29 Mbit/s |
| bursts only | 3468.3 MB = 7.20 Mbit/s | 106.1 MB = 3.48 Mbit/s |
| bandwidth saved, bursts only | 22.7 % | 18.8 % |
| low-res stream as well (every frame) | 488.5 MB = 1.01 Mbit/s | 15.6 MB = 0.51 Mbit/s |
| bandwidth saved, bursts + low-res stream | 11.8 % | 6.8 % |
| same clips as continuous H.264 (libx264 crf 23) | 227.0 MB = 0.47 Mbit/s | 4.5 MB = 0.15 Mbit/s |
| JPEG bursts against that H.264 stream | 15.3 times larger | 23.5 times larger |

Projection, not a measurement: MERL rates scaled by the share of time someone is at the shelf (MERL itself is 100 %: a shopper is in front of the shelf for the whole of every clip).

| someone at the shelf | bursts, Mbit/s | saved against the full-res JPEG stream |
|---|---|---|
| 100 % of the time | 7.20 | 22.7 % |
| 50 % of the time | 3.60 | 61.4 % |
| 20 % of the time | 1.44 | 84.5 % |
| 5 % of the time | 0.36 | 96.1 % |

| CPU, one core, OpenCV on 1 thread | measured here (Apple M1 Max, machine otherwise idle) | ESTIMATE for a Pi Zero 2 W (one Zero 2 W core is 8x to 20x slower than one core here (assumed, not measured)) |
|---|---|---|
| trigger, per 320x237 frame | 0.61 ms | 4.9 to 12.2 ms |
| trigger at 10 fps, share of one core | 0.6 % | 4.9 % to 12.2 % |
| JPEG encode, 920x680 clip frame | 1.4 ms | not estimated |
| JPEG encode, clip frame upscaled to 2304x1296 | 6.1 ms | 49 to 122 ms |
| JPEG encode, clip frame upscaled to 4608x2592 | 22.3 ms | 178 to 446 ms |

| SYNTHETIC empty scene, nobody in view, 3000 frames at 10 fps: frames with the trigger active | gain_norm off (first version) | gain_norm on (default) | gain_norm on, shake_px 1 (option) |
|---|---|---|---|
| still | 0 (0 runs) | 0 (0 runs) | 0 (0 runs) |
| shake_2px | 2995 (1 runs) | 2995 (1 runs) | 0 (0 runs) |
| flicker_10pct | 1338 (162 runs) | 0 (0 runs) | 0 (0 runs) |

What the numbers say:
- **No reach was missed** on the MERL test split (425 of 425), and all but one lay fully inside a
  full-res burst. On the training videos the drawn shelf polygon alone found 128 of 155 reaches; growing
  the zone by 6 low-res pixels found 155 of 155 (the misses were hands at the front edge of the shelf).
  That margin is the default.
- **The trigger means "someone is at this shelf", not "a hand is in it".** It is active 55.4 % of the time
  on MERL, where a shopper stands at the shelf for the whole clip, and 77.1 % of the full-res frames are
  sent once pre-roll and post-roll are added. So with a shopper present the saving is small (22.7 %).
  The saving comes from the time nobody is at that shelf, and no data here measures how much of the day
  that is. The projection rows only scale the MERL rate.
- **False triggers: 72 per hour of a shopper standing at the shelf** (77 in 64.3 minutes: bodies and arms
  near the zone without a labelled reach). A false trigger costs upload, not an alert (8.1 MB per trigger
  on average over all 429 triggers at 920x680). The always-on row is the control: a trigger that is
  active on every frame has the same recall and saves nothing, so recall alone says little; the saving
  and the false trigger rate are the numbers to watch.
- **Empty aisle (SYNTHETIC textured scene, nobody in view, last table).** Still: 0 active frames of 3,000.
  10 % brightness flicker: 1,338 active frames with the first version, 0 with the brightness
  normalisation that is now the default. A 2 pixel camera shake keeps the trigger on for 2,995 of 3,000
  frames unless `shake_px: 1` is set, which is off by default. A real sensor's noise, a cooler door
  reflection and a real mount's vibration are not measured: there is no such clip here.
- **JPEG bursts are far larger than H.264.** The same clips as one continuous H.264 stream are 15 times
  smaller than the JPEG bursts. Against a camera that can stream H.264 at full resolution, bursts of
  JPEGs lose. The Pi's hardware H.264 encoder stops at 1080p, so for 2304x1296 or 12 MP frames the
  choices on a Zero 2 W are JPEG stills or software encoding; and stills keep the label detail that
  product recognition needs. Still, the byte cost per burst is the weak point of this design. Next
  steps that would cut it, not built: send only the crop around the active zone, and lower the burst
  frame rate once the hand is in (a few sharp frames matter more than ten a second).
- **The low-res stream is not free**: every 320 pixel frame as JPEG at 10 fps is 1.01 Mbit/s. It is off
  by default (`lowres_uplink_fps: 0`); the trigger does not need it to leave the node.
- **CPU**: the trigger costs 0.61 ms per frame on one M1 Max core on an idle machine. Three re-runs on 2026-10-05 with a system job running (load average 3 to 5) gave 0.62, 0.91 and 0.91 ms, and JPEG encodes of 1.6 to 2.2, 6.9 to 9.7 and 25.6 to 39.0 ms for the three frame sizes, so read every time in this table as a best case. The Pi column is an estimate from an
  assumed 8 to 20 times slowdown, not a measurement, and it leaves out the camera stack and the copy of
  each full-res frame into the ring. The real figure arrives in the first heartbeat (`trigger_ms`).
  JPEG encoding is the heavy part: at the estimated 49 to 122 ms per 2304x1296 frame a 10 fps burst
  cannot be encoded in real time on one core, so the node encodes on a second thread and capture waits
  if the encoder falls a ring buffer behind. Expect to run bursts below 10 fps on the Zero 2 W.
- Clip frames are 920x680 and 1280x720, not 12 MP. Byte counts will be larger on the real sensor; the
  saved share depends on how often the trigger fires, not on the frame size.

Tested on this Mac (`tests/test_edge_node.py`): a hand entering the zone triggers and releases, a dark
glove triggers, movement outside the zone and a lighting change do not; a burst holds 2 s of pre-roll,
the active frames and the post-roll with camera id, time and zone ids; a long trigger is cut into parts
with no frame lost; node to hub over real HTTP stores full-res frames; the hub being down loses nothing
(spool, then drain); a busy hub answers 429 and the node keeps the burst; a wrong token is refused; a
repeated message is stored once; a toy clip through node, hub and the existing pipeline produces a pick
event (toy data); bursts left on disk reach the pipeline after a hub restart; brightness flicker on an
empty scene does not trigger and a hand under flicker still does; a node whose trigger is stuck on says
so in its heartbeat; and the stored bursts of a camera read back as one frame stream in time order
(`BurstSource`, added at integration).

Not done:
- Nothing has run on a Raspberry Pi. `Picamera2Source` is written from the picamera2 manual and untested.
- The hub's own `--store` mode still runs the pipeline once per burst, with no shopper identity or basket
  across bursts. `bree.edge.hub.BurstSource` (integration) reads a camera's stored bursts as one stream
  with gaps, so node cameras and ordinary streams run into one ledger with one identity pool; that path
  has run on SIMULATED views only (`make e2e-sim`, section "End to end on simulated data") and reads
  recorded bursts, it does not follow a live hub.
- No TLS: the token and frames travel in clear text on the camera network.
- Stored bursts are raw frames with faces. Since 2026-10-05 the hub deletes them 24 hours after they arrive
  (`--retain-hours`, DECISIONS "Audit fixes"); before that nothing deleted them.
- The pre-roll is trimmed by time since 2026-10-05. Before, it was a frame count: a camera delivering 5 frames
  a second gave 4.0 s of pre-roll for a 2.0 s setting and one delivering 20 gave 1.0 s (audit). The second case
  still holds, because the frame count is the RAM bound.

# Human review feedback loop (2026-10-04)

What exists: a review store (SQLite), a local reviewer page, label export with a manifest, metrics, a
weekly owner report, a retrain hook, and retention with automatic deletion. `src/bree/review/`,
tests in `tests/test_review.py`. One of them drives the page in headless Chrome and is skipped unless
`BREE_PLAYWRIGHT` points at a `node_modules/playwright` folder.

**Everything below is SYNTHETIC.** The alerts are drawn rectangles, the two "reviewers" are a seeded
random number generator, and the theft mix is made up. The table shows that the loop runs and that
the numbers add up. It says nothing about how well the vision pipeline detects theft. No real
reviewer has used the page on real store footage yet.

Reproduce: `make review-demo` (`.venv/bin/python scripts/review/synthetic_e2e.py`, re-run at integration with the same output) (60 synthetic alerts, seed 0, plus one
alert logged twice on purpose to test dedup; same output on every run).

| Measure (SYNTHETIC) | Value |
|---|---|
| Alerts recorded | 61 |
| Of those, from staged tests (counted on their own) | 2 |
| Of those, withdrawn by a late receipt | 0 |
| Customer alerts (every figure below is on these) | 59 |
| Alerts reviewed | 55 |
| Review rate | 0.932 |
| Confirmed theft | 30 |
| Theft, wrong item | 6 |
| Not theft | 13 |
| Unclear | 3 |
| Reviewers disagree | 3 |
| Alert precision | 0.735 |
| Item precision | 0.612 |
| Median time from event to decision | 38,236 s |
| Median time a reviewer looked | 5.6 s |
| Precision by week (W38, W39, W40) | 0.722, 0.722, 0.769 |
| Owner report precision for the same three weeks | 0.722, 0.722, 0.769 |
| Owner report, week of 2026-09-14: sent, staged, customer | 21, 2, 19 |
| Precision by zone and camera (cooler, snacks) | 0.708, 0.76 |
| Alerts seen by two reviewers | 17 |
| Same decision | 14 (0.824) |
| Cohen's kappa | 0.749 |
| Detector examples exported | 37 (6 with a corrected class, 0 with an unknown class) |
| Pick examples exported | 53 |
| Conceal examples exported | 38 |
| Duplicates dropped at export | 2 |
| New examples on a second export | 0 |
| Retrain hook, new examples on first call | 128 (command exit code 0) |
| Retrain hook, new examples on second call | 0 |
| Staged tests: staged, caught, missed | 3, 2, 1 |
| Clips on disk before and after retention | 60, 0 |
| Dataset examples left after retention | 0 |

The precision figures are lower than in the first version of this table (0.745 overall, 0.75 for
W38) because `metrics()` now leaves out the 2 alerts caused by staged tests, the same as the owner
report. Both were confirmed thefts, so counting them had raised the number.

Left out of the export in that run: 3 alerts where the reviewers disagreed, and 16 conceal windows
where no reviewer answered "concealment seen".

Checks on real output from this repo:
- `out/rehearsal/alerts.jsonl` with its `frames.jsonl` ingests as 1 alert with 39 pose frames tagged
  "pick" and its real clip. After one decision the export gives 1 pick example and 0 detector
  examples (no clean frame, see below).
- Ingesting that file from a different working directory still finds the clip, and ingesting it a
  second time adds 0 alerts.
- The reviewer page was driven in Chrome: the clip loaded and played (320 px wide, 1.6 s), key 1
  recorded a decision and loaded the next alert, key 3 with no item typed did not submit, and the page
  made 0 requests to anything other than 127.0.0.1.
- Keyboard flow in headless Chrome on a SYNTHETIC store (`scripts/review/page_keys.mjs`, run by the
  test suite): key 3 leaves the item field empty and focused, typing `chips` and Enter stores
  `wrong_item` with `chips`, and text left in the field is not stored with a later "confirmed theft".
  An earlier version typed the "3" into the field and could store `3chips` as a class.
- Same script, guards against unseen decisions: key 1 held for 30 repeat keydowns took the queue
  from 6 left to 5 left (one decision), five fast taps of key 2 took it from 5 to 4, a key pressed
  right after an alert appeared did nothing, key U reopened the alert just decided and key 4 then
  replaced that decision, key 3 held for 10 repeats left the item field empty, and an item not in
  the list (`chpis`) was stored only after a second Enter. With the repeat, busy and 0.4 s guards
  taken out of the page, the same script on a 14 alert SYNTHETIC store decided all 14 alerts and
  then failed.
- A SYNTHETIC dataset of 4 alerts exported to a folder outside the store (7 `.npz`, 4 `.jpg`, 4 `.txt`)
  has 0 of each after the store is reopened 60 days later.

Not done, and why:
- **Real alerts produce no detector examples yet.** The pipeline has to save a clean, head-pixelated
  frame with the item box when it alerts (`bree.review.store.save_evidence_frame`). `pipeline.py` is
  not in this stream.
- **Alerts are not written to the store live.** They are ingested from `alerts.jsonl` or
  `would_be_alerts.jsonl` with a command (`make e2e-sim` does this on SIMULATED alerts). Wiring `on_alert` to the store is a few lines in the pipeline
  or shadow code, also outside this stream.
- **The retrain hook has only been run with a stand-in command.** The detector fine-tune script now reads
  the manifest (`scripts/train/finetune_real.py --manifest`, integration): it builds the YOLO dataset from
  the detector examples, split by alert, and that step is tested on a SYNTHETIC export
  (`tests/test_detector_train.py::test_review_manifest_feeds_the_finetune_dataset`). No training run has
  been made from review labels, and the pick and conceal windows have no trainer reading them yet.
- **Pipeline alerts have no wall-clock time.** Their event time is the ingest time, so "time from
  event to decision" on real pipeline alerts is really time from ingest to decision until the
  pipeline writes a clock time.
- **No reviewer login.** The reviewer id is typed in. The page only listens on 127.0.0.1 and only
  answers requests addressed to `127.0.0.1` or `localhost`.
- **The head-pixelation check is by folder name only.** Clips must come from a folder named `alerts`
  and frames from one named `frames`. Nothing checks the pixels.
- **No minimum watch time and no full history.** The page stops held keys and double presses, and Back
  goes one alert back. A reviewer can still confirm an alert after 0.4 s without watching the clip.
- **Retractions have only been tested with hand-written records.** No pipeline run with a late receipt
  has been ingested.
- **Time to decision on real alerts is unknown.** The 5.6 s above is a number the generator drew.

# Closed-world identity (2026-10-04)

*Update from the integration: closed world now also runs across cameras as one pool for the store (section "Calibration, 3D slots and store-wide identity" above). The text below is as it was written for the single-camera flag; where it says a multi-camera store falls back to the floor-plan handoff, that is no longer the case.*

**Bottom line.** Kiro's idea is built behind `rules: {closed_world: true}`: a store is a closed room with a door, people are only created at the door, a track that starts anywhere else is one of the shoppers already inside, and identities end at the exit or after 60 minutes unseen. On real footage (MERL) it cuts visits per shopper from 2.43 to 1.36, with or without re-ID, and removes the damage re-ID did there (4.64). **It is not the default yet.** The rule was "clearly better on store data without more false merges". MERL has one shopper per video and cannot show a false merge; the only multi-shopper store test here is scripted, and there closed world without appearance makes more than twice as many wrong joins as today (1,238 vs 554), because position cannot tell two people apart who were lost at the same time. 92% of those are marked uncertain and capped at review, so wrong joins that could silently reach an alert fall from 554 to 105, but the raw count goes up and 62% of visits end up review-only. Closed world plus re-ID beats today on every count, but re-ID is still waiting on legal advice. Everything below is from `results/reid_bench.json` (`make reid-bench`, run 2026-10-03/04); design in DECISIONS.md "Closed-world identity".

**The four settings.** current = position/time stitching + ambiguity guard (the default). re-ID only = `reid: true`. A false merge is a track joined to another shopper's identity; "silent" means it was not marked uncertain, so an alert could rest on it. A split is one extra identity for one shopper.

**MERL Shopping test subjects (real footage, 28 videos, overhead, one shopper per video: every extra visit is a split, no false merge is possible).**
| | current | re-ID only | closed world | closed world + re-ID |
|---|---|---|---|---|
| visits per shopper | 2.43 | 4.64 | **1.36** | **1.36** |
| videos with a split | 68% | 100% | 36% | 36% |
| uncertain marks (28 videos) | n/a | n/a | 62 | 141 |
The lab has no door in view, so this ran on the no-door fallback. The remaining 10 extra visits over 28 videos (out of 43 `no_door` and 39 warm-up births, most gone within 2 s) are tracks that appear while the shopper is still visible and stay or stand apart: a false detection or a second box that lasts. Closed world has no lost person to give those to. The uncertain marks come from the same ghost identities sitting in the lost pool as a second candidate; with re-ID on, more are added because the shopper seen from above often looks different from their own earlier crop (the same effect that broke re-ID only), which now costs a review-tier cap instead of a new person.

**Scripted store (SYNTHETIC tracks, not footage: 1,000 episodes, 3,018 shoppers, 7,046 tracks, 4,028 reappearances after an occlusion; door zone of the example store; half the occlusions hide two shoppers at once and half of those trade places unseen; 5% of entries missed; 25% of extra shoppers dressed like someone else; appearance drawn at side-view quality).**
| | current | re-ID only | closed world | closed world + re-ID |
|---|---|---|---|---|
| correct joins | 802 | 2,895 | 2,729 | 3,652 |
| false merges | 554 | 17 | 1,238 | 314 |
| of those, silent | 554 | 17 | 105 | 26 |
| splits | 3,209 | 1,130 | 1,214 | 329 |
| visits with one clean identity | 23% | 73% | 58% | 87% |
| visits marked uncertain (alerts capped at review) | n/a | n/a | 62% | 17% |
| missed-entry people created | n/a | n/a | 156 | 156 |
This is a stress script (a tracker-id break every 4 to 10 s) written by the same hand as the code, and its appearance model is kinder than an overhead camera. It shows the shape of the trade, not rates to expect in a store: without appearance, closed world swaps splits for wrong joins it knows it is unsure about; with appearance it wins on both. A "split" under closed world is usually two shoppers swapped, not a new person.

**Toy clips (full pipeline, 8 clips, 10 people, 2 thieves).** Same scorecard in all four settings: 2 alerts on thieves, 0 alerts and 0 reviews on honest shoppers. Tracks never break in these clips (10 births at the door, 0 assignments), so they only show the flag does no harm. Pipeline FPS in this run (14.7 / 7.7 / 8.3 / 5.0) was taken on a machine that slept and ran other jobs; not a cost measurement.

**Simulator.** The event-level simulator (`make sim`, `results/bench.json`) does not go through the event engine, so the flag cannot change it. The simulator fixture (`make sim-fixture`) is three cameras, where closed world switches itself off (see limits), so it was not re-scored.

**MOT16 train: sanity check only, excluded from the decision.** Street footage is open world: people walk in and out of every frame edge and most never come back, so the lost pool fills with people who are gone and each new pedestrian is forced onto one of them.
| | current | re-ID only | closed world | closed world + re-ID |
|---|---|---|---|---|
| IDF1 | 0.441 | 0.443 | 0.290 | 0.294 |
| ID switches | 441 | 434 | 354 | 357 |
| joins | 29 | 31 | 554 | 555 |
| false merges | 13 | 7 | 308 | 315 |
As expected it is wrong there, and it knows it: 303 of the 308 wrong joins were marked uncertain. The flags-off rows are identical to the previous run, and the MOT16 regression guard in `make test` now asserts that.

**Decision.** `closed_world` stays off in `EngineRules` and in `configs/store_gas_station_small.yaml`. What is in the way, exactly:
1. Raw false merges rise without re-ID on the only multi-shopper store test (554 to 1,238), even though silent ones fall (554 to 105). If Kiro reads "false merges" as the silent ones, the rule is met and the switch is one line in the store YAML.
2. No real store footage with several shoppers and identity labels. One labelled hour from the pilot camera would replace the scripted table.
3. Re-ID is the setting that makes closed world win outright, and it is waiting on counsel (DECISIONS "Re-identification without the face"). With closed world on, re-ID no longer hurts MERL (1.36 either way).
4. Single camera only: a multi-camera store falls back to the floor-plan handoff.

**Tests.** `tests/test_closed_world.py`: 25 tests, including two shoppers lost at once and reappearing swapped (appearance sorts them out; without it both are marked uncertain), same clothes (one identity each, both uncertain), joint vs greedy assignment, missed entry, no-door fallback, frame border, timeout, features cleared at exit and timeout, basket kept through a loss, alert downgraded to review with the reason in the ledger log, and a seeded scripted-store guard against `tests/fixtures/reid_guard.json`.

**Limits.** The uncertain period defaults to the whole visit; a finer rule (only items picked before the uncertain moment) is not built. The ledger drops a track silent for 1 h on its own, so the engine timeout should not be set above 3600 s. Score weights (0.5 / 0.35 / 0.15) and the 0.1 margin were set once and not fitted. The second-box rule was added after seeing first MERL numbers on train and test; its thresholds are existing ones.

# Re-ID without the face (2026-10-03)

**Bottom line.** Body re-identification is built, tested and benchmarked, and it is **off by default** (`rules: {reid: true}` turns it on). On angled street footage with crowds (MOT16) it roughly halves false merges, the error that puts one shopper's basket on another, and nudges IDF1 up. On the overhead single-shopper lab footage (MERL) the same setting breaks correct stitches and doubles the number of split visits. It does not beat the current numbers on both, so it does not become the default. No face is used, features stay in memory for one visit, and DECISIONS.md "Re-identification without the face" lists the laws to take to a lawyer first. Everything below is from `results/reid_bench.json` (`make reid-bench`) unless another file is named; all public data here is evaluation only.

**What it does.** `src/bree/track/reid.py`: appearance embedding of the body below the shoulder line (DINOv2 ViT-S/14, Apache-2.0 code and weights, ONNX on CPU), clothing colour per body part from pose (upper, lower, shoes), body proportions, carried bag (detector's backpack / handbag / suitcase classes), fused by one logistic score. Used in three places: engine stitching (a candidate that looks different is rejected; someone lost up to 60 s ago is relinked if they look the same and could have walked there), cross-camera handoff on the floor plan, and through those the register link (a shopper lost at the shelf and picked up as a new track at the counter keeps the basket, so the receipt lands on it).

**Re-ID retrieval (MOT16 train ground-truth boxes, one sample per person per second, 2,042 samples of 315 people; query against samples of the same sequence at least 3 s away; 1,735 queries; fusion weights fitted on the other six sequences).**
| cue | rank-1 | mAP |
|---|---|---|
| DINOv2 embedding alone | 58.6% | 39.8% |
| clothing colour per body part alone | 65.0% | 49.4% |
| **fused (embedding, colours, shape, bag)** | **72.8%** | **57.4%** |
Not comparable to published Market-1501 numbers (different protocol, single camera, small low-resolution pedestrians). Market-1501 itself was not downloaded: its terms are research only and the official links are Drive/Baidu; MOT16 was already here under the same evaluation-only rule. The general-purpose embedding is weaker than hand-made colour histograms on this data, which says how much a re-ID-trained model would add if one with a usable licence existed (DECISIONS).

**Tracking on MOT16 train (7 sequences, 517 people, same protocol and detector as the earlier MOT16 table; "before" reproduces `results/mot16.json` exactly).** A false merge is a stitch that joined tracks of two different ground-truth people; stitches involving a track with no ground-truth match are not scored.
| setting | IDF1 | ID switches | MOTA | stitches | correct merges | false merges | false-merge rate |
|---|---|---|---|---|---|---|---|
| no stitching | 0.438 | 454 | 0.327 | 0 | 0 | 0 | n/a |
| **before: position/time + ambiguity guard (default)** | **0.441** | **441** | 0.327 | 29 | 8 | 13 | 62% (13 of 21) |
| guard + re-ID veto only | 0.443 | 435 | 0.327 | 29 | 15 | 7 | 32% (7 of 22) |
| **after: guard + re-ID (veto + long-gap relink), `reid: true`** | **0.443** | **434** | 0.327 | 31 | 17 | 7 | **29% (7 of 24)** |
| exploratory: veto threshold 0.05 | 0.444 | 430 | 0.327 | 41 | 20 | 10 | 33% |
| exploratory: veto threshold 0.5 | 0.438 | 453 | 0.327 | 3 | 2 | 0 | 0% (of 2) |
| exploratory: re-ID without the guard | 0.430 | 374 | 0.327 | 160 | 79 | 45 | 36% |
Reading: the position-only guard was right in only 8 of its 21 scored stitches on crowds, which the earlier IDF1 number hid. Appearance removes six of the 13 wrong merges and, by ruling out the lookalike-by-position candidate, frees nine more correct ones. The IDF1 change itself is small (0.441 to 0.443) because stitches are a small share of all identity errors here; most are tracker switches and missed detections. Re-ID is not good enough to replace the guard (last row).

**MERL Shopping test subjects (28 videos, overhead camera, one shopper per video, so every extra visit is a split and no false merge is possible).**
| setting | visits per shopper | videos with a split |
|---|---|---|
| before: position/time + guard | 2.43 | 68% |
| guard + re-ID veto only | 4.82 | 100% |
| guard + re-ID (`reid: true`, same thresholds as above) | 4.64 | 100% |
The veto rejects the same person: seen from straight above, a body's crop changes with every turn, and the score was fitted on side views. A follow-up sweep on MERL train videos and MOT16 (`results/reid_sweep.json`, exploratory) found no veto threshold that helps one without hurting the other: at 0.02 MOT16 IDF1 is 0.4435 but false merges are back to 13 and MERL train is still worse (2.75 visits vs 1.92); with the veto off and only the long-gap relink, both are within noise of the guard (MOT16 0.441, 11 to 13 false merges; MERL train 1.83 to 1.92 visits, 58% split vs 67%). (The "before" row is 2.43 here vs 2.54 in the Overnight table, which came from tracks cached on 2026-10-01 and an offline duplicate filter; the difference was not investigated.)

**Decision.** `reid` defaults to off. Turn it on per camera where the view is angled and more than one shopper is usually in frame; for an overhead camera set `reid_reject: 0` (relink only) or leave it off. The pilot's own footage should settle the default: it needs a few hours with identities labelled, which is the same labelling already asked for in P2.8.

**Toy clips (full pipeline, 8 clips, 10 people, 2 thieves).** Same scorecard with re-ID off and on (2 alerts on thieves, 0 on honest shoppers); pipeline 15.4 vs 15.0 FPS.

**Cost on CPU (ONNX Runtime, this Mac's CPU, not the edge box).** Embedding 23 ms per crop on 4 threads, 77 ms on 1 thread; colours + shape 0.6 ms per crop; measured inside the pipeline 23 ms per crop for all cues. A track gets features on its first frame and then every 0.5 s, so 5 people in view cost about 230 ms of CPU per second of video. Not yet measured on the actual edge hardware or exported to TensorRT.

**Constant testing.** `tests/test_reid.py`: 19 unit tests (head pixels never reach a feature, lighting change, veto, two-candidate case, shelf to register relink, payment lands on the right shopper, cross-camera cases, features forgotten at exit and never serialised) plus a regression guard that runs the quick benchmark on two MOT16 sequences inside `make test` and fails if IDF1 or false merges with re-ID on get worse than `tests/fixtures/reid_guard.json` or worse than position-only in the same run (about 2 minutes; skipped if MOT16 or weights are missing). `make bench` now runs `make reid-bench` first.

**Not done / limits.**
- Height in metres: floor points give the floor plane only, so a head height cannot be measured without a full camera calibration. Left out.
- Gait: only walking speed, and its fitted weight is 0 because a new track has no motion history when it must be matched. Speed is used as a "could they have walked there" gate.
- Cross-camera handoff and the register link are covered by synthetic unit tests only; there is no multi-camera footage with identities here.
- MOT16 is street scenes, MERL is a lab. Neither is a gas station. The simulator is event-level (no pixels), so it cannot measure re-ID; only the toy clips ran.
- Fusion weights and `reid_long_p` were fitted or chosen on MOT16; the reported MOT16 numbers use leave-one-sequence-out weights, the thresholds are shared.
- Speed numbers were taken while another job was using the machine.

# Overnight (2026-10-01, ~00:20 to ~06:20 EDT)

**Bottom line.** No GPU quota yet (ticket #2610010040000169 still open, limits 0), so everything ran on the Mac. New capabilities are in and tested (148 tests pass, including vision smoke tests), perception tuning and long-gap stitching gave mixed results, and **concealment from pose is still not solved**.

**New capabilities (merged)**
- **Late POS receipts can retract an alert** (`late_receipt_window_s`, default 300 s). Simulator, seed 2, baseline noise, 200 h, with the POS exporting in 60 s batches (new `VisionNoise.pos_batch_s` option; default 0 keeps the headline bench unchanged): without retraction **2.54 false alerts/h at 28.1% precision**; with retraction **0.435/h at 69.0%**; with `exit_grace_s: 65` plus retraction 0.455/h at 72.1% and 47.5% recall, about the live-POS baseline (0.45/h, 72.5%, 47.9%). Source: `results/late_receipts.json` (`scripts/late_receipts.py`). If the operator's POS exports in batches, this matters a lot.
- **Multi-camera stores feed one ledger** (`bree run` with several `--source`/`--store`, `multicam:` in shadow mode); per-camera floor homographies, no appearance features. Tested on synthetic per-camera streams only; a handoff bug that dropped returning shoppers was found and fixed.
- **ONNX runtime** (`--runtime onnx`) with the best ONNX Runtime provider present (TensorRT > CUDA > CoreML > CPU). With the same letterboxed input it gives the same detections as PyTorch within 0.0003 px (`results/onnx_parity.json`); on the pipeline path, padding to 640x640 shifts boxes (median 11 px on MERL). CoreML parity holds on the default compute units used in the speed run too (`results/onnx_parity_coreml_all.json`: same-letterbox boxes within 0.0003 px). **On this Mac, CoreML runs the full pipeline at 77.0 FPS with the small models vs 45.9 FPS for PyTorch MPS** (`results/speed.json`), about 5 camera streams at 15 fps instead of 3.
- **Isaac Sim 4.5 gas-station generator, ready to run** (`src/bree/sim/isaac/`, runbook in its README): store, cameras, behaviours, randomisation, truth files in our schema, converter, overlays, 8/1/1 split by scene seed for the full run (2,080 clips; pilot 52 clips, 11/1/1). Not executed (no GPU). It also fixed `scripts/azure_gpu.sh sim`, which could not have worked (current Isaac Automator only deploys Isaac Sim 5.x; now pinned to v3.13.0 for the 4.5.0 container) and closes the Automator's open VNC/NoMachine ports while restricting SSH to this IP. Isaac Sim 4.5 has no concealment animation, so these renders help detection, tracking and event timing, not concealment poses.

**Biggest gain tonight: pick detection and visit continuity** (event engine; choices made on MERL train videos, measured once on the 28 test videos)
| MERL test split | Phase 2 | + track stitching | + hand point + duplicate-box removal (new defaults) |
|---|---|---|---|
| reach recall (pick detection upper bound) | 64.2% | 64.2% | **87.6%** |
| false reaches / min | 0.78 | 0.78 | 1.07 |
| visits per single-shopper video | 5.8 | 3.1 | **2.5** |
| shoppers split into more than one visit | 100% | 79% | **68%** |
Sources: `results/merl_stitch.json` (Phase 2 column: no stitching), `results/merl_offline_train.json` (selection), `results/merl_offline_test.json` (other columns: current engine incl. the stitching ambiguity guard below). The first stitching run, before the guard, gave 3.0 visits and 82% split (`results/merl_stitch.json`). Stitching continues a visit when a new tracker id appears away from the door within 10 s and one body height of someone just lost (position and time only). The hand point is the wrist pushed half a forearm further (fingertips reach deeper than the wrist). Duplicate removal drops a person box lying 85% inside a larger one. Stitching can mis-merge different people; MOT16 (below) showed that in crowds, so stitching now skips ambiguous cases (two lost people qualify, or someone visible stands where the new track appeared).

**A missed-theft bug found on the way.** The put-back rule counted any hand in a shelf zone; a resting hand next to a gondola turned a concealment by the other hand into a put-back. Now only the holding hand counts (regression test fails on the old code).

**Simulator with the new measured rates** (`results/bench.json`, POS feed, 200 h, seed 2; measured pick detection 87.6%, visit splits 68%):
| vision noise | precision | recall (alert) | recall (alert + review) | false alerts / hour |
|---|---|---|---|---|
| assumed baseline | 72.5% | 47.9% | 83.4% | 0.45 |
| measured, Phase 2 rates (Phase 2 section; that bench.json since overwritten) | 24.2% | 13.5% | 37.4% | 1.05 |
| **measured, rates after tonight** | **28.6%** | **18.0%** | **53.3%** | 1.11 |
| measured tonight, ID switch at the assumed 3% | 54.9% | 28.1% | 79.0% | 0.57 |
More thieves are caught (alert + review recall 37% to 53%) at slightly more false alerts (1.05 to 1.11 per hour). Visit splitting is still the biggest drag: at the assumed split rate, false alerts would be 0.57 per hour. The toy clips are unchanged (2 of 2 thieves alerted, no flags on honest shoppers).

**Tracking on real angled footage with ground truth (MOT16 train, 7 sequences, 517 people; evaluation only; `results/mot16.json`).** Settings A to C (first three rows) fixed before running; the stitching rows were added afterwards:
| setting | MOTA | IDF1 | ID switches | fragmentations | recall | precision |
|---|---|---|---|---|---|---|
| old default (YOLO26s, conf 0.3, 2 s buffer) | 0.325 | 0.435 | 487 | 1,369 | 38.5% | 87.4% |
| **current default** (+ duplicate-box removal) | **0.327** | **0.438** | 454 | 1,357 | 37.9% | 88.6% |
| MERL-tuned (YOLO26n, conf 0.15, 5 s buffer) | 0.300 | 0.395 | 430 | 978 | 34.3% | 89.8% |
| current default + engine stitching, no guard | 0.328 | 0.388 | 290 | 1,390 | 37.9% | 88.6% |
| same, stricter distance (0.5 heights; exploratory) | 0.328 | 0.411 | 308 | 1,383 | 37.9% | 88.6% |
| **current default + stitching with ambiguity guard** | **0.327** | **0.441** | 441 | 1,359 | 37.9% | 88.6% |
Duplicate removal helps slightly on real angled footage; the MERL-tuned detector loses too much recall, which confirms not making it the default. Unguarded stitching cut ID switches by a third but merged different people in crowds (IDF1 0.438 to 0.388); with the ambiguity guard it is slightly better than no stitching (IDF1 0.441, 441 switches) while keeping most of the single-shopper gain on MERL. MOT16 is street scenes with many small pedestrians (hence the low recall); stitching here runs with no door zone.

**Pilot day-one tools** (merged): `scripts/draw_zones.py` (click zone polygons and multi-camera floor points on a camera still, video or RTSP frame; writes the store YAML, validated by loading it back) and **POS CSV import** through a declarative column mapping (`configs/pos_mapping_example.yaml`: columns, time format and timezone, terminal names, POS item names to our SKUs or categories, clock offset). `bree pos-convert` turns an export into our receipt format; with `--video-start` it adds stream time so recorded footage can be replayed with its POS export (`bree run --payments`). Shadow mode reads CSV exports straight from the POS folder. Rehearsed end to end on a toy clip with the sample CSV. **Pilot runbook:** `docs/PILOT_RUNBOOK.md` (before the visit, install day, daily review, what to measure before going live, privacy).

**Tried and dropped (negative results, recorded in DECISIONS.md):**
- Open-vocabulary product detection (YOLOE, text prompts, no training) to see a product in the hand: on MERL train videos it fired near the hand about as often with empty hands as with a product (best: 62% vs 50%; `results/merl_product_in_hand_train.json`). A product detector still needs our own labelled products.
- Long-gap stitching (30 to 120 s when the new track is within 0.25 to 0.5 body heights of where the shopper was lost): on the train cache, visits per shopper 1.58 to 1.50 but split shoppers 33% to 50% (`results/merl_offline_train_extra.json`); left off.
- Unsupervised concealment scoring (distance to normal shopping poses, fit on PoseLift normal only): held-out AUC-ROC RetailS staged 0.531, DCSASS 0.481, UCF-Crime 0.642 (`results/conceal_knn.json`). Like the classifiers, near chance. Conclusion: public pose-only data does not give a concealment signal that transfers; concealment evidence should come from the product leaving the hand near the torso (needs a product detector) and from shadow-mode labels.

**Perception tuning** (chosen on MERL's train split, measured once on the test split; `results/tuning/`, `results/merl_measure_tuned.json`)
| MERL test split (28 videos, 64.3 min) | current default (YOLO26s, conf 0.3, 2 s buffer) | tuned (YOLO26n, conf 0.15, 5 s buffer, new-track 0.15) |
|---|---|---|
| reach recall | 64.2% | 62.7% |
| false reaches / min | 0.78 | 0.25 |
| extra track ids per single-shopper video | 11.0 | 7.3 |
| detector + pose + tracker, ms / frame (MPS; different sessions, same-session train split: 25.0 vs 24.1) | 30.4 | 23.3 |
On the train split the tuned setting had 67.7% vs 57.1% reach recall; that gain did not carry over to the test split. Fewer track splits did carry over (5.3 to 3.8 on train, 11.0 to 7.3 on test). False reaches went up on train (0.00 to 0.22/min) and down on test (0.78 to 0.25/min), so that change is not consistent. Defaults are unchanged; the trade-off is a choice to make on our own camera footage.

**Classifier v2** (features: wrist position relative to hips and torso, speed; augmentation: mirroring, speed change, keypoint dropout, jitter). Selected on PoseLift leave-one-camera-out only: AUC-ROC 0.681 vs 0.585 for v1 (`results/conceal_select.json`, 2 seeds; P2.3's 0.592 for v1 is the same protocol from a different seed run). Held-out sets, scored once (`results/conceal_variant.json`):
| AUC-ROC | v1 (shipped) | v2 + augmentation |
|---|---|---|
| RetailS staged | 0.501 | 0.504 |
| DCSASS (clip level) | 0.517 | 0.541 |
| UCF-Crime shoplifting | 0.671 | 0.685 |
Slightly better everywhere, still near chance on RetailS and DCSASS. v1 stays the shipped model; v2 is kept as `models/conceal_poselift_v2_aug2.pt`.

**Not done overnight:** YouTube or other unlicensed footage (terms of service, and the people filmed never consented); GPU work (no quota).

---

# Phase 2 (2026-09-30): real data

> **Superseded overnight:** the measured pick-detection and ID-switch rates and the simulator "measured" rows below are the 2026-09-30 values. `results/measured_error_rates.json` and `results/bench.json` now hold the overnight values; see the Overnight section above.

**Bottom line.**
- **No GPUs yet.** Azure GPU quota is 0; all 9 automatic quota requests were rejected (`ContactSupport`). Kiro opened support ticket **#2610010040000169** (East US: NC A100 v4 -> 24 vCPUs, NVads A10 v5 -> 72), status Open. So Isaac Sim, the A100 training and TensorRT speed did not run; everything else ran on the Mac (M1 Max).
- **First real measurements of our own perception** (MERL Shopping, real video, overhead camera): the pose model puts a wrist in the shelf zone for **64.2%** of 589 labelled reach instances (assumed pick detection: 92%), and every single-shopper video got more than one track id (mean 12.0) (`results/merl_measure.json`).
- **Concealment from pose does not work yet.** A classifier trained on real shoplifting poses (PoseLift, Apache-2.0) beats a pose-only rule on held-out PoseLift incidents (AUC-ROC **0.649** vs 0.550), but is **at chance** on every dataset it was not trained on: RetailS staged 0.501, DCSASS 0.517 (`results/conceal_*.json`). UCF-Crime 0.671 is the one exception and is weak evidence (see P2.3).
- **False triggers on real normal footage** (RetailS, 36.4 h of real shoppers, evaluation only): the classifier alone would fire **15.4 times per hour** at threshold 0.5 (5.5 at 0.9); the pose-only rule **99 per hour**. Pose-based concealment can only ever corroborate the ledger, never alert on its own.
- **Simulator re-run with the measured rates:** precision falls from 72.5% to **24.2%**, alert recall from 47.9% to **13.5%**, false alerts rise from 0.45 to **1.05 per hour** (POS feed, 200 h, `results/bench.json`). With the assumed ID-switch rate instead of MERL's overhead worst case: 42.9% / 19.6% / 0.65 per hour.
- **Shadow mode is built** (`bree shadow`), so the pilot can collect labelled data from our own cameras without showing staff anything. That data, plus Isaac Sim once quota lands, is the path forward.
- An independent adversarial review found real flaws in the first version of these numbers (incident leakage, empty frames inflating AUC, loose event counting). All were fixed and every result was re-run; see DECISIONS.md, "Adversarial review of the classifier".

## P2.1 What changed since last night
- `bree shadow` + `/review` page + `bree shadow-labels` (Phase 5; README "Shadow mode").
- `src/bree/conceal.py`: PoseLift/RetailS loaders, incident chains, gap splitting, pose-only rule, temporal CNN classifier, metrics, triggers. Scripts: `conceal_experiment.py`, `eval_retails.py`, `eval_dcsass.py`, `eval_ucf.py`, `measure_merl.py`, `measured_error_rates.py`, `speed.py`, `drive_fetch.py`, `azure_gpu.sh`, `phase2_rerun.sh` (re-runs every real-data result in order).
- `make bench` prints a **REAL DATA** section (read from the result files) before the **SIMULATED** section, and adds "measured" noise rows to the simulator.
- `data/README.md` (counts, formats, closeness to a gas station) and `data/LICENSES.md`.
- 98 tests, all passing (the Phase 1 report had 80).

## P2.2 Datasets
| Dataset | License | Used for | What it gave us |
|---|---|---|---|
| PoseLift (official Drive) | Apache-2.0 | **train + eval** | 151 clips (47 labelled) from 6 cameras; 36 incident chains contain labelled clips. Trains the shipped classifier. |
| RetailS | none stated ("academic use only" on an unfinished page) | eval only | 624 staged clips; 36.4 h sampled normal footage. Its real-world test set **is** the PoseLift test set, so not used. |
| MERL Shopping | research only | eval only | 28 test videos, 64.3 min, 589 labelled reach instances. |
| UCF-Crime (authors' Dropbox) | research only | eval only | 21 shoplifting test videos + 15 of 150 normal test videos (0.38 h) through our pipeline. |
| DCSASS (Kaggle) | research only (from UCF-Crime) | eval only | 896 labelled shoplifting-category clips (155 shoplifting) through our pipeline. |
| Simuletic (Kaggle) | CC BY 4.0 | not used | The free release is only 8 clips. |
| SKU-110K | research | not used | Download paused; no gas-station product labels to pair it with. |
Details and counts: `data/README.md`, `data/LICENSES.md`.

## P2.3 Real-data results (no simulator)
**Pose/track layer, MERL Shopping test split** (`results/merl_measure.json`; overhead lab camera; YOLO26s + crop pose + ByteTrack on MPS at 15 fps):
- Reach recall **64.2%**: 378 of 589 labelled "Reach To Shelf" + "Hand In Shelf" instances had a wrist in the shelf zone within +-0.5 s. An **upper bound** on pick detection: a pick also needs the product detected, and MERL's products aren't COCO classes.
- False reaches: **0.78 per minute**.
- Tracking: all 28 single-shopper videos got more than one track id (3 to 22 ids, mean 12.0, i.e. 11.0 extra ids per shopper). Straight-overhead views are hard for a COCO person detector.

**Concealment, PoseLift** (`results/conceal_poselift.json`). Folds hold out whole incident chains (5 folds x 3 seeds); frame metrics on the 3,721 frames with a pose (1,489 shoplifting):
| scorer | AUC-ROC | AUC-PR | EER | per-fold AUC-ROC mean [min-max] | AUC-ROC incl. empty frames |
|---|---|---|---|---|---|
| pose-only rule | 0.550 | 0.435 | 0.475 | 0.560 [0.37-0.74] | 0.673 |
| classifier | **0.649 +- 0.010** | **0.602** | 0.390 | 0.651 [0.49-0.81] | 0.746 |
- Per unique clip, mean over seeds, at threshold 0.5: classifier caught 10.3 of 41 shoplifting clips and triggered on 1 of 6 clean clips; the rule caught 31 of 41 and triggered on all 6 clean clips.
- Leave one camera out, AUC-ROC rule / classifier: cam 1 0.46 / 0.61, cam 2 0.69 / 0.68, cam 3 0.47 / 0.74, cam 4 0.64 / 0.71, cam 5 0.45 / 0.57, cam 6 0.69 / 0.24 (2 clips); mean 0.567 / 0.592.

**Concealment, RetailS** (`results/conceal_retails.json`; evaluation only; model trained on all of PoseLift; leak checks in DECISIONS.md found no shared or near-identical poses):
| scorer | staged AUC-ROC | AUC-PR | EER | staged clips caught (th 0.5) | triggers / hour, 36.4 h normal footage | person tracks (>= 2 s) triggered |
|---|---|---|---|---|---|---|
| pose-only rule | 0.618 | 0.575 | 0.405 | 514 / 622 | 99.2 (th 0.5) | 90.1% of 3,131 |
| classifier | **0.501** | 0.507 | 0.507 | 73 / 622 (11.7%) | **15.4** (th 0.5), 10.6 (0.7), **5.5** (0.9) | 13.9% (0.5), 4.9% (0.9) |
The staged set has only 2 clean clips, so staged "caught" counts say nothing about false alarms; the normal footage does.

**Concealment, DCSASS through our own pipeline** (`results/conceal_dcsass.json`; 896 clips, 155 shoplifting, from 28 UCF-Crime videos; clip score = max frame score):
| scorer | clip AUC-ROC | AUC-PR | clips triggered at 0.5: shoplifting / normal |
|---|---|---|---|
| pose-only rule | 0.412 | 0.144 | 30% / 43% |
| classifier | 0.517 | 0.184 | 0.6% / 2.7% |

**Concealment, UCF-Crime through our own pipeline** (`results/conceal_ucf.json`; 21 shoplifting test videos with the official temporal annotation, 32,272 frames with a person, 3,578 in the incident window; 15 normal test videos, 0.38 h):
| scorer | AUC-ROC | AUC-PR | EER | triggers / hour on normal videos |
|---|---|---|---|---|
| pose-only rule | 0.548 | 0.126 | 0.460 | 789 (th 0.5) |
| classifier | 0.671 | 0.199 | 0.367 | 7.9 (0.5), 2.6 (0.7), 0 (0.9) |
UCF's labels mark the whole incident window, not the concealment itself, and only 0.38 h of normal video was scored, so this is weak evidence; DCSASS, cut from the same source videos with clip labels, shows no signal.

## P2.4 Measured vs assumed vision error rates
From `results/measured_error_rates.json`. Only four have a real measurement; the other 13 simulator parameters stay assumed and say why.
| parameter | assumed | measured | source |
|---|---|---|---|
| pick detected | 92% | <= 64.2% | MERL reach recall (upper bound) |
| concealment detected | 60% | 11.7% | RetailS staged, classifier th 0.5 (never trained on) |
| false concealment | 3% per carried item | 13.9% per person track (per-shopper stand-in) | RetailS normal footage, classifier th 0.5 |
| visit's track splits | 3% | 100% | MERL, overhead camera (worst case) |

**Event-level simulator** (held-out seed 2, 200 h per row, `results/bench.json`; vision error rates are inputs):
| vision noise | payment feed | precision | recall (alert) | recall (alert + review) | false alerts / hour |
|---|---|---|---|---|---|
| assumed baseline | POS | 72.5% | 47.9% | 83.4% | 0.45 |
| **measured** | POS | **24.2%** | **13.5%** | 37.4% | **1.05** |
| measured, ID switch assumed | POS | 42.9% | 19.6% | 64.6% | 0.65 |
| assumed baseline | dwell only | 65.1% | 33.1% | 59.6% | 0.44 |
| measured | dwell only | 23.1% | 12.7% | 29.1% | 1.05 |

## P2.5 False alerts per hour: the number the operator will care about
- **From the simulator with measured rates: 1.05 false alerts per hour** (POS feed), about 25 a day in a 24-hour store; 0.65 per hour if tracking is as good as assumed. Source: `results/bench.json`, "measured" rows.
- **From real footage, concealment signal alone: 15.4 triggers per hour** on 36.4 h of real normal shopping (RetailS, classifier, th 0.5). Not the pipeline's alert rate (an alert also needs an unpaid item from the ledger), but it means concealment-from-pose would flag many honest shoppers.
- A real end-to-end false-alert rate needs our cameras, zones, products and POS: that is what shadow mode is for.

## P2.6 Speed per hardware
Measured on this Mac only, re-run 2026-10-01 with nothing else running (`results/speed.json`; 300 frames of a real 920x680 MERL video; detector + crop pose + ByteTrack + event engine):
| runtime | nano models | small models |
|---|---|---|
| PyTorch CPU (M1 Max) | 25.1 FPS | 16.0 FPS |
| PyTorch MPS (M1 Max GPU) | 49.3 FPS | 45.9 FPS |
| ONNX Runtime CPU | 28.6 FPS | 11.2 FPS |
| **ONNX Runtime CoreML** (`--runtime onnx`) | **89.5 FPS** | **77.0 FPS** |
- A100, A10 and TensorRT: **not measured** (no GPU quota yet).
- Store cameras run about 15 fps, so this Mac handles about 5 streams with the small models on CoreML (77.0 / 15 = 5.1), or about 3 on PyTorch MPS (45.9 / 15 = 3.1); more people per frame means more pose crops, so budget lower.
- (The first run on 2026-09-30, without CoreML, measured 41.2 / 40.5 FPS on MPS while other jobs were running; superseded.)
- **Recommendation (not measured on the device itself):** for the pilot, one Apple-silicon mini PC per station (Mac mini class, roughly $600 list) running `--runtime onnx` (CoreML); measure a Jetson Orin (roughly $250 to $500 list for Orin Nano / NX kits) with TensorRT once we have one, because it is the cheaper ship target. Prices are list prices from memory, to confirm before buying.

## P2.7 Azure resources used
None created. GPU quota is 0 pending the support ticket; no VM, disk or resource group exists, so spend is **$0**. `scripts/azure_gpu.sh` creates `bree-rg` (tagged `project=bree`), locks SSH to this machine's IP, sets auto-shutdown, and `stop` deallocates everything.

## P2.8 Still missing before the pilot
1. **GPU quota** (ticket #2610010040000169) -> Isaac Sim gas-station renders with labelled behaviours and products, A100 training, TensorRT speed.
2. **Footage from our own cameras**: operator stills + a few hours + matching POS export, or the staged session. Every measured rate above comes from someone else's store, camera angle or lab.
3. **A store-specific product detector.** Nothing measures product picks yet; MERL only bounds the reach half.
4. **Tracking on the real camera angle.** MERL's overhead angle broke tracking; an angled ceiling camera must be checked before trusting multi-minute visits.
5. **Concealment that generalises.** Train on synthetic concealment (Isaac Sim) + pilot labels from shadow mode; until then treat it as weak corroboration only. The first live policy should not depend on it.
6. RetailS license answer from the authors (email not sent yet).

---

# Phase 1 report (last night, unchanged)


**Bottom line.** The full pipeline runs end to end, **at 9.7 FPS on a 4-core CPU with no GPU** on real footage:

```
video in → people / products / pose / tracks → store events → per-person basket
         → reconciliation against POS receipts → theft alert (+ short clip, heads pixelated)
```

**When vision is right**, the theft logic (the "ledger") works well. On 200 simulated store hours it scored:
- **97.9% precision** (almost every alert was a real thief)
- **0.04 false alerts per hour**
- **76.4% recall** at alert level: the share of thieves who triggered an alert. That rises to 96.8% if you also count "review" flags.

**Under the vision error rates I *assumed*** for a decent camera, it scores:
- **72.5% precision, 47.9% recall, 0.45 false alerts per hour**
- Across 5 simulator seeds: precision 67–76%, recall 45–53%, 0.40–0.53 false alerts per hour.
- That is roughly 11 false alerts a day in a 24-hour store: **too many to confront anyone on.**

Alerts backed by an observed concealment are the exception: 93% of those were right.

**None of the accuracy numbers come from real store footage; we have none yet.** They describe the logic's behaviour under stated assumptions. The pilot's first job is to replace those assumptions with measurements.

---

## 1. What works right now, and how to see it

Run everything from the repo root after `make setup`. That creates `.venv`, installs pinned dependencies, and downloads the YOLO26 weights.

| What | Command | What you'll see |
|---|---|---|
| Hardware check | `make hw` | This CPU-only box → YOLO26 **nano** models (small models on CUDA / Apple GPUs) |
| Full pipeline on a video, webcam or RTSP camera | `.venv/bin/python -m bree.cli run --source <file\|0\|rtsp://...> --out out/x [--payments pos.jsonl\|stdin\|http:8765] [--no-video]` | Output files listed below this table |
| Toy end-to-end demo | `make demo` | 8 rendered **toy** clips go through the real tracker, event engine, ledger and alerting: 10 of 10 people handled correctly (2 thieves alerted, 0 flags on 8 honest shoppers) |
| Unit tests | `make test` | **80 tests**: ledger 42, event engine 18, simulator 7, payments 5, multi-camera 4, vision smoke tests 4 |
| Ledger accuracy under vision noise | `make sim` | Precision, recall and false alerts per hour at perfect, baseline and 2x vision error rates |
| Everything measurable | `make bench` (~9–10 min) | Writes `results/bench.json` and `results/bench.md` |
| Live dashboard | `make dashboard`, then open http://127.0.0.1:8080 | Camera view; alerts with their reasons; who is in the store and what's in their basket. Replays a toy clip by default; use `--source rtsp://...` for a real camera |
| Edge export | `make export` | ONNX versions of both models |

What `bree run` writes to `--out`:
- `annotated.mp4`: a debug video of everyone, heads pixelated. Skip it with `--no-video`.
- `frames.jsonl`: every person's boxes and keypoints, per frame. No images.
- `events.jsonl`: the store events.
- `alerts/*.json` + `alerts/*.mp4`: one record and one evidence clip per flagged person.
- `ledger_log.txt`: plain-English reasoning for every person.
- `summary.json`: FPS and per-stage timings.

### Perception (`detect/`, `pose/`, `track/`)
- One YOLO26n detector pass finds people and products.
- Body keypoints (COCO-17) come from a pose model run on each **person crop**, not on the whole frame. On one real frame with ~7 small, distant people, the full-frame nano pose model found 0 of them; the crop approach found all 7.
- People are tracked with ByteTrack. Products use a centroid tracker, because ByteTrack's box-overlap matching lost small items moving in a hand.

### Events (`events/engine.py`)
Readable rules, one per event:

| Event | Rule |
|---|---|
| `pick` | A product that was seen in a shelf/cooler zone moves into a hand that was just inside that zone |
| `put_back` | An item leaves the hand inside a zone |
| `conceal` | An in-hand item vanishes inside the shoulders-to-hips box, away from the register, and **stays gone for 1.5 s**. If it reappears, the conceal is cancelled |
| `pay` | The person dwells in the register zone |
| `exit` | The track ends at the door and **stays gone for 3 s** |

Zones are polygons in `configs/*.yaml`. The example single-camera gas station layout has 3 shelf runs, a cooler bank, a register and a door.

### Ledger (`ledger/ledger.py`)
- Keeps a basket per person.
- Credits POS receipts and cooler taps to the right person, first by **what the receipt contains**, then by who was at the counter.
- On exit: basket minus paid = unpaid, scored with an explicit evidence table.
- Tiers:
  - **alert** (tell staff): score ≥ 0.7, **and** at least one unpaid item was either seen being concealed or seen in hand at the exit.
  - **review** (a manager looks later; nobody is confronted): score ≥ 0.4.
- Every decision is explained in plain English.

### Payments (`ledger/payments.py`)
- Accepts JSON lines from a file, from stdin, or via `POST /payments` on a local port; a mock source for tests.
- A POS or tap reader only needs to send `{"terminal", "ts", "items":[{"sku","qty"}]}`.

## 2. Benchmark results

Sources:
- `results/bench.json` and `results/bench.md`, from a full `make bench` at the final code version.
- The event-level section was then re-run with `bench --only event`, after one change to how the threshold sweep counts capped decisions. The toy and real-footage sections come from the full run.
- `results/export.json` for ONNX.

Machine: 4-core x86_64 Xeon @ 2.8 GHz (from `/proc/cpuinfo`), no GPU.

### 2a. Theft logic on simulated shoppers (event-level simulator, no video)

**Setup:**
- 200 simulated store hours per row: 10,404 visitors, 495 of whom stole something (4.8%). The theft rate is deliberately higher than a real store's so recall can be measured.
- The reported seed (2) is **held out**: tuning used seed 1. The alert threshold (0.7) was not changed after seeing seed 2.
- "POS feed" means the register sends item lists.

| vision error rates | precision | recall (alert) | recall (alert or review) | false alerts / hour | false alerts / 1000 honest visitors | honest "review" flags / hour | baskets exactly right |
|---|---|---|---|---|---|---|---|
| perfect vision | 97.9% | 76.4% | 96.8% | 0.04 | 0.81 | 0.03 | 100.0% |
| **baseline (assumed)** | **72.5%** | **47.9%** | **83.4%** | **0.45** | **9.08** | **4.07** | **75.5%** |
| 2x the baseline error rates | 45.4% | 20.0% | 65.1% | 0.59 | 12.01 | 5.44 | 58.0% |

**Seed-to-seed spread** (seeds 2–6, POS feed, mean [min–max]):

| error rates | precision | recall | recall (alert or review) | false alerts / hour |
|---|---|---|---|---|
| baseline | 72.1% [67.0–75.5] | 50.2% [45.3–53.2] | 85.8% [83.4–88.3] | 0.46 [0.40–0.53] |
| perfect vision | 97.6% [96.6–98.5] | 77.8% [76.0–79.5] | — | 0.04 [0.03–0.07] |

Treat any single-seed number as **± ~3–4 points**.

**Without a POS feed** ("dwell" mode: standing at the register counts as paying for everything picked so far, except concealed items), at baseline:
- Recall drops to **33.1%** at alert level (59.6% counting reviews).
- Precision is 65.1%, with 0.44 false alerts per hour.
- A POS feed therefore gives **~1.45x the recall** at alert level (47.9 vs 33.1) and ~1.4x counting reviews. It is also the only way to catch "paid for one, pocketed another" openly.

**How to read the pieces:**
- **Precision and recall** are counted per person. A "thief" is anyone who left with something unpaid.
- **Threshold sweep.** Raising the alert threshold buys few false alerts back and costs a lot of recall. Precision stays at 70–73% between thresholds 0.6 and 0.85:

  | threshold | false alerts / hour | recall |
  |---|---|---|
  | 0.7 | 0.45 | 47.9% |
  | 0.85 | 0.20 | 21.6% |
  | 0.9 | 0.09 | 16.0% |

- **Evidence matters more than threshold.** At baseline:
  - Alerts where a concealment was seen were right 163 of 175 times (**93.1%**).
  - Alerts without one were right 74 of 152 times (48.7%).
  - An alert tier restricted to concealment-backed alerts would have caught 163 of 495 thieves (**~33% recall**), with 12 false alerts in 200 hours (**~0.06 per hour**, about 1.4 a day).
- **Recall by theft type** at baseline, alert level:

  | theft type | alerted | total | alert or review |
  |---|---|---|---|
  | walkout | 143 | 215 | 191 |
  | conceal + partial pay | 89 | 176 | 147 |
  | paid for one item, carried another out openly | 5 | 104 | 75 |

  The last type mostly lands in review only (70 of 104). That is by design: without concealment it looks the same as a missed put-back or a misread receipt.
- **Basket accuracy:** 75.5% of baskets exactly right at baseline (item-level precision 88.9%, recall 89.0%). 100% with perfect vision.
- **Decision latency** (from the person walking out to the alert, in camera time):
  - The median is **5.0 s**, which is the deliberate wait for late POS messages.
  - The 95th percentile is 34.5 s. By design, long waits come from holding a decision while a group mate or a crowded-pick candidate is still inside; I did not measure how much of the tail each accounts for.

**Which vision errors cause false alerts?** Same seed, 200 hours, perfect vision with one error type switched on at its baseline rate:

| error type | false alerts / hour |
|---|---|
| none | 0.04 |
| **missed register visit** | **0.24** |
| track ID split in two | 0.12 |
| two people's track IDs swapped | 0.10 |
| POS message dropped or clock off | 0.10 |
| crowded-pick misattribution | 0.06 |
| missed exit | 0.06 |
| all others | 0.03–0.04 |

- The single-error increases add up to about the whole baseline excess (≈0.41 per hour). So false alerts are mainly driven by **register-visit detection, track stability and POS integration**, not by product recognition.
- Recall is hurt most by **missed concealments** (76.4% → 58.4%) and **missed picks** (→ 68.5%).

**What this proves:** the reconciliation logic, payment attribution, hold-and-wait rules and scoring behave correctly on correct inputs. Payment attribution here covers queues, groups paying together, cooler taps, and late or missing POS messages. It also shows which vision errors to invest in.

**What it does not prove:** real-world accuracy.
- The baseline error rates are my guesses: missed pick 8%, missed put-back 15%, missed conceal 40%, missed register visit 3%, track ID split 3%, ID swap 1%. The full list is in `bench.json → noise_model_baseline`.
- The simulator also encodes my assumptions about how people shop and steal.

### 2b. Full pipeline on toy video clips

**Data:** 8 rendered 2D clips (`make demo`), clearly marked **TOY DATA**. The toy detector reads the **pixels**, not the renderer's ground truth.

| clip | truth | result |
|---|---|---|
| walkout | thief | alert |
| conceal + partial pay | thief | alert (flags the candy bar, not the soda they paid for) |
| normal pay | honest | nothing |
| put-back | honest | nothing |
| crowded cooler | honest | nothing (both picks flagged ambiguous; decision waited for the other person) |
| lingerer | honest | nothing |
| pocket then pay | honest | nothing |
| group, one person pays | honest | nothing |

- Speed: 11.1 FPS over 3,652 frames.
- Time from reading the frame that triggered an alert to having the alert JSON and clip written: 444 ms and 316 ms. This total includes clip encoding; I did not time the steps separately.

**Proves:** the pieces fit together, and the outputs, clips and logs are produced. **Does not prove:** anything about YOLO on real store footage. I wrote these clips alongside the rules they test.

### 2c. Real footage throughput

- **Data:** OpenCV's `vtest.avi`: real pedestrians, 795 frames, 768x576. **No theft labels.** The store zones don't match that scene, so this measures speed and tracking only.
- **Speed:** **9.66 FPS end to end** on 4 CPU cores. Per frame:

  | stage | ms |
  |---|---|
  | detect + pose | 92.6 |
  | tracking | 1.2 |
  | events | 0.8 |
  | ledger | 0.02 |
  | drawing | 7.3 |

  45 person tracks.
- **ONNX** (`results/export.json`, same CPU, measured with nothing else running; a first attempt made while the benchmark was running was discarded):

  | model | PyTorch | ONNX Runtime |
  |---|---|---|
  | detector | 43.8 ms | 46.1 ms |
  | pose (160 px crop) | 13.9 ms | 8.2 ms |

  It found the **same number of detections** on the one test frame (9/9 and 2/2). I did not compare the boxes and keypoints themselves.
- **No GPU numbers:** I had no GPU. Jetson/GPU speed is unmeasured.

### 2d. Adversarial review (independent reviewers tried to break it)

Three independent reviewers ran in parallel. One re-checked every number in this report against the result files. Two built concrete event sequences and frame sequences to break the ledger and the event engine. They found real holes; **all of the following were fixed, each with a regression test that fails on the old code:**

**Event engine — honest shoppers turned into "concealments" or picks:**
- A customer's **own** drink became a "concealment" after a half-second detection dropout, a tracker ID change, or a hidden wrist.
- A set-down item that was **still visible** was flagged as concealed.
- A re-acquired item counted as a second pick.
- The customer's own phone, or an item merely brushed past near a shelf, became a "pick".
- A 1.6 s occlusion at the door became a false exit.
- A passer-by inherited someone else's drink.

**Ledger — false alerts from several weak signals:**
- Two or three missed put-backs added up to an alert. Alerts now need concealment or item-in-hand evidence.
- A vision/POS category mismatch alerted a paying customer.
- A very low-confidence concealment counted in full.

**Ledger — thieves slipping through:**
- A thief who picked two items, pocketed one and put one back lost the pocketed one from their basket.
- A walkout thief could claim a stranger's unattributed receipt.
- Dwell mode cleared concealed items.
- A thief's own concealment still left a crowded pick at "ambiguous".
- A reused track ID hid the second visit.

**Crash and hang bugs:**
- A malformed register event crashed the ledger.
- A floating-point rounding bug at the 120 s hold deadline made the replay loop forever. It appeared on 2 of 6 simulator seeds; a loop guard now prevents any repeat.

**Known and not fixed** (listed so nobody is surprised):
- **Counter tie-break.** At the counter, a tie between a thief who arrived first and the honest payer goes to the thief.
- **Strangers entering together.** A stranger who enters within 4 s of someone is treated as their group, and their extra payment can cover that person's concealed item.
- **No retraction.** Once an alert is out, a receipt that arrives late, or a group mate who pays after 120 s, does not retract it. (Update: a late POS receipt now retracts or downgrades it within `late_receipt_window_s`; see DECISIONS.md, Phase 2. A group mate paying after 120 s still does not.)
- **Unbounded history.** The ledger keeps every person record for the life of the process. Fine for a day; needs rotation for a months-long deployment.
- **Handoffs.** An item handed to someone outside the group stays in the picker's basket.

## 3. What didn't work or got skipped, and why

- **No real shoplifting video was evaluated.**
  - The Kaggle (Simuletic, DCSASS), Mendeley, HuggingFace and Google Drive hosts are **blocked by this build environment's network policy** (HTTP 403).
  - Instructions are in `data/README.md`. `make data` fetches the Kaggle sets once the `kaggle` CLI is installed (`pip install kaggle`) and an API token is present.
- **PoseLift (real shoplifting, pose-only) was not used.**
  - The official data is on Google Drive, which is blocked here.
  - A helper agent found and cloned a third-party GitHub copy into `data/poselift/`. The automated safety check then blocked pulling data from that unofficial mirror.
  - So **no result uses it**, and `scripts/download_data.sh` doesn't reference it. You may want to delete `data/poselift*` and download the official copy instead.
- **Concealment classifier over pose sequences (Stage 5): skipped.** Without PoseLift there was no labelled real pose data, and training on my own synthetic poses would only learn my renderer.
- **Multi-camera:** hand-off between cameras by floor position is built and unit-tested (`track/multicam.py`). It uses homographies (a per-camera mapping from image pixels to floor coordinates) and no appearance features. It is **not wired into `bree run`**: I had no multi-camera footage to run it on. (Update: now wired into `bree run` and `bree shadow`, tested on synthetic per-camera streams only, still no real multi-camera footage; see DECISIONS.md, Phase 2.)
- **Isaac Sim:** prep only, as asked (`src/bree/sim/isaac/`).
  - The IRA config is a draft that has not been loaded by a real IRA build. Its keys must be checked against the installed version.
  - Hand-object behaviours (reach, conceal) need custom animations; the README explains how.
- **Products:** COCO (the detector's training set) knows only "bottle", "cup" and "cell phone" among relevant items. A store-specific product model is needed before product-level results mean anything.
- **Bugs found and fixed while building:** listed in DECISIONS.md, "Bugs found and fixed".
  - Notably, a simulator queue bug combined with a naive "whoever reached the counter first paid" rule produced 7.5 false alerts per hour even with perfect vision.
  - It also turned up a non-reproducible simulation (a Python hash-ordering issue); there is now a test that runs the simulation under 3 hash seeds.

## 4. What I need from you

1. **Footage from the 2 pilot stores:** a few hours per camera, including busy periods, plus a still frame from each camera so I can draw zones.
2. **The POS transaction export** for the same hours (details in section 6). The ledger relies on it: without it, recall drops from 47.9% to 33.1% and partial-pay theft is invisible.
3. **A Kaggle API token** (`~/.kaggle/kaggle.json`), plus `pip install kaggle`. Or run `make data` somewhere with normal internet access.
4. **The official PoseLift download** (public Google Drive link in `data/README.md`).
5. **A GPU box, even briefly:** a Jetson Orin (the likely ship target) or any RTX machine, for real edge FPS and for Isaac Sim.
6. **Decisions:**
   - What may staff do on an alert? My recommendation: nothing customer-facing during the pilot.
   - Who reviews the "review" queue?
   - Clip retention: my suggestion is 7 days for flagged clips.
   - Today, alert evidence clips are only kept for flagged people. But the debug outputs `annotated.mp4` (heads pixelated) and `frames.jsonl` (boxes and keypoints) cover **everyone** unless run with `--no-video`. In production, those debug outputs should be off.

## 5. Honest take: top 3 things before this goes into the 2 gas stations

1. **Run in shadow mode first, then measure the real error rates.**
   - Record the pilot cameras and the POS for 1–2 weeks, with alerts shown to no one.
   - Hand-label a few hours: picks, put-backs, concealments, register visits, exits.
   - Put the measured rates into `VisionNoise` and re-run `make bench`.
   - Only then choose the live policy. At my assumed rates, 0.45 false alerts per hour (~11 a day) is too many to act on in front of customers.
   - The evidence split points to a sane first live mode: **alert only when a concealment was seen**. That was ~93% precise and ~0.06 false alerts per hour at ~33% recall. Everything else goes to a manager's review queue.
2. **Get the POS integration and camera placement right: that's where false alerts come from.**
   - Missed register visits, track-ID splits and swaps, and POS drops or clock drift are the top false-alert sources. Product recognition errors barely matter.
   - Concretely:
     - a camera with a clean view of the counter
     - a POS feed with timestamps synced to camera time (NTP on both)
     - a camera covering the door
     - a second camera on the cooler bank for crowded reaches and concealment
3. **Train a store-specific product detector and validate pose on the real cameras.**
   - COCO can't tell an energy drink from a candy bar.
   - Collect a few hundred labelled frames per camera of the operator's most-stolen items, plus Isaac Sim renders.
   - Check that wrist keypoints from the crop-pose approach hold up at the real camera height and resolution. Every pick, put-back and concealment rule depends on them.

## 6. What to ask the operator for

- **Cameras:**
  - make and model; resolution, FPS and codec
  - whether we can pull RTSP locally (and credentials)
  - mounting height and where each camera points
  - a still frame from every camera in both pilot stores
  - whether he's open to adding 1–2 cameras (counter-facing, cooler-facing)
- **Existing CCTV footage:**
  - a few hours per store, including rush hours and any known theft incidents (with rough timestamps)
  - how long his system keeps footage
- **POS:**
  - which system (e.g. Verifone Commander, Gilbarco Passport, NCR, Clover)
  - real-time export or batch only
  - fields: timestamp, register/terminal ID, line items with SKU/UPC and quantity, voids and refunds
  - how fuel and lottery lines appear
  - whether the POS clock is NTP-synced
  - a sample export covering the same hours as the footage
- **Shrink:**
  - which items get stolen most (energy drinks, beer, candy, lighters, phone accessories, cigarettes?)
  - estimated monthly shrink in dollars per store
  - when theft happens (time of day)
  - common methods (walkouts, pocketing, "paid for one, took two", employee theft)
  - how he finds out today
- **Store layout:**
  - a floor plan or phone photos of each store
  - cooler brand, and whether coolers are already card-tap / locked
  - number of registers
  - typical staff per shift
- **Policy:**
  - what staff may do when an alert fires
  - local rules on camera analytics and signage
  - who at his company signs off on privacy

---

- Design choices and their reasons: `DECISIONS.md`
- Build status: `PROGRESS.md`
- Raw numbers: `results/bench.json`, `results/bench.md`, `results/export.json`
