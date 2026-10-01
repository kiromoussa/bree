# Isaac Sim 4.5 synthetic gas-station data (Phase 3)

Photoreal, labelled multi-camera footage of a small gas-station c-store: shoppers who pick, put
back, conceal, pay and walk out; two people at the same cooler; a queue at the register; an
employee restocking. Output is in the `bree.sim.toy_render` truth schema, so
`bree.eval.toy_eval` scores it unchanged, plus per-frame boxes, track ids and COCO-17 keypoints
for training.

**Not run yet.** No GPU quota on the build machine. Everything below that does not need Isaac
Sim is tested (`tests/test_isaac_sim.py`); the Isaac Sim part is written against the 4.5.0
extension sources and docs (see "Verified vs assumed") and has to pass the first-run checklist.

## Files

| File | Runs where | What |
|---|---|---|
| `plan.py` | anywhere (repo venv) | Episode plans: store layout and products (per scene seed), camera rig with jitter, lighting, people, omni.anim.people command lines, the product action tied to each command, intended outcome. Split train/val/test by scene seed. |
| `gen_cstore.py` | Isaac Sim 4.5 `python.sh` | Executes plans headless: builds the store, cameras, lights, NavMesh, characters (IRA's own setup calls), moves products, records RGB + 2D boxes + skeletons + camera matrices. Resumable. |
| `convert.py` | any Python with numpy + OpenCV | Raw recordings to clips (`isaac_<episode>_<cam>.mp4/.truth.json/.payments.jsonl/.store.yaml/.frames.jsonl`), overlays, label checks. |

Terms: a **scene** (scene seed) fixes layout, product placement and camera rig; each scene is
recorded as one or more **episodes** (takes: new people, lighting, small camera jitter); a
**clip** is one camera of one episode (the unit the pipeline and `toy_eval` consume).

| Mode | Scenes x takes | Episodes | Clips (4 cameras) | Split by scene seed % 10 |
|---|---|---|---|---|
| pilot | 13 x 1 (seeds 90000+) | 13 | 52 | 11 / 1 / 1 |
| full | 130 x 4 (seeds 1-130) | 520 | 2,080 | 416 / 52 / 52 episodes |

## The scene

12 x 9 m store, 3.2 m ceiling, Z up, metres, origin at the centre, +y toward the cooler wall.
Built from primitives with semantic labels = our item categories, because the 4.5 asset pack has
no c-store merchandise (SimReady `common_assets/props` is boxes, pallets, racks, fruit, a milk jug).

- Cooler bank on the back wall: 5 glass doors (hinged, swing open during a reach), 5 shelves, soda / energy drinks / water / beer packs.
- 3 two-sided gondolas (1.4 m) perpendicular to the coolers: candy, chips + jerky, phone accessories + lighters (groups shuffled per scene, ~12% empty facings).
- Counter on the east side with a POS terminal, an impulse rack (candy, lighters) and a cigarette wall (decor). 0.8 m staff aisle behind it.
- Entrance door at the front-left; shoppers wait outside behind the opaque front wall, so they are off camera until they enter.
- Cameras (ceiling, 2.5-3.0 m, looking down): `cam_main` corner above the register, `cam_cooler` facing the coolers, `cam_door` facing the door, `cam_register` above the register. Per scene +-0.2 m / target +-0.35 m and focal +-8%; per episode +-0.04 m more. 1280x720, 15 fps out (30 fps sim).
- Lighting per episode: day (sky dome through the door), night (dimmer, warmer), fluorescent flicker (1-3 panels drop out at random); colour temperature 3200-6500 K.
- People: 1-8 shoppers from 16 Isaac/People characters (4 more are used for staff), a cashier (85% of episodes), a restocking employee (20%). Behaviour mix: normal buy 36%, put-back 12%, browse 10%, conceal + partial pay 8%, conceal + walkout 7%, pocket-then-pay 5%, walkout in hand 7%; plus two-at-the-cooler (15% of episodes) and crowded register (15%). Concealment anchors: pocket, waistband, under clothing, bag (bag prop follows the hip).

## How hand-object interaction works (not in IRA)

1. **Reach** = the `PushButton` custom command: `Isaac/People/Animations/push_button.skelanim.usd`,
   registered with omni.anim.people's `CustomCommandManager` before the animation graph is built.
   Forward kinematics on its 101 frames: the right wrist goes from 1.0 m to 1.4 m high and 0.5 m
   forward, peaking ~1.2 s in. The plan puts each shopper so that point lands on the product.
   Only products whose centre is 1.0-1.6 m high are picked (no crouch animation exists).
2. **Which product**: `gen_cstore.py` subscribes to `omni.anim.people/CommandStartEvent` (payload:
   `agent_name`, `command` = the command line) and looks up the action the plan tied to that line.
3. **Grasp**: from `--grasp-at` s after the reach starts, the product prim follows the hand joint
   every frame (`omni.anim.graph.core` `Character.get_joint_transform`, world space; R_Hand or
   R_Wrist, whichever answers). A second pick moves the first item to the left hand.
4. **Put-back / restock**: the product is set back on its slot at the grasp moment.
5. **Conceal**: during a 1.2 s `Idle`, the product slides from the hand to the anchor and is made
   invisible (state `hidden`). There is **no concealment arm motion** in the 4.5 asset pack, so the
   body pose does not show it. Train the pose-based conceal classifier on PoseLift, not on this; use
   these clips for boxes, tracking, product state and event timing.
6. **Pay**: `Queue` -> `PushButton` (card tap at the terminal) -> `Dequeue`; the POS receipt (SKUs of
   the items planned as paid) is time-stamped at the tap.
7. **Enter / exit**: when the character root crosses the door line.

Ground truth in `truth.json` comes from what actually happened (event log), and each person row
has `matches_plan` so a failed command shows up instead of silently mislabelling.

## Runbook

Prerequisites on the Mac: `az login` done, GPU quota for `Standard NVADSA10v5 Family` (36 vCPUs),
`export NGC_API_KEY=...`, Docker running (Isaac Automator runs in a container).

```bash
cd ~/bree-vision
scripts/azure_gpu.sh quota eastus
scripts/azure_gpu.sh sim eastus      # Isaac Automator v3.13.0 -> NV36ads_A10_v5 with nvcr.io/nvidia/isaac-sim:4.5.0
```
The first deploy asks for an Azure device login inside the Automator container (a human opens the
URL and types the code once). The script then replaces the open NSG rules with SSH-from-this-IP,
tags every `bree-sim` resource `project=bree`, sets auto-shutdown and prints the ssh command.
Log the image tag and VM hours in `PROGRESS.md`.

**1. Plans + code to the VM** (Mac)
```bash
IA=../IsaacAutomator
PYTHONPATH=src .venv/bin/python -m bree.sim.isaac.plan --mode pilot --out $IA/uploads/bree_plans/pilot
PYTHONPATH=src .venv/bin/python -m bree.sim.isaac.plan --mode full  --out $IA/uploads/bree_plans/full
rsync -a --delete --exclude .git --exclude .venv --exclude data --exclude out --exclude runs ./ $IA/uploads/bree-vision/
(cd $IA && ./run ./upload bree-sim)          # -> ~/uploads on the VM, /uploads in the container
KEY=$IA/state/bree-sim/key.pem
IP=$(az vm list -d -g bree-rg --query "[?contains(name,'bree-sim')].publicIps | [0]" -o tsv)
ssh -i $KEY ubuntu@$IP
```

**2. One-time VM setup**
```bash
nvidia-smi                                   # GRID driver >= 535.129.03 (Automator v3.13.0 installs 535.161.08)
docker images | grep isaac-sim               # nvcr.io/nvidia/isaac-sim:4.5.0, pulled by the Automator
docker kill isaacsim 2>/dev/null || true     # GUI container, if it was started
sudo apt-get update -qq && sudo apt-get install -y -qq tmux python3-venv
python3 -m venv ~/cv && ~/cv/bin/pip install -q numpy opencv-python-headless pyyaml
```

**3. Pilot** (13 episodes = 52 clips), in tmux so an SSH drop does not kill it
```bash
tmux new -s gen
~/isaacsim.sh --container-name=bree-gen --cmd="/isaac-sim/python.sh /uploads/bree-vision/src/bree/sim/isaac/gen_cstore.py \
   --plans /uploads/bree_plans/pilot --out /results/raw_pilot" 2>&1 | tee ~/results/gen_pilot.log
```
`~/isaacsim.sh` is the Automator's launcher (docker run with GPU, `ACCEPT_EULA=Y`, shader/asset
caches on disk, `~/results -> /results`, `~/uploads -> /uploads`). The first start compiles shaders
(NVIDIA's 4.5 benchmark: 278 s non-async startup including shader install, 4 s cached).

**4. Convert, check, overlay** (files from the container are root-owned, hence sudo)
```bash
cd ~/uploads/bree-vision
sudo PYTHONPATH=src ~/cv/bin/python -m bree.sim.isaac.convert convert ~/results/raw_pilot ~/results/bree_sim_pilot
sudo PYTHONPATH=src ~/cv/bin/python -m bree.sim.isaac.convert check   ~/results/bree_sim_pilot
sudo PYTHONPATH=src ~/cv/bin/python -m bree.sim.isaac.convert overlay ~/results/bree_sim_pilot --n 5
grep -h WARNING ~/results/gen_pilot.log | sort | uniq -c | sort -rn | head -30
```
On the Mac: `scp -i $KEY "ubuntu@$IP:results/bree_sim_pilot/overlays/*.mp4" data/isaac_pilot_overlays/` and
watch all 5 (boxes on people and products, ids stable across frames, skeleton on the body, the item
in the hand after a PICK, gone after a CONCEAL, back on the shelf after a PUT_BACK, zone outlines on
the right fixtures). Go through the checklist below; fix and re-run the pilot (`rm -r ~/results/raw_pilot`
or delete single episodes' `done.json`) until it is clean.

**5. Full run** (520 episodes = 2,080 clips): generator and converter side by side
```bash
tmux new -s gen   # window 1
~/isaacsim.sh --container-name=bree-gen --cmd="/isaac-sim/python.sh /uploads/bree-vision/src/bree/sim/isaac/gen_cstore.py \
   --plans /uploads/bree_plans/full --out /results/raw_full <knobs fixed in the pilot>" 2>&1 | tee -a ~/results/gen_full.log
# window 2 (Ctrl-b c): converts each finished episode and deletes its JPEGs (the OS disk is 256 GB)
cd ~/uploads/bree-vision && sudo PYTHONPATH=src ~/cv/bin/python -m bree.sim.isaac.convert convert \
   ~/results/raw_full ~/results/bree_sim --delete-raw --watch
```
It resumes after a stop/deallocate (episodes with `done.json` are skipped; the converter skips
episodes with `converted.json`). If `nvidia-smi` shows the GPU well under 70% busy, run a second
generator with `--container-name=bree-gen2 ... --shard 1/2` and the first with `--shard 0/2`.
Progress: `ls ~/results/raw_full/*/done.json | wc -l`.

**6. Copy to bree-train** (VM-to-VM SSH is blocked by design; go through a storage account)
```bash
# Mac
SA=breesim$RANDOM
az storage account create -g bree-rg -n $SA --sku Standard_LRS --tags project=bree --allow-blob-public-access false -o none
az storage container create --account-name $SA -n sim --auth-mode login -o none
END=$(date -u -v+2d +%Y-%m-%dT%H:%MZ)
SAS=$(az storage container generate-sas --account-name $SA -n sim --permissions acdlrw --expiry $END \
      --auth-mode login --as-user -o tsv)      # secret: never echo or commit it
# both VMs: install azcopy once
wget -qO- https://aka.ms/downloadazcopy-v10-linux | tar xz && sudo cp azcopy_linux_amd64_*/azcopy /usr/local/bin/
# sim VM (pass SAS in the environment of the ssh command, not in a file)
azcopy copy ~/results/bree_sim "https://$SA.blob.core.windows.net/sim?$SAS" --recursive
# bree-train
azcopy copy "https://$SA.blob.core.windows.net/sim/bree_sim?$SAS" ~/bree-vision/data/ --recursive
```
Fallback for small sets: `rsync -az -e "ssh -i $KEY" ubuntu@$IP:results/bree_sim_pilot/ data/bree_sim_pilot/` to the Mac, then to bree-train.

**7. Deallocate** (Mac): `scripts/azure_gpu.sh stop && scripts/azure_gpu.sh status` (or
`(cd ../IsaacAutomator && ./run ./stop bree-sim)`). Never `./destroy`, see `scripts/azure_gpu.sh`.

**8. Score with the pipeline** (bree-train, after Phase 4 has a product detector)
```python
from bree.eval.toy_eval import run_toy_suite, score_rows
from bree.cli import make_backend
rows, _ = run_toy_suite("data/bree_sim/test", "out/isaac_test", "configs/store_gas_station_small.yaml",
                        prefix="isaac_", make_backend=lambda store: make_backend("yolo", store))
print(score_rows(rows))
```
Each clip carries its own `.store.yaml` (zones projected into that camera); cameras that cannot see
the door have no exit zone, so score exits on `cam_main` / `cam_door` clips.

## Expected runtime on an A10

Not documented for the A10. NVIDIA's 4.5 benchmark page gives, for two 720p cameras: 79 MP/s
rendered with RGB-D annotators only and 5 MP/s with every annotator (RTX 3070, Ubuntu; 163 / 5.3 on
an RTX 4080). One captured frame here is 4 x 0.92 MP plus boxes and skeletons, and a 70 s episode
is ~1,050 captures, so somewhere between ~1 and ~13 minutes per episode plus 0.5-1.5 min of setup
(stage, NavMesh bake, characters); the full run is roughly 20-130 A10-hours. The pilot measures it:
`meta.json` has `wall_s` and `wall_s_per_video_s` per episode. Decide shards and the full-run size
from those numbers, and log them.

## First-run checklist

Each item names where to look. Everything marked "assumed" in the next section is on this list.

1. Container starts, `nvidia-smi` inside it works, log shows the extensions enabled and no Python import errors.
2. Assets resolve: `get_assets_root_path()` gives the 4.5 S3 root; characters load (not the fallback `<folder>.usd` name).
3. `PushButton` registered (else the script exits) and visibly plays: the right arm reaches in the overlays.
4. NavMesh: log `navmesh baked`; characters walk through the door and down 0.8-1.7 m aisles; `meta.people_done` all true; no `hit max_duration_s`.
5. Facing: `meta.diagnostics_median.facing_err_deg` under ~30. If ~90/180, re-run with `--facing-offset 90|180|270`.
6. Grasp timing: `diagnostics_median.reach_peak_s` vs `--grasp-at 1.2`; set `--grasp-at` to the measured peak.
7. Hand: `meta.hand_joints` shows `R_Hand`/`R_Wrist` (not null); `reach_miss_m` median under 0.35 m; `check` reports `held_product_near_wrist` above ~0.8.
8. Exposure: no brightness warnings; tune `--light-scale` / `--panel-intensity`. Glass doors render as glass (OmniGlass) and swing without cutting through shoppers.
9. No `unplanned command` warnings (the event payload matched the command lines).
10. `check`: `reprojection_px` under ~3 (3D joints projected with the camera matrices land on the annotator's 2D joints; this validates the zone projection too).
11. `check`: `visible_kps_per_person` around 8-12; nose between the eyes, ears unlabeled.
12. Concealed products have no box after the CONCEAL; put-back products are back on the shelf.
13. Queue episodes: everyone lines up and pays (one `pay` per payer in `events`); restocking employee places items.
14. `manifest.jsonl`: `plan_mismatch` empty (or explained), `zones` per camera sensible.
15. Timing per episode noted; full-run hours estimated and logged in `PROGRESS.md`.

## Verified vs assumed

The 4.5.0 docs were removed from docs.isaacsim.omniverse.nvidia.com (404 as of 2026-10-01); they are
cited from the Wayback Machine. The 4.5.0 extension sources were read directly from NVIDIA's 4.5.0.0
pip wheels (`isaacsim-extscache-kit` contains isaacsim.replicator.agent.core 0.5.14, omni.anim.people
0.6.7, omni.anim.graph.core 106.5.1, omni.anim.navigation.core 106.4.0, omni.replicator.core 1.11.35).

Verified (4.5 docs, 4.5 extension source, or 4.5 assets):

| Item | Source |
|---|---|
| IRA extension names `isaacsim.replicator.agent.core` / `.ui`; config sections global/scene/sensor/character/robot/replicator, `version: 0.5.x`, `simulation_length` in frames at 30 fps; `camera_list` = camera prim paths; NavMesh required | [IRA tutorial 4.5](http://web.archive.org/web/2025/https://docs.isaacsim.omniverse.nvidia.com/4.5.0/replicator_tutorials/tutorial_replicator_agent.html), `config/default_config.yaml` in the wheel |
| Headless IRA runs via `tools/agent_sdg/sdg_scheduler.py -c config.yaml` (container only) | same page |
| Writers `IRABasicWriter`, `TaoWriter`, `StereoWriter`; params `object_info_bounding_box_2d_tight/_loose/_3d`, `agent_info_skeleton_data`; default `semantic_filter_predicate: class:character\|robot;id:*` | [Writer control 4.5](http://web.archive.org/web/2025/https://docs.isaacsim.omniverse.nvidia.com/4.5.0/replicator_tutorials/ext_replicator-agent/writer_control.html), `data_generation/writers/writer.py` |
| Commands `GoTo x y z angle\|_`, `Idle s`, `LookAround s`, `Sit prim s`, `Queue` / `Queue_Spot q i x y z angle` (i from 0) / `Dequeue q x y z angle`; custom commands = animation USD with `CustomCommandName/Template/AnimStartTime/AnimEndTime` (Timing, TimingToObject, GoToBlend); sample `push_button` and `type_keyboard` | [Agent control 4.5](http://web.archive.org/web/2025/https://docs.isaacsim.omniverse.nvidia.com/4.5.0/replicator_tutorials/ext_replicator-agent/agent_control.html), `omni/anim/people/scripts/*` |
| `CommandStartEvent` / `CommandEndEvent` payloads (`agent_name`, `command`, `status`); command file setting `/exts/omni.anim.people/command_settings/command_file_path`, read on play; character root `/World/Characters` | `character_behavior.py`, `commands/base_command.py`, `config/extension.toml` |
| Character setup = `SimulationManager.load_default_skeleton_and_animations` (Biped_Setup + `populate_anim_graph`), `setup_animation_graph_to_character`, `setup_python_scripts_to_character`, `CharacterUtil.load_character_usd_to_stage`, `ApplyNavMeshAPICommand` + `NavSchema.NavMeshExcludeAPI` | `isaacsim/replicator/agent/core/simulation.py`, `stage_util.py` |
| NavMesh: `NavMeshVolume` prim, `nav.acquire_interface().start_navmesh_baking()`, `EVENT_TYPE_NAVMESH_UPDATED` | IRA `simulation.py`, omni.anim.navigation.core tests |
| `omni.anim.graph.core.get_character(skelroot).get_joint_transform(name, Float3, Float4)` in world space, `get_world_transform`; only while the timeline plays | `include/ICharacter.h`, `tests/test_apis.py` |
| Replicator: `rep.create.render_product(path, (w, h), name=)`, annotators `rgb`, `bounding_box_2d_tight` (`semanticTypes`), `skeleton_data`, `camera_params` and their output fields; `rep.orchestrator.step(rt_subframes, pause_timeline, delta_time)`; `/omni/replicator/captureOnPlay`; `/app/player/useFixedTimeStepping`; `rp.hydra_texture.set_updates_enabled`; `add_update_semantics`; `CreateMdlMaterialPrim` with OmniPBR | [Replicator getting started 4.5](http://web.archive.org/web/2025/https://docs.isaacsim.omniverse.nvidia.com/4.5.0/replicator_tutorials/tutorial_replicator_getting_started.html), [Useful snippets 4.5](http://web.archive.org/web/2025/https://docs.isaacsim.omniverse.nvidia.com/4.5.0/replicator_tutorials/tutorial_replicator_isaac_snippets.html), [Randomizers 4.5](http://web.archive.org/web/2025/https://docs.isaacsim.omniverse.nvidia.com/4.5.0/replicator_tutorials/tutorial_replicator_isaac_randomizers.html), `annotators_default.py` |
| `cameraViewTransform` is used as world-to-view in row-vector form (IRA takes the camera position from `inv(view)[3, :3]`); the annotator docstring calls it camera-to-world, so `convert.py` also checks it at runtime | IRA `object_info_manager.py`, `writers/stereo.py` |
| Assets: `Isaac/People/Characters/` (21 character folders + `biped_demo` + `Biped_Setup.usd` + `filter.json`), `Isaac/People/Animations/push_button.skelanim.usd`; characters Z-up metres, Reallusion joint names (`R_Hand`, `L_Upperarm`, `L_Eye`, ...), facing -y in bind pose; Biped joints `R_Wrist`, `L_UpArm`; no c-store goods in SimReady | [asset bucket](http://omniverse-content-production.s3-us-west-2.amazonaws.com/?prefix=Assets/Isaac/4.5/Isaac/People/&delimiter=/), files inspected with usd-core |
| Container `nvcr.io/nvidia/isaac-sim:4.5.0`, Linux driver 535.129.03, GPUs without RT cores (A100/H100) unsupported | [Requirements 4.5](http://web.archive.org/web/2025/https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html), [Container 4.5](http://web.archive.org/web/2025/https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_container.html) |
| Isaac Automator v3.13.0: `--isaac-image`, `--isaac-instance-type` (default `Standard_NV36ads_A10_v5`), `--ngc-api-key`, `--resource-group` (resource ID, imported into terraform), `~/isaacsim.sh` launcher, GRID 535.161.08, NSG rules open to `*`, `./stop` = `az vm deallocate`. v4.x installs Isaac Sim from GitHub source tags (v5.0.0+ only) | [IsaacAutomator v3.13.0](https://github.com/isaac-sim/IsaacAutomator/tree/v3.13.0), [current README](https://github.com/isaac-sim/IsaacAutomator) |
| PIL and PyYAML ship with 4.5 (omni.kit.pip_archive, omni.services.pip_archive) | wheel listing |

Assumed (diagnostic or knob in brackets):

- `PushButton` plays on the retargeted Isaac/People characters the way it does on the biped. [overlay, `reach_peak_s`]
- `get_joint_transform` answers to the character's own joint names (R_Hand) or the biped's (R_Wrist). [`hand_joints`, body-offset fallback]
- GoTo angle 0 faces -y for animated characters, as in the bind pose. [`facing_err_deg`, `--facing-offset`]
- Light intensities give a sane exposure. [brightness warning, `--light-scale`, `--panel-intensity`]
- `OmniGlass.mdl` exists with a `glass_color` input. [falls back to OmniPBR]
- Default NavMesh agent settings fit 0.8 m aisles and the 1.4 m door. [`people_done`, `max_duration` warning]
- The event payload `command` string equals our command line. [match falls back to command name; `unplanned command` warning]
- `skeleton_data` `get_data()` returns the flat dict (the string form is also handled).
- Runtime per episode on an A10 (estimate above).

## Known limits

- No concealment, crouch or left-hand reach animation in 4.5: pose data shows walking, a right-arm reach and standing; concealment is visible only as the product leaving the hand.
- Products are primitives with category colours, not textured SKUs: fine for category-level detection and tracking, weak for SKU recognition.
- Cooler doors are glass panels without reflections tuning; the entrance door is open.
- One store layout family (randomised placement, not a different floor plan per scene).
