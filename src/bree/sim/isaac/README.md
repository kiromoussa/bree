# Isaac Sim synthetic data (prep only, not run)

Goal: photoreal, perfectly labelled footage of a gas-station c-store with shoppers
who pick, put back, conceal, pay and walk out, from the camera positions we will
actually install. Used for (1) a store-specific product detector, (2) a real
test of the event engine's pick / put-back / conceal rules on realistic bodies
and occlusions, and (3) tuning `VisionNoise` in the event-level simulator with
measured rather than assumed error rates.

Nothing here was executed: the build machine has no GPU. Needs an RTX GPU
machine (Isaac Sim 4.x/5.x; 24 GB+ VRAM recommended for multi-camera runs).

## Files
| File | What |
|---|---|
| `ira_gas_station.yaml` | Draft Isaac Sim **Replicator Agent** (IRA, `isaacsim.replicator.agent`) config: scene, 3 cameras, characters, writer |
| `commands_mixed_episode.txt` | Draft IRA character command script for one episode covering normal buy, walkout, conceal + partial pay, two strangers at the cooler, lingerer |

**Validate the YAML keys against the IRA version you install.** IRA's schema has
changed between releases (keys, writer names). The structure here follows the
documented layout (global / scene / character / sensor / replicator), but it
has not been loaded by a real IRA build.

## Scene (to build as USD)
Single-room store, ~12 m x 9 m, 3.2 m ceiling, matching `configs/store_gas_station_small.yaml`:
- Back wall: 5-door reach-in cooler bank (glass doors, shelves of bottles/cans: soda, water, energy drinks, beer 6-packs).
- 3 gondola runs (1.4 m high) perpendicular to the cooler: candy/snacks, chips/jerky, auto/phone/lighters.
- Counter with POS terminal and impulse rack (right front); cigarette wall behind counter.
- Glass entry door (left front) with a 1.5 m approach mat.
- Product assets: SimReady / Omniverse retail props where available; otherwise simple textured boxes/cylinders in the right sizes. Each product gets a semantic label = our item **category** (`soda_bottle`, `energy_drink`, `candy_bar`, ...).

## Cameras (what we'd install; mirror them in sim)
| name | mount | aim | purpose |
|---|---|---|---|
| `cam_main` | ceiling corner above the register, 3.0 m | back toward the cooler wall | the single-camera setup in the example config: whole floor, door, counter |
| `cam_cooler` | ceiling, 2.5 m in front of the cooler bank | down at ~60 deg onto the cooler doors | hands at the cooler: picks, put-backs, crowded reaches |
| `cam_door` | above the door, inside | down/in at the approach | exits, held items at exit, handoff between cameras |

Randomise per episode: camera height +-0.2 m, yaw +-5 deg, lighting (fluorescent, dusk, night), lens distortion, compression noise. Real c-store cameras are 1080p @ 15 fps, often H.264 at low bitrate.

## Behaviours and how to get them
IRA ships walking / idling / looking around / queueing. It does **not** ship hand-object interaction. Plan:
1. Paths, dwell, queueing at the register, group arrivals: IRA command scripts (see `commands_mixed_episode.txt`).
2. Reach into cooler / shelf, take item, put item back, move item to pocket / waistband / bag: custom animation clips (mocap, or retargeted from a motion library) played via a custom IRA/omni.anim.people command, with the product prim attached to the hand joint on grasp and detached (placed back, or hidden at the pocket) on release.
3. Ground truth comes from the command scripts themselves: every take / put-back / conceal / pay / exit is written to a per-episode JSON in the same schema as `bree.sim.toy_render` truth files, so `bree.eval.toy_eval` can score it unchanged.

Behaviour mix to render first (mirrors `SimConfig`): normal buyer 50%, browse + put-back 15%, lingerer 10%, group where one pays 8%, pocket-then-pay 5%, walkout 5%, conceal + partial pay 5%, two strangers reaching into the cooler at once 2%.

## Outputs we need from the writer
RGB (H.264-compressed afterwards, to look like CCTV), 2D tight boxes for persons and products, COCO-17 skeletons (IRA can write character skeleton keypoints), instance IDs (tracking ground truth), and the per-episode behaviour JSON. **No faces are needed**: randomised synthetic characters only, so privacy is not an issue for synthetic data.

## Then
1. Fine-tune a product detector on the rendered categories (+ real photos of the operator's top-stolen SKUs).
2. Run `bree run` on rendered episodes and score with `bree.eval.toy_eval` -> measured pick / put-back / conceal / exit detection rates.
3. Put those measured rates into `VisionNoise` and re-run `make bench`.
