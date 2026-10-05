# Plate reader and drive-off rule: measured on SYNTHETIC data

Every plate is drawn with system fonts on a crude car shape and every scene is scripted (`bree.plates.synth`, `bree.plates.scenarios`). None of it is real footage, so these numbers show how the pipeline reacts to plate size, light and noise. They are not a forecast for a real forecourt.

Command: `.venv/bin/python scripts/plates/bench.py` (seed 20261005, 994 s). The reader table below is from that full run (779 s). The vote calibration and the scripted timelines were rerun afterwards with `.venv/bin/python scripts/plates/bench.py --scenarios-only` (215 s), which keeps the reader table.

## Plate read accuracy by plate width and condition

Per cell: 150 plate crops for the OCR columns, 30 plates x 5 frames for the vote, 30 forecourt frames for the detector columns. Text reader: fast-plate-ocr `cct-s-v2-global-model`. `open` = open-image-models YOLOv9-t plate detector run inside the vehicle box, `cv` = the OpenCV-only localiser, `both` = candidates from both, keeping the one the text reader is most sure about. Plate found means a box with IoU of 0.4 or more with the true plate; a looser box can still be read, so whole plate can be above plate found.

| plate width px | condition | OCR char | OCR whole plate | vote of 5 whole plate | open: plate found | open: whole plate | cv: plate found | cv: whole plate | both: whole plate |
|---|---|---|---|---|---|---|---|---|---|
| 32 | day | 49.4% | 3.3% | 20.0% | 96.7% | 6.7% | 0.0% | 0.0% | 6.7% |
| 32 | day_motion | 49.3% | 4.0% | 30.0% | 93.3% | 3.3% | 0.0% | 0.0% | 3.3% |
| 32 | night | 38.7% | 0.0% | 3.3% | 90.0% | 0.0% | 3.3% | 0.0% | 0.0% |
| 32 | night_noisy | 2.3% | 0.0% | 0.0% | 3.3% | 0.0% | 3.3% | 0.0% | 0.0% |
| 48 | day | 91.3% | 62.0% | 86.7% | 90.0% | 43.3% | 100.0% | 46.7% | 46.7% |
| 48 | day_motion | 94.3% | 72.0% | 96.7% | 90.0% | 43.3% | 100.0% | 53.3% | 53.3% |
| 48 | night | 88.5% | 54.0% | 96.7% | 90.0% | 33.3% | 83.3% | 20.0% | 30.0% |
| 48 | night_noisy | 0.0% | 0.0% | 0.0% | 3.3% | 0.0% | 0.0% | 0.0% | 0.0% |
| 64 | day | 97.8% | 88.7% | 96.7% | 70.0% | 56.7% | 86.7% | 73.3% | 73.3% |
| 64 | day_motion | 92.7% | 71.3% | 86.7% | 73.3% | 53.3% | 93.3% | 76.7% | 76.7% |
| 64 | night | 95.3% | 82.0% | 90.0% | 100.0% | 76.7% | 40.0% | 23.3% | 66.7% |
| 64 | night_noisy | 0.2% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 96 | day | 99.3% | 96.7% | 96.7% | 56.7% | 56.7% | 96.7% | 86.7% | 86.7% |
| 96 | day_motion | 97.4% | 88.7% | 90.0% | 50.0% | 46.7% | 66.7% | 83.3% | 83.3% |
| 96 | night | 98.6% | 90.7% | 93.3% | 100.0% | 100.0% | 40.0% | 66.7% | 100.0% |
| 96 | night_noisy | 11.9% | 0.0% | 3.3% | 43.3% | 0.0% | 0.0% | 0.0% | 0.0% |
| 160 | day | 100.0% | 100.0% | 100.0% | 66.7% | 66.7% | 90.0% | 73.3% | 73.3% |
| 160 | day_motion | 99.8% | 98.7% | 100.0% | 66.7% | 66.7% | 96.7% | 80.0% | 80.0% |
| 160 | night | 99.7% | 98.0% | 96.7% | 100.0% | 93.3% | 40.0% | 56.7% | 86.7% |
| 160 | night_noisy | 65.5% | 16.7% | 46.7% | 76.7% | 13.3% | 3.3% | 0.0% | 13.3% |

Conditions (brightness gain, sensor noise sigma in grey levels, motion blur as a fraction of plate width): day: 1.0, 2.0, 0.0; day_motion: 1.0, 2.0, 0.04; dusk: 0.45, 6.0, 0.02; night: 0.18, 10.0, 0.02; night_noisy: 0.1, 16.0, 0.04.
Text reader speed: 19.9 ms per plate crop on this laptop CPU while other jobs were running.

## How often a voted plate is right, by vote confidence

360 drawn plates with seed 20261006 (not the seed of the table above): widths 48, 64 and 96 px, day, day with motion blur and night, 5 frames per plate, one vote per plate. The drive-off monitor stores a plate only at vote confidence 0.9 or more (`min_plate_conf`); this table is what that threshold rests on. The monitor votes over 3 to 12 frames, so 5 is a stand-in.

| vote confidence | plates | voted plate right | share right |
|---|---|---|---|
| 0.9 and up | 319 | 313 | 98% |
| 0.8 to 0.9 | 20 | 9 | 45% |
| under 0.8 | 21 | 4 | 19% |

## Drive-off rule on scripted timelines

Each timeline is rendered at 2 frames per second and played through `DriveOffMonitor` with plates read from pixels, once with the open plate detector and once with both detectors. Vehicle boxes are the scripted ones. Grace period 20 s here (default 120 s) so a timeline stays short. A plate is stored only when the vote over frames reaches confidence 0.9.

| scenario | what happens | light | alerts expected | alerts raised | retractions expected | retractions | correct | plate truth -> read (open) | plate truth -> read (both) | plates left in store (open run) |
|---|---|---|---|---|---|---|---|---|---|---|
| pay_inside | fuels, pays inside, then leaves | day | 0 | 0 | 0 | 0 | yes | n/a | n/a | 0 |
| prepaid | pays at the pump, fuels, leaves | day | 0 | 0 | 0 | 0 | yes | n/a | n/a | 0 |
| drive_off | fuels and leaves without paying | day | 1 | 1 | 0 | 0 | yes | 7XYZ123 -> 7XYZ123 | 7XYZ123 -> 7XYZ123 | 1 |
| moves_car_then_pays | leaves the pump, pays 12 s later (inside the grace period) | day | 0 | 0 | 0 | 0 | yes | n/a | n/a | 0 |
| late_payment | leaves, pays after the grace period: alert, then retraction | day | 1 | 1 | 1 | 1 | yes | BRE5521 -> deleted | BRE5521 -> deleted | 0 |
| drive_through | stops at a pump for 12 s, no fuel | day | 0 | 0 | 0 | 0 | yes | n/a | n/a | 0 |
| two_pumps | P1 pays, P2 drives off at the same time | day | 1 | 1 | 0 | 0 | yes | RUN2002 -> not read | RUN2002 -> not read | 0 |
| same_pump_two_cars | first car pays, the next car at the same pump drives off | day | 1 | 1 | 0 | 0 | yes | ZZZ9999 -> not read | ZZZ9999 -> ZZZ9999 | 0 |
| sale_reported_late | the controller reports the unpaid sale 10 s after the car left | day | 1 | 1 | 0 | 0 | yes | LTE4040 -> not read | LTE4040 -> LTE4040 | 0 |
| camera_blocked_then_pays | the car is hidden for 15 s after fuelling, pays later, then leaves | day | 0 | 0 | 0 | 0 | yes | n/a | n/a | 0 |
| camera_blocked_past_grace | hidden for longer than the grace period, then pays: alert, then retraction (the known limit: the rule can not tell a hidden car from a gone car) | day | 1 | 1 | 1 | 1 | yes | HID4040 -> deleted | HID4040 -> deleted | 0 |
| blocked_mid_fuelling_then_pays | hidden for 6 s while fuel is flowing, pays inside, then leaves | day | 0 | 0 | 0 | 0 | yes | n/a | n/a | 0 |
| blocked_mid_fuelling_drive_off | hidden for 10 s while fuel is flowing, then leaves without paying | day | 1 | 1 | 0 | 0 | yes | MDO6161 -> MDO6161 | MDO6161 -> MDO6161 | 1 |
| track_id_changes_then_pays | the tracker gives the car a new id while fuel is flowing; it pays, then leaves | day | 0 | 0 | 0 | 0 | yes | n/a | n/a | 0 |
| next_car_takes_the_spot | drive-off with an unreadable plate; 10 s later the next car takes the same spot, fuels, pays and leaves: one alert, and the second car's plate must not be stored | day | 1 | 1 | 0 | 0 | yes | n/a | n/a | 0 |
| no_vehicle_seen | unpaid sale, the camera never saw a vehicle | day | 1 | 1 | 0 | 0 | yes | n/a | n/a | 0 |
| night_drive_off | drive-off at night | night | 1 | 1 | 0 | 0 | yes | NGT6060 -> NGT6060 | NGT6060 -> NGT6060 | 1 |
| night_pays | pays inside at night | night | 0 | 0 | 0 | 0 | yes | n/a | n/a | 0 |

Timelines with the right alerts and retractions: 18 of 18.
Drive-off plates, open detector: 3 of 6 read exactly right, 3 not read (nothing stored), 0 stored wrong. Retracted alerts have their plate deleted and are not counted.
Drive-off plates, both detectors: 5 of 6 read exactly right, 1 not read (nothing stored), 0 stored wrong. Retracted alerts have their plate deleted and are not counted.
