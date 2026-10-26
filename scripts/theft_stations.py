"""Write configs/stations/trumanrd/chNN.yaml from polygons drawn on 1280x720 stills (scaled x2 to the 2560x1440 stream).

DRAFT zones for the Truman Rd. store: drawn by eye from one still per channel, never validated on footage and with
no floor calibration. They are enough to give the theft gates (bree.theft.gate) something to work with; replace them
with scripts/draw_zones.py output once someone can check them. Run: python scripts/theft_stations.py
"""
from pathlib import Path

import yaml

OUT = Path(__file__).resolve().parents[1] / "configs" / "stations" / "trumanrd"
S = 2   # 1280x720 still -> 2560x1440 stream


def poly(pts):
    return [[int(x * S), int(y * S)] for x, y in pts]


CATALOG = {"UNKNOWN": {"category": "unknown", "price": 0.0}}
COCO_MAP = {"bottle": "soda_bottle", "cup": "soda_bottle", "cell phone": "phone_accessory"}

# name: (role, zones [(name, kind, pts)], staff [(name, pts)], ignore [(name, pts)], extra theft keys)
CH = {
    "ch07": ("door", [
        ("front_doors", "exit", [(225, 330), (520, 330), (520, 445), (225, 445)]),
        ("register", "register", [(650, 400), (1000, 400), (1000, 720), (640, 720)]),
        ("drink_fridge", "cooler", [(140, 150), (240, 150), (240, 380), (140, 380)]),
        ("snack_rack_left", "shelf", [(0, 330), (230, 330), (230, 720), (0, 720)]),
        ("perfume_case", "shelf", [(385, 395), (650, 395), (650, 720), (385, 720)]),
        ("counter_display", "shelf", [(700, 250), (900, 250), (960, 420), (760, 400)]),
    ], [("behind_counter", [(790, 100), (1280, 100), (1280, 720), (1000, 720), (930, 560), (850, 380)])], [], {}),
    "ch29": ("counter", [
        ("counter_surface", "register", [(50, 110), (960, 110), (960, 520), (50, 520)]),
        ("display_left", "shelf", [(0, 0), (500, 0), (500, 330), (0, 330)]),
    ], [("staff_side", [(0, 520), (1280, 520), (1280, 720), (0, 720)])], [], {}),
    "ch12": ("aisle", [
        ("chest_freezers", "cooler", [(310, 270), (480, 270), (470, 620), (310, 620)]),
        ("gondola_drinks", "shelf", [(640, 220), (1000, 350), (980, 620), (780, 600), (620, 300)]),
        ("wall_shelves_right", "shelf", [(850, 130), (1280, 330), (1280, 520), (1020, 400)]),
    ], [], [("window_glass", [(0, 0), (400, 0), (380, 260), (0, 520)])], {}),
    "ch14": ("aisle", [
        ("rack_left", "shelf", [(80, 260), (520, 260), (520, 720), (200, 720)]),
        ("rack_right", "shelf", [(920, 330), (1280, 330), (1280, 720), (910, 720)]),
        ("cooler_far_right", "cooler", [(1140, 60), (1280, 60), (1280, 340), (1230, 340)]),
    ], [], [], {}),
    "ch15": ("cooler", [
        ("cooler_doors", "cooler", [(20, 420), (130, 300), (430, 170), (700, 40), (715, 230), (600, 430), (470, 720), (340, 720)]),
        ("snack_stand", "shelf", [(710, 350), (1000, 350), (960, 670), (720, 650)]),
        ("shelves_right", "shelf", [(960, 200), (1280, 230), (1280, 620), (960, 610)]),
    ], [], [("lottery_led_display", [(1090, 80), (1170, 80), (1170, 170), (1090, 170)])], {}),
    "ch10": ("cooler", [
        ("cooler_back_wall", "cooler", [(665, 20), (1060, 60), (1050, 130), (670, 150)]),
        ("cooler_tall", "cooler", [(405, 10), (630, 10), (630, 290), (510, 290), (405, 250)]),
        ("snack_end", "shelf", [(830, 215), (1000, 290), (970, 520), (800, 500)]),
        ("chip_tower", "shelf", [(590, 380), (730, 380), (730, 700), (600, 690)]),
    ], [], [], {}),
    "ch20": ("outdoor", [
        ("pump_area", "shelf", [(500, 100), (900, 100), (1000, 300), (400, 300)]),
    ], [], [], {}),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ch, (role, zones, staff, ignore, extra) in CH.items():
        d = {
            "# draft": "zones drawn by eye from one still, NOT validated (scripts/theft_stations.py)",
            "store": {"name": "Truman Rd. store, " + ch},
            "camera": {"id": ch, "resolution": [2560, 1440], "fps": 30},
            "zones": [{"name": n, "kind": k, "polygon": poly(p)} for n, k, p in zones],
            "terminals": {"pos_1": next((n for n, k, _ in zones if k == "register"), zones[0][0])},
            "catalog": CATALOG, "product_classes": COCO_MAP,
            "rules": {"foot_point": "bottom"},
            "ledger": {"payment_mode": "dwell", "max_tier": "review"},
            "theft": {"role": role,
                      "staff_zones": [{"name": n, "polygon": poly(p)} for n, p in staff],
                      "ignore_zones": [{"name": n, "polygon": poly(p)} for n, p in ignore], **extra},
        }
        d.pop("# draft")
        (OUT / f"{ch}.yaml").write_text("# DRAFT zones drawn by eye from one still, not validated (scripts/theft_stations.py)\n"
                                        + yaml.safe_dump(d, sort_keys=False, default_flow_style=None, width=200))
        print("wrote", OUT / f"{ch}.yaml")


if __name__ == "__main__":
    main()
