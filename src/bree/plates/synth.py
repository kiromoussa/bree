"""SYNTHETIC US-style plates and forecourt frames for tests and the plate bench.

Nothing here is real footage. Plates are drawn with system fonts (DIN Condensed, Arial Narrow),
not the embossed dies real states use, so numbers measured on them say how the pipeline behaves
with size, light and noise, not how it will do at a real pump.
"""
from __future__ import annotations

import string
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LETTERS, DIGITS = string.ascii_uppercase, string.digits
# Serial patterns in use on US passenger plates: L letter, D digit, a space is the gap.
FORMATS = ("LLL DDDD", "DLLL DDD", "LLL DDD", "DDD LLL", "LL DDDDD", "LLLDDDD")
# (background, text) in RGB: white/blue, white/black, yellow/black, white/red, light blue/navy.
STYLES = (((245, 245, 245), (20, 40, 110)), ((240, 240, 235), (15, 15, 15)), ((240, 200, 40), (10, 10, 10)),
          ((245, 245, 245), (150, 20, 25)), ((200, 220, 240), (15, 30, 70)))
STATES = ("CALIFORNIA", "TEXAS", "FLORIDA", "NEW YORK", "OHIO", "GEORGIA", "ARIZONA", "MASSACHUSETTS")
_FONT_DIR = Path("/System/Library/Fonts/Supplemental")
_FONTS = ("DIN Condensed Bold.ttf", "Arial Narrow Bold.ttf", "DIN Alternate Bold.ttf")
_FALLBACK = ("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",)

# Light and sensor noise levels. gain scales brightness, sigma is Gaussian sensor noise in grey
# levels (0 to 255) added after the gain, blur is motion blur length as a fraction of plate width.
CONDITIONS = {
    "day": dict(gain=1.0, sigma=2.0, blur=0.0),
    "day_motion": dict(gain=1.0, sigma=2.0, blur=0.04),
    "dusk": dict(gain=0.45, sigma=6.0, blur=0.02),
    "night": dict(gain=0.18, sigma=10.0, blur=0.02),
    "night_noisy": dict(gain=0.10, sigma=16.0, blur=0.04),
}


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    for p in (_FONT_DIR / name, *map(Path, _FALLBACK)):
        if p.exists():
            return ImageFont.truetype(str(p), size)
    raise FileNotFoundError("no plate font found; set bree.plates.synth._FALLBACK to a bold TTF")


def random_text(rng: np.random.Generator) -> str:
    """A serial without the gap, e.g. 'ABC1234'."""
    fmt = FORMATS[rng.integers(len(FORMATS))]
    return "".join(rng.choice(list(LETTERS)) if c == "L" else rng.choice(list(DIGITS)) if c == "D" else " "
                   for c in fmt)


def render_plate(text: str, rng: np.random.Generator, width: int = 480) -> np.ndarray:
    """A flat, clean plate (BGR, 2:1 like the 12 x 6 inch US plate). `text` may hold one space."""
    w, h = width, width // 2
    bg, fg = STYLES[rng.integers(len(STYLES))]
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((3, 3, w - 4, h - 4), radius=h // 10, outline=fg, width=max(2, h // 50))
    small = _font("Arial Narrow Bold.ttf", h // 7)
    state = STATES[rng.integers(len(STATES))]
    d.text((w / 2, h * 0.12), state, font=small, fill=fg, anchor="mm")
    for x in (0.2, 0.8):                                    # bolt holes
        d.ellipse((w * x - h / 40, h * 0.11 - h / 40, w * x + h / 40, h * 0.11 + h / 40), fill=(90, 90, 90))
    name = _FONTS[rng.integers(len(_FONTS))]
    size = int(h * 0.62)
    font = _font(name, size)
    while d.textlength(text, font=font) > w * 0.88:         # shrink until the serial fits
        size -= 4
        font = _font(name, size)
    d.text((w / 2, h * 0.56), text, font=font, fill=fg, anchor="mm")
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def degrade(img: np.ndarray, rng: np.random.Generator, gain: float, sigma: float, blur: float,
            plate_w: float | None = None) -> np.ndarray:
    """Motion blur, lower light, sensor noise, then JPEG, in that order (what a camera does)."""
    out = img.astype(np.float32)
    k = int(round(blur * (plate_w or img.shape[1])))
    if k >= 2:
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1.0 / k                            # horizontal: the car rolls along the pump
        out = cv2.filter2D(out, -1, kern)
    out = out * gain + rng.normal(0, sigma, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _quad(rng: np.random.Generator, w: float, max_yaw: float = 25.0, max_pitch: float = 12.0) -> np.ndarray:
    """Corners (tl, tr, br, bl) of a plate about `w` px wide seen off axis, centred on (0, 0)."""
    yaw, pitch = np.radians(rng.uniform(-max_yaw, max_yaw)), np.radians(rng.uniform(-max_pitch, max_pitch))
    roll = np.radians(rng.uniform(-4, 4))
    pts = np.array([[-1, -0.5, 0], [1, -0.5, 0], [1, 0.5, 0], [-1, 0.5, 0]], np.float64)
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    p = pts @ (rz @ rx @ ry).T
    z = p[:, 2] + 8.0                                        # camera 8 plate half-widths away
    uv = p[:, :2] / z[:, None]
    uv *= w / (uv[:, 0].max() - uv[:, 0].min())              # so the box is exactly w px wide
    return (uv - (uv.max(0) + uv.min(0)) / 2).astype(np.float32)


@dataclass
class PlateSample:
    image: np.ndarray            # BGR
    text: str                    # truth, no space
    box: tuple[int, int, int, int]   # plate box x1, y1, x2, y2 in `image`


def plate_crop(rng: np.random.Generator, width_px: int, condition: str = "day", margin: float = 0.08,
               text: str | None = None) -> PlateSample:
    """What a plate detector hands the reader: a plate `width_px` wide, off axis, with a little
    car body around it, under the named light and noise condition."""
    spaced = text or random_text(rng)
    flat = render_plate(spaced, rng)
    quad = _quad(rng, width_px)
    m = max(2, int(margin * width_px))
    w, h = int(np.ceil(np.ptp(quad[:, 0]))) + 2 * m, int(np.ceil(np.ptp(quad[:, 1]))) + 2 * m
    dst = quad + np.array([w / 2, h / 2], np.float32)
    body = tuple(int(c) for c in rng.integers(30, 200, 3))
    fh, fw = flat.shape[:2]
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [fw, 0], [fw, fh], [0, fh]]), dst)
    img = cv2.warpPerspective(flat, H, (w, h), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=body)
    img = degrade(img, rng, plate_w=width_px, **CONDITIONS[condition])
    x1, y1, x2, y2 = dst[:, 0].min(), dst[:, 1].min(), dst[:, 0].max(), dst[:, 1].max()
    return PlateSample(img, spaced.replace(" ", ""), (int(x1), int(y1), int(x2), int(y2)))


def car_box(plate_w: float) -> tuple[float, float]:
    """Width and height in pixels of the car rear drawn around a plate `plate_w` px wide
    (a plate is 0.30 m wide, a car about 1.8 m)."""
    return plate_w * 6.0, plate_w * 4.2


def draw_car(frame: np.ndarray, rng: np.random.Generator, cx: float, cy: float, plate_w: int,
             text: str, colour: tuple[int, int, int] | None = None) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Draw a crude car rear (body, window, lights, bumper, plate) centred on (cx, cy), clean.
    Returns (car box, plate box). Call `degrade` on the whole frame afterwards."""
    cw, ch = car_box(plate_w)
    x1, y1 = int(cx - cw / 2), int(cy - ch / 2)
    x2, y2 = int(x1 + cw), int(y1 + ch)
    body = colour or tuple(int(c) for c in rng.integers(30, 200, 3))
    cv2.rectangle(frame, (x1, y1), (x2, y2), body, -1)
    cv2.rectangle(frame, (int(x1 + cw * 0.12), int(y1 + ch * 0.06)), (int(x2 - cw * 0.12), int(y1 + ch * 0.38)),
                  (60, 50, 45), -1)                                             # rear window
    for sx in (0.02, 0.84):                                                      # tail lights
        cv2.rectangle(frame, (int(x1 + cw * sx), int(y1 + ch * 0.45)), (int(x1 + cw * (sx + 0.14)), int(y1 + ch * 0.58)),
                      (30, 30, 190), -1)
    cv2.rectangle(frame, (x1, int(y1 + ch * 0.82)), (x2, y2), (45, 45, 45), -1)   # bumper
    flat = render_plate(text, rng)
    quad = _quad(rng, plate_w, max_yaw=15, max_pitch=8) + np.array([cx, y1 + ch * 0.68], np.float32)
    fh, fw = flat.shape[:2]
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [fw, 0], [fw, fh], [0, fh]]), quad)
    size = (frame.shape[1], frame.shape[0])
    warped = cv2.warpPerspective(flat, H, size, flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full((fh, fw), 255, np.uint8), H, size)
    frame[mask > 127] = warped[mask > 127]
    px1, py1, px2, py2 = quad[:, 0].min(), quad[:, 1].min(), quad[:, 0].max(), quad[:, 1].max()
    return (x1, y1, x2, y2), (int(px1), int(py1), int(px2), int(py2))


def forecourt(rng: np.random.Generator, size: tuple[int, int] = (1280, 720)) -> np.ndarray:
    """An empty forecourt background: asphalt with texture, lane paint and a pump island."""
    w, h = size
    base = rng.normal(95, 9, (h // 4, w // 4, 1)).repeat(3, axis=2)
    frame = cv2.resize(np.clip(base, 0, 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    for x in (0.25, 0.75):
        cv2.line(frame, (int(w * x), 0), (int(w * x), h), (200, 200, 200), 4)
    cv2.rectangle(frame, (int(w * 0.46), int(h * 0.05)), (int(w * 0.54), int(h * 0.30)), (40, 40, 160), -1)   # pump
    cv2.putText(frame, "PUMP 3  REGULAR 3.49", (int(w * 0.36), int(h * 0.04)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (230, 230, 230), 2)                                              # distractor text
    return frame


def scene(rng: np.random.Generator, plate_w: int, condition: str = "day",
          size: tuple[int, int] = (1280, 720), text: str | None = None) -> tuple[PlateSample, tuple[int, int, int, int]]:
    """One forecourt frame with one car. Returns the sample (plate box in frame pixels) and the car box."""
    frame = forecourt(rng, size)
    cw, ch = car_box(plate_w)
    cx = rng.uniform(cw / 2 + 5, max(cw / 2 + 6, size[0] - cw / 2 - 5))
    cy = rng.uniform(ch / 2 + 5, max(ch / 2 + 6, size[1] - ch / 2 - 5))
    spaced = text or random_text(rng)
    car, plate = draw_car(frame, rng, cx, cy, plate_w, spaced)
    frame = degrade(frame, rng, plate_w=plate_w, **CONDITIONS[condition])
    return PlateSample(frame, spaced.replace(" ", ""), plate), car
