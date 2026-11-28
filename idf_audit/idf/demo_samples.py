"""Sample photos for the live demo page (DEMO_UI): drawn pictures with no GPS
or time in the file, so the scenario chosen on the page - where the app says
the phone was, gallery or camera, fake GPS - is what decides the location check.
All invented. `car_park` is the wrong picture for every requirement, to show a
rejection on content."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf"):
        for folder in ("C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu"):
            try:
                return ImageFont.truetype(f"{folder}/{name}", size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def badminton_court() -> Image.Image:
    img = Image.new("RGB", (1200, 800), (30, 110, 60))
    d = ImageDraw.Draw(img)
    d.rectangle((150, 100, 1050, 700), outline="white", width=8)
    d.line((600, 100, 600, 700), fill="white", width=6)
    d.line((150, 400, 1050, 400), fill="white", width=4)
    d.rectangle((560, 60, 640, 110), fill=(60, 60, 60))
    d.text((360, 20), "BADMINTON COURT 1", fill="white", font=_font(44))
    return img


def tennis_court() -> Image.Image:
    img = Image.new("RGB", (1200, 800), (150, 190, 230))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 260, 1200, 800), fill=(60, 140, 80))
    d.polygon([(380, 300), (820, 300), (1100, 760), (100, 760)], fill=(40, 90, 170), outline="white", width=6)
    d.line((240, 530, 960, 530), fill="white", width=10)
    for x in range(240, 961, 40):
        d.line((x, 500, x, 530), fill=(230, 230, 230), width=2)
    d.line((600, 300, 600, 760), fill="white", width=4)
    d.text((390, 110), "TENNIS COURT", fill="white", font=_font(60))
    return img


def running_track() -> Image.Image:
    img = Image.new("RGB", (1400, 900), (60, 140, 70))
    d = ImageDraw.Draw(img)
    d.ellipse((100, 150, 1300, 850), fill=(180, 60, 50))
    d.ellipse((300, 280, 1100, 720), fill=(70, 160, 80))
    for r in range(1, 4):
        d.ellipse((100 + r * 50, 150 + r * 32, 1300 - r * 50, 850 - r * 32), outline="white", width=3)
    d.text((480, 40), "RUNNING TRACK", fill="white", font=_font(56))
    return img


def activity_poster() -> Image.Image:
    img = Image.new("RGB", (900, 1200), (235, 225, 210))
    d = ImageDraw.Draw(img)
    d.rectangle((100, 150, 800, 1050), fill=(255, 140, 0), outline="black", width=8)
    for i, line in enumerate(("WALK 10,000", "STEPS A DAY", "", "TAKE THE STAIRS")):
        d.text((140, 300 + i * 100), line, fill="white" if i < 2 else "black", font=_font(64))
    return img


def wellness_poster() -> Image.Image:
    img = Image.new("RGB", (900, 1200), (235, 225, 210))
    d = ImageDraw.Draw(img)
    d.rectangle((60, 520, 840, 1150), fill=(20, 120, 200), outline="black", width=8)
    for i, line in enumerate(("MOVE MORE", "SIT LESS", "STRETCH HOURLY")):
        d.text((100, 580 + i * 95), line, fill="white", font=_font(60))
    return img


def car_park() -> Image.Image:
    img = Image.new("RGB", (1200, 800), (110, 110, 115))
    d = ImageDraw.Draw(img)
    for x in range(100, 1100, 200):
        d.line((x, 300, x, 780), fill="white", width=6)
        d.rounded_rectangle((x + 30, 420, x + 170, 700), radius=30, fill=(180, 40, 40) if x % 400 else (40, 70, 160))
    d.rectangle((380, 60, 820, 200), fill=(20, 60, 160))
    d.text((440, 95), "PARKING", fill="white", font=_font(72))
    return img


SAMPLES = {"badminton-court": badminton_court, "tennis-court": tennis_court, "running-track": running_track,
           "activity-poster": activity_poster, "wellness-poster": wellness_poster, "car-park": car_park}


def jpeg(name: str) -> bytes | None:
    """The sample as JPEG bytes, without EXIF; None for an unknown name."""
    draw = SAMPLES.get(name)
    if draw is None:
        return None
    buf = io.BytesIO()
    draw().save(buf, "JPEG", quality=90)
    return buf.getvalue()
