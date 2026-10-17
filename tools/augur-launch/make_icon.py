"""Render the lituus brand mark to augur.ico (16-256 px). Developer tool; the .ico is committed."""
from pathlib import Path
from PIL import Image, ImageDraw

BG, BRASS = (20, 24, 31, 255), (212, 176, 98, 255)
# SVG path M25 30V12A9 9 0 0 0 7 12A6 6 0 0 0 19 12A3 3 0 0 0 13 12 on a 32-unit grid:
# a vertical stem, then three half-circle arcs spiralling inward.
SEGMENTS = [('line', (25, 30), (25, 12)), ('arc', (16, 12), 9, 0, 180),
            ('arc', (13, 12), 6, 180, 360), ('arc', (16, 12), 3, 0, 180)]


def render(size):
    s = 8  # supersample, then downscale for smooth edges
    n = size * s
    img = Image.new('RGBA', (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, n - 1, n - 1), radius=n * 0.22, fill=BG)
    k = n / 32
    w = max(1, round(2.6 * k))
    for seg in SEGMENTS:
        if seg[0] == 'line':
            (x0, y0), (x1, y1) = seg[1], seg[2]
            d.line((x0 * k, y0 * k, x1 * k, y1 * k), fill=BRASS, width=w)
            for x, y in ((x0, y0), (x1, y1)):
                d.ellipse((x * k - w / 2, y * k - w / 2, x * k + w / 2, y * k + w / 2), fill=BRASS)
        else:
            (cx, cy), r, a0, a1 = seg[1], seg[2], seg[3], seg[4]
            # Arcs sweep counter-clockwise from the right through the top in SVG terms (y down):
            # PIL angles are clockwise from +x, so the upper half is 180..360.
            box = ((cx - r) * k, (cy - r) * k, (cx + r) * k, (cy + r) * k)
            start, end = (180, 360) if a0 == 0 else (0, 180)
            d.arc(box, start, end, fill=BRASS, width=w)
    return img.resize((size, size), Image.LANCZOS)


def main():
    out = Path(__file__).with_name('augur.ico')
    sizes = [16, 24, 32, 48, 64, 128, 256]
    images = [render(s) for s in sizes]
    images[-1].save(out, format='ICO', sizes=[(s, s) for s in sizes], append_images=images[:-1])
    images[-1].save(out.with_name('augur-256.png'))
    print(out, out.stat().st_size)


if __name__ == '__main__':
    main()
