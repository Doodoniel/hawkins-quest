"""Находит на фото стены реальные буквы и лампочки гирлянды."""
import json
from collections import deque

import numpy as np
from PIL import Image, ImageDraw

SRC = r"C:\projects\Hawkins\assets\byers-wall.jpg"
OUT = r"C:\Users\kazan\AppData\Local\Temp\claude\C--projects\956c214c-3624-4375-9705-fc73ba17dfd4\scratchpad"

im = Image.open(SRC).convert("RGB")
W, H = im.size
A = np.asarray(im).astype(np.int16)

ROWS = [
    {"chars": "АБВГДЕЁЖЗИЙ",  "letters": (420, 610), "bulbs": (300, 452)},
    {"chars": "КЛМНОПРСТУФХ", "letters": (680, 860), "bulbs": (565, 700)},
    {"chars": "ЦЧШЩЪЫЬЭЮЯ?",  "letters": (930, 1140), "bulbs": (820, 950)},
]
X0, X1 = 360, 1900


def erode(mask, n=1):
    m = mask.copy()
    for _ in range(n):
        q = np.ones_like(m)
        q[1:, :] &= m[:-1, :]; q[:-1, :] &= m[1:, :]
        q[:, 1:] &= m[:, :-1]; q[:, :-1] &= m[:, 1:]
        m = m & q
    return m


def components(mask, min_area):
    h, w = mask.shape
    seen = np.zeros_like(mask)
    out = []
    for y0 in range(h):
        row = np.nonzero(mask[y0] & ~seen[y0])[0]
        for x0 in row:
            if seen[y0, x0]:
                continue
            q = deque([(x0, y0)])
            seen[y0, x0] = True
            pts = []
            while q:
                x, y = q.popleft()
                pts.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((nx, ny))
            if len(pts) >= min_area:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                out.append({"x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys),
                            "area": len(pts)})
    return out


result = []
preview = im.copy()
dr = ImageDraw.Draw(preview)

for row in ROWS:
    ly0, ly1 = row["letters"]
    sub = A[ly0:ly1, X0:X1]
    v = sub.max(axis=2)
    mask = v < 105

    # границы строки букв меряем проекцией, затем делим на N слотов:
    # соседние буквы кое-где слипаются, а компонент связности их не разделит
    col = mask.sum(axis=0)
    thr = max(3, col.max() * 0.06)
    cols = np.nonzero(col > thr)[0]
    xs, xe = int(cols[0]), int(cols[-1])
    rowmask = mask[:, xs:xe + 1]
    rws = np.nonzero(rowmask.sum(axis=1) > thr)[0]
    ys, ye = int(rws[0]), int(rws[-1])

    n = len(row["chars"])
    slot = (xe - xs + 1) / n
    merged = []
    for i in range(n):
        a = xs + slot * i
        b = a + slot
        seg = mask[:, int(a):int(b)]
        cs = seg.sum(axis=0)
        on = np.nonzero(cs > 1)[0]
        if len(on):
            lx0, lx1 = int(a) + int(on[0]), int(a) + int(on[-1])
        else:
            lx0, lx1 = int(a), int(b)
        rs = np.nonzero(seg.sum(axis=1) > 1)[0]
        if len(rs):
            ly_0, ly_1 = int(rs[0]), int(rs[-1])
        else:
            ly_0, ly_1 = ys, ye
        merged.append({"x0": lx0, "x1": lx1, "y0": ly_0, "y1": ly_1, "area": int(seg.sum())})

    by0, by1 = row["bulbs"]
    bsub = A[by0:by1, X0:X1]
    mx = bsub.max(axis=2); mn = bsub.min(axis=2)
    bmask = ((mx - mn) > 75) & (mx > 135)
    bmask = erode(bmask, 2)
    bulbs = components(bmask, 90)
    bulbs.sort(key=lambda c: c["x0"])

    row["_letters"] = merged
    row["_bulbs"] = bulbs
    print(row["chars"], "букв найдено:", len(merged), "лампочек:", len(bulbs))

    for c in merged:
        dr.rectangle([c["x0"] + X0, c["y0"] + ly0, c["x1"] + X0, c["y1"] + ly0],
                     outline=(0, 255, 90), width=4)
    for c in bulbs:
        dr.rectangle([c["x0"] + X0, c["y0"] + by0, c["x1"] + X0, c["y1"] + by0],
                     outline=(255, 210, 0), width=4)

preview.save(OUT + r"\wall_detect.png")


def pct(v, total):
    return round(v / total * 100, 3)


data = []
for row in ROWS:
    ly0, _ = row["letters"]
    by0, _ = row["bulbs"]
    lets, bulbs = row["_letters"], row["_bulbs"]
    for i, ch in enumerate(row["chars"]):
        if i >= len(lets):
            break
        c = lets[i]
        lcx = (c["x0"] + c["x1"]) / 2 + X0
        lcy = (c["y0"] + c["y1"]) / 2 + ly0
        lw = c["x1"] - c["x0"]
        lh = c["y1"] - c["y0"]
        # ближайшая по x лампочка
        best, bd = None, 1e9
        for b in bulbs:
            bcx = (b["x0"] + b["x1"]) / 2 + X0
            d = abs(bcx - lcx)
            if d < bd:
                bd, best = d, b
        if best is None:
            continue
        bcx = (best["x0"] + best["x1"]) / 2 + X0
        bcy = (best["y0"] + best["y1"]) / 2 + by0
        bw = best["x1"] - best["x0"]
        bh = best["y1"] - best["y0"]
        patch = A[int(bcy) - 6:int(bcy) + 6, int(bcx) - 6:int(bcx) + 6].reshape(-1, 3)
        col = patch.mean(axis=0)
        m = max(col.max(), 1)
        col = (col / m * 255).clip(60, 255).astype(int)
        data.append({
            "c": ch,
            "lx": pct(lcx, W), "ly": pct(lcy, H),
            "lw": pct(lw, W), "lh": pct(lh, H),
            "bx": pct(bcx, W), "by": pct(bcy, H),
            "bw": pct(max(bw, bh), W),
            "col": "#%02x%02x%02x" % tuple(col),
        })

print("итого пар буква+лампочка:", len(data))
open(OUT + r"\wall_data.json", "w", encoding="utf-8").write(
    json.dumps(data, ensure_ascii=False))
print(json.dumps(data[:4], ensure_ascii=False))

# Как пользоваться:
#   python tools/detect-wall.py
# Скрипт находит на assets/byers-wall.jpg реальные буквы и лампочки гирлянды
# и пишет их координаты (в процентах кадра) в wall_data.json рядом с собой.
# Нужен, только если поменяется фотография стены: тогда данные из wall_data.json
# надо перенести в массив WALL в index.html.
