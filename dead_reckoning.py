"""
DEAD RECKONING
CVPR AI Art Gallery 2026 — Proof of Concept

Interactive installation: participant types a sentence about something lost.
System generates a 4-frame strip navigating from photographic present
through memory, painterly abstraction, to pure colour field.

Each frame represents a different distance from the original moment:
  I   PRESENT        — photographic, evidence, grain
  II  FIVE YEARS AGO — the Kodachrome fade, the drawer you don't open
  III A DECADE BACK  — flattened into Hopper's hard geometry
  IV  AT ONE LIMIT   — two colour fields. warmth and dark.

The model has never lost anything.
It cannot navigate toward absence.
Everything you see is inference.

Usage:
    python dead_reckoning_FINAL.py
    # Renders all 16 scenes + main submission to /outputs/
    # Each scene takes ~6s. Full batch ~2min.

Requirements:
    pip install numpy pillow scipy
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
import math, os, sys

# ── Canvas dimensions ──────────────────────────────────────────────────────────
W, H = 1200, 720
RNG  = np.random.default_rng(2024)
OUT  = "/mnt/user-data/outputs"

# ── Utilities ──────────────────────────────────────────────────────────────────

def fbm(shape, scale=100, octaves=5, seed=None):
    rng = np.random.default_rng(seed) if seed is not None else RNG
    h, w = shape
    out = np.zeros((h, w), np.float32)
    amp, freq, total = 1.0, 1.0, 0.0
    for _ in range(octaves):
        sh = max(2, h // max(1, int(scale / freq)))
        sw = max(2, w // max(1, int(scale / freq)))
        n = np.array(Image.fromarray(
            rng.random((sh, sw)).astype(np.float32)).resize((w, h), Image.BICUBIC))
        out += n * amp; total += amp; amp *= 0.5; freq *= 2.0
    return out / total

def to_img(arr):
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

def to_arr(img):
    return np.array(img.convert("RGB")).astype(np.float32) / 255.0

def _vignette(img, strength=0.38):
    h, w = img.shape[:2]
    y_, x_ = np.ogrid[:h, :w]
    dist = np.sqrt(((x_-w/2)/(w/2))**2 + ((y_-h/2)/(h/2))**2)
    mask = np.clip(1 - dist * strength * 0.9, 0, 1)
    mask = gaussian_filter(mask, sigma=int(h*0.08))
    return img * mask[:,:,np.newaxis]

def _contrast(img, factor):
    return np.clip((img - 0.5)*factor + 0.5, 0, 1)

# ── Four frame transforms ──────────────────────────────────────────────────────

def f1_present(base):
    """I  PRESENT — Film photograph. Kodak Portra 400."""
    out = base.copy()
    lum = 0.299*out[:,:,0] + 0.587*out[:,:,1] + 0.114*out[:,:,2]
    grain_mask = (0.75 - lum) * 0.042 + 0.010
    grain = RNG.standard_normal((H, W, 3)).astype(np.float32)
    grain[:,:,1] *= 0.72
    out += grain * np.clip(grain_mask, 0.008, 0.050)[:,:,np.newaxis]
    bright = np.clip(out[:,:,0] - 0.72, 0, 1) ** 1.5
    for sig, s in [(8, 0.22), (18, 0.11), (40, 0.05)]:
        b = gaussian_filter(bright, sigma=sig)
        out[:,:,0] = np.clip(out[:,:,0] + b*s*1.15, 0, 1)
        out[:,:,1] = np.clip(out[:,:,1] + b*s*0.60, 0, 1)
        out[:,:,2] = np.clip(out[:,:,2] + b*s*0.12, 0, 1)
    out[:,:,0] = np.roll(out[:,:,0], -2, axis=1)
    out[:,:,2] = np.roll(out[:,:,2],  2, axis=1)
    for c in range(3):
        out[:,:,c] = np.clip(out[:,:,c] * 1.04 + 0.015 * (1-out[:,:,c]), 0, 1)
    out = _vignette(out, 0.32)
    return np.clip(out, 0, 1)

def f2_memory(base):
    """II  FIVE YEARS AGO — Kodachrome fade."""
    out = base.copy()
    out = out * 0.76 + 0.052
    out[:,:,2] = out[:,:,2] * 0.70 + 0.038
    out[:,:,1] = out[:,:,1] * 0.86 + 0.012
    out[:,:,0] = np.clip(out[:,:,0] * 1.05 + 0.018, 0, 1)
    cy_, cx_ = H/2, W/2
    y_, x_ = np.ogrid[:H, :W]
    dist = np.sqrt(((x_-cx_)/cx_)**2 + ((y_-cy_)/cy_)**2)
    blend = np.clip(dist * 0.60, 0, 1)[:,:,np.newaxis]
    blurred = np.stack([gaussian_filter(out[:,:,c], sigma=3) for c in range(3)], axis=2)
    out = out * (1-blend) + blurred * blend
    for y in range(0, H, 4):
        out[y:y+1, :] = np.clip(out[y:y+1, :] - 0.005, 0, 1)
    tex = fbm((H, W), scale=10, octaves=2, seed=99) * 0.022 - 0.011
    out += tex[:,:,np.newaxis]
    out = _vignette(out, 0.48)
    return np.clip(out, 0, 1)

def f3_hopper(base):
    """III  A DECADE BACK — Edward Hopper. Hard light. Absolute stillness."""
    from scipy.ndimage import sobel as _sobel
    out = base.copy()
    gray = 0.299*out[:,:,0] + 0.587*out[:,:,1] + 0.114*out[:,:,2]
    wall_mask = gray > 0.62
    out[wall_mask, 0] = np.clip(gray[wall_mask] * 0.90, 0, 1)
    out[wall_mask, 1] = np.clip(gray[wall_mask] * 0.88, 0, 1)
    out[wall_mask, 2] = np.clip(gray[wall_mask] * 0.74, 0, 1)
    table_mask = (gray > 0.35) & (gray < 0.70) & (out[:,:,0] > out[:,:,2] * 1.3)
    out[table_mask, 0] = np.clip(gray[table_mask] * 1.08, 0, 1)
    out[table_mask, 1] = np.clip(gray[table_mask] * 0.78, 0, 1)
    out[table_mask, 2] = np.clip(gray[table_mask] * 0.42, 0, 1)
    for c in range(3):
        out[:,:,c] = gaussian_filter(out[:,:,c], sigma=3)
    for y in range(int(H*0.08), int(H*0.38)):
        t = abs(y - H*0.22) / (H*0.14)
        bar = max(0, 1 - t*1.2) * 0.18
        out[y, int(W*0.55):, 0] = np.clip(out[y, int(W*0.55):, 0] + bar, 0, 1)
        out[y, int(W*0.55):, 1] = np.clip(out[y, int(W*0.55):, 1] + bar*0.85, 0, 1)
        out[y, int(W*0.55):, 2] = np.clip(out[y, int(W*0.55):, 2] + bar*0.52, 0, 1)
    table_top = int(H * 0.56)
    for y in range(table_top, H):
        t = (y - table_top) / (H - table_top)
        shadow_x = int(W * (0.72 - t * 0.35))
        if shadow_x > 0:
            out[y, :shadow_x, :] = np.clip(out[y, :shadow_x, :] * 0.78, 0, 1)
    out[:,:,2] = np.clip(out[:,:,2] * 1.06, 0, 1)
    out = _contrast(out, 1.22)
    return np.clip(out, 0, 1)

def f4_rothko(base):
    """IV  AT ONE LIMIT — Rothko. Two colour fields from the scene's palette."""
    out = np.zeros((H, W, 3), np.float32)
    lower_half = base[H//2:, :]
    upper_half = base[:H//2, :]
    warm = np.percentile(lower_half.reshape(-1, 3), 65, axis=0).astype(np.float32)
    cool = np.percentile(upper_half.reshape(-1, 3), 35, axis=0).astype(np.float32)
    warm = np.clip(warm * np.array([1.05, 0.90, 0.70]), 0.12, 0.92)
    cool = np.clip(cool * np.array([0.55, 0.45, 0.40]), 0.05, 0.45)
    split = int(H * 0.48); trans = int(H * 0.24)
    for y in range(H):
        d = y - split
        t = np.clip(0.5 + 0.5 * math.tanh(d / (trans / 3.5)), 0, 1)
        out[y] = cool*(1-t) + warm*t
    glow_cx = W * 0.42; glow_cy = H * 0.52
    for sig, s in [(int(H*0.08), 0.18), (int(H*0.16), 0.09), (int(H*0.32), 0.04)]:
        ctr = np.zeros((H, W), np.float32)
        ctr[int(glow_cy), int(glow_cx)] = 1.0
        b = gaussian_filter(ctr, sigma=sig)
        out[:,:,0] = np.clip(out[:,:,0] + b*s*1.2, 0, 1)
        out[:,:,1] = np.clip(out[:,:,1] + b*s*0.72, 0, 1)
        out[:,:,2] = np.clip(out[:,:,2] + b*s*0.18, 0, 1)
    for layer in range(6):
        noise = fbm((H, W), scale=RNG.integers(80, 320), octaves=3, seed=100+layer)
        shift = (RNG.random(3) - 0.5).astype(np.float32) * 0.020
        for c in range(3):
            out[:,:,c] += noise * shift[c]
    for c in range(3):
        out[:,:,c] = gaussian_filter(out[:,:,c], sigma=10)
    canvas = fbm((H, W), scale=14, octaves=4, seed=6) * 0.012 - 0.006
    out += canvas[:,:,np.newaxis]
    return np.clip(out, 0, 1)

# ── Compositor ─────────────────────────────────────────────────────────────────

FRAME_LABELS = [
    ("I",   "PRESENT"),
    ("II",  "FIVE YEARS AGO"),
    ("III", "A DECADE BACK"),
    ("IV",  "AT ONE LIMIT"),
]

def compose_2x2(frames, scene_title, participant_text, outpath):
    """2×2 grid. Black bar label inside each frame. Clean gallery presentation."""
    CANVAS_W = 3200; MARGIN = 110; COL_GAP = 70; ROW_GAP = 60
    HDR_H = 220; FTR_H = 150
    COL_W   = (CANVAS_W - MARGIN*2 - COL_GAP) // 2
    FRAME_H = int(COL_W * 0.62)
    CANVAS_H = HDR_H + 2*(FRAME_H + ROW_GAP) - ROW_GAP + FTR_H + 60

    # Warm linen background
    bg = np.zeros((CANVAS_H, CANVAS_W, 3), np.float32)
    paper = fbm((CANVAS_H, CANVAS_W), scale=200, octaves=4)
    for y in range(CANVAS_H):
        t = y / CANVAS_H
        bg[y,:,0] = 0.983 - t*0.010 + paper[y,:]*0.006
        bg[y,:,1] = 0.974 - t*0.016 + paper[y,:]*0.005
        bg[y,:,2] = 0.956 - t*0.030 + paper[y,:]*0.004

    canvas = Image.fromarray((np.clip(bg, 0, 1)*255).astype(np.uint8))
    draw   = ImageDraw.Draw(canvas)

    def F(name, size):
        for p in [f"/usr/share/fonts/truetype/dejavu/{name}.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
            if os.path.exists(p):
                try: return ImageFont.truetype(p, size)
                except: pass
        return ImageFont.load_default()

    f_title = F("DejaVuSans-Bold",    56)
    f_quote = F("DejaVuSerif-Italic", 28)
    f_scene = F("DejaVuSans",         22)
    f_bar   = F("DejaVuSans-Bold",    30)
    f_foot  = F("DejaVuSerif-Italic", 20)

    INK=(16,12,8); BROWN=(100,82,62); LGREY=(148,132,110); RULE=(175,158,138)

    # Header
    draw.text((MARGIN, 40),  "DEAD RECKONING", fill=INK,   font=f_title)
    draw.text((MARGIN, 110), f'"{participant_text}"', fill=BROWN, font=f_quote)
    draw.text((CANVAS_W-MARGIN, 44), scene_title.upper(),
              fill=(160,140,110), font=f_scene, anchor="rt")
    draw.line([(MARGIN, HDR_H-18), (CANVAS_W-MARGIN, HDR_H-18)], fill=RULE, width=2)
    draw.line([(MARGIN, HDR_H-12), (CANVAS_W-MARGIN, HDR_H-12)], fill=(200,185,165), width=1)

    # Grid: top-left, top-right, bottom-left, bottom-right
    cols  = [MARGIN, MARGIN + COL_W + COL_GAP]
    rows  = [HDR_H,  HDR_H + FRAME_H + ROW_GAP]
    order = [(0,0), (1,0), (0,1), (1,1)]

    for i, (frame_arr, (roman, label)) in enumerate(zip(frames, FRAME_LABELS)):
        fx = cols[order[i][0]]
        fy = rows[order[i][1]]

        fimg = to_img(frame_arr).resize((COL_W, FRAME_H), Image.LANCZOS)
        canvas.paste(fimg, (fx, fy))

        # Black bar — blended into frame bottom-left
        BAR_H = 52; BAR_W = int(COL_W * 0.64)
        bar_y = fy + FRAME_H - BAR_H
        region = np.array(canvas.crop((fx, bar_y, fx+BAR_W, bar_y+BAR_H))).astype(np.float32)
        blended = (region*0.12 + np.array([8,5,3], np.float32)*0.88).astype(np.uint8)
        canvas.paste(Image.fromarray(blended), (fx, bar_y))
        draw = ImageDraw.Draw(canvas)
        draw.text((fx+22, bar_y+11), f"{roman}  {label}", fill=(242,230,210), font=f_bar)
        draw.rectangle([fx-1, fy-1, fx+COL_W, fy+FRAME_H], outline=(140,122,98), width=1)

    # Footer
    foot_y = HDR_H + 2*(FRAME_H+ROW_GAP) - ROW_GAP + 30
    draw.line([(MARGIN, foot_y),   (CANVAS_W-MARGIN, foot_y)],   fill=RULE, width=2)
    draw.line([(MARGIN, foot_y+6), (CANVAS_W-MARGIN, foot_y+6)], fill=(200,185,165), width=1)
    draw.text((MARGIN, foot_y+24),
        "The model has never lost anything.   "
        "It cannot navigate toward absence.   "
        "Everything you see is inference.",
        fill=LGREY, font=f_foot)
    draw.text((MARGIN, foot_y+56),
        "CVPR AI Art Gallery 2026   ·   Interactive Installation & Archival Print",
        fill=(165,148,122), font=f_foot)

    # Soften & save
    arr = np.array(canvas).astype(np.float32)/255.0
    for c in range(3):
        arr[:,:,c] = gaussian_filter(arr[:,:,c], sigma=0.3)
    final = Image.fromarray((np.clip(arr,0,1)*255).astype(np.uint8))
    final.save(outpath, quality=97, dpi=(300,300))
    final.resize((final.width//2, final.height//2), Image.LANCZOS)\
         .save(outpath.replace('.jpg','_prev.jpg'), quality=92)
    print(f"  ✓  {scene_title}")

def make_frames(base):
    return [f1_present(base), f2_memory(base), f3_hopper(base), f4_rothko(base)]

# ── 16 Scene builders ──────────────────────────────────────────────────────────

def scene_empty_swing():
    img = np.zeros((H, W, 3), np.float32)
    for y in range(int(H*0.68)):
        t = y/(H*0.68)
        img[y,:,0]=0.72-t*0.15; img[y,:,1]=0.76-t*0.14; img[y,:,2]=0.82-t*0.10
    gt = int(H*0.68)
    for y in range(gt, H):
        t=(y-gt)/(H-gt)
        img[y,:,0]=0.68+t*0.05; img[y,:,1]=0.58+t*0.02; img[y,:,2]=0.42-t*0.05
    n = fbm((H,W), scale=120, octaves=4, seed=10)
    img[:int(H*0.68),:] += n[:int(H*0.68),:,np.newaxis]*0.04-0.02
    pil = to_img(img); d = ImageDraw.Draw(pil)
    post = (55,42,30)
    for x_top, x_bot in [(int(W*0.30),int(W*0.26)),(int(W*0.50),int(W*0.54))]:
        d.line([(x_top,int(H*0.08)),(x_bot,gt+20)], fill=post, width=8)
    d.line([(int(W*0.28),int(H*0.08)),(int(W*0.52),int(H*0.08))], fill=post, width=8)
    sx1=int(W*0.36)
    d.line([(sx1,int(H*0.09)),(sx1,int(H*0.55))], fill=(62,48,32), width=4)
    d.line([(sx1+8,int(H*0.09)),(sx1+8,int(H*0.55))], fill=(62,48,32), width=4)
    d.rectangle([sx1-14,int(H*0.53),sx1+22,int(H*0.60)], fill=(72,52,30))
    sx2=int(W*0.44)
    d.line([(sx2,int(H*0.09)),(sx2-28,int(H*0.52))], fill=(62,48,32), width=3)
    d.line([(sx2+6,int(H*0.09)),(sx2-22,int(H*0.52))], fill=(62,48,32), width=3)
    d.rectangle([sx2-42,int(H*0.50),sx2-8,int(H*0.57)], fill=(72,52,30))
    img = to_arr(pil)
    img += (fbm((H,W),scale=8,octaves=3,seed=11)*0.015-0.007)[:,:,np.newaxis]
    return np.clip(img,0,1)

def scene_phone_booth():
    img = np.zeros((H,W,3),np.float32)
    for y in range(H):
        t=y/H; img[y,:,0]=0.10+t*0.06; img[y,:,1]=0.12+t*0.04; img[y,:,2]=0.18+t*0.02
    img += (fbm((H,W),scale=80,octaves=4,seed=20)*0.03-0.015)[:,:,np.newaxis]
    pil=to_img(img); d=ImageDraw.Draw(pil)
    bx,bw,by_,bh = int(W*0.38),int(W*0.24),int(H*0.08),int(H*0.88)
    d.rectangle([bx,by_,bx+bw,by_+bh], fill=(140,28,22))
    # glow
    gx,gy_=int(bx+bw*0.1),int(by_+bh*0.15)
    gw2,gh2=int(bw*0.8),int(bh*0.6)
    d.rectangle([gx,gy_,gx+gw2,gy_+gh2], fill=(220,180,80))
    img=to_arr(pil)
    glow=np.zeros((H,W,3),np.float32)
    glow[gy_:gy_+gh2,gx:gx+gw2] = np.array([0.86,0.70,0.32])
    for sig in [20,50,100]:
        for c in range(3): glow[:,:,c]=gaussian_filter(glow[:,:,c],sigma=sig)*0.4
    img=np.clip(img+glow*0.5,0,1)
    # rain
    rng2=np.random.default_rng(21)
    rain=np.zeros((H,W,3),np.float32)
    for _ in range(800):
        rx=rng2.integers(0,W); ry=rng2.integers(0,H-30)
        l=rng2.integers(8,28); a=rng2.random()*0.18+0.04
        rain[ry:ry+l,rx:rx+1]=a
    img=np.clip(img+rain,0,1)
    img=_vignette(img,0.55)
    return np.clip(img,0,1)

def scene_hospital_window():
    img = np.zeros((H,W,3),np.float32)
    for y in range(H):
        t=y/H
        img[y,:,0]=0.88-t*0.06; img[y,:,1]=0.88-t*0.05; img[y,:,2]=0.88-t*0.04
    img+=(fbm((H,W),scale=60,octaves=3,seed=30)*0.015-0.007)[:,:,np.newaxis]
    pil=to_img(img); d=ImageDraw.Draw(pil)
    wx,wy_,ww,wh=int(W*0.28),int(H*0.08),int(W*0.44),int(H*0.72)
    # sky outside
    for y in range(wy_,wy_+wh):
        t=(y-wy_)/wh
        r=int((0.72-t*0.20)*255); g=int((0.80-t*0.15)*255); b=int((0.92-t*0.08)*255)
        d.rectangle([wx,y,wx+ww,y+1], fill=(r,g,b))
    # frame
    d.rectangle([wx,wy_,wx+ww,wy_+wh], outline=(180,170,158), width=12)
    d.line([(wx+ww//2,wy_),(wx+ww//2,wy_+wh)], fill=(180,170,158), width=8)
    d.line([(wx,wy_+wh//2),(wx+ww,wy_+wh//2)], fill=(180,170,158), width=8)
    # bed rail
    d.rectangle([int(W*0.05),int(H*0.72),int(W*0.95),int(H*0.78)],fill=(190,182,170))
    img=to_arr(pil)
    img=_vignette(img,0.28)
    return np.clip(img,0,1)

def scene_lighthouse():
    img = np.zeros((H,W,3),np.float32)
    for y in range(int(H*0.65)):
        t=y/(H*0.65)
        img[y,:,0]=0.08+t*0.06; img[y,:,1]=0.10+t*0.08; img[y,:,2]=0.22+t*0.14
    gt=int(H*0.65)
    for y in range(gt,H):
        t=(y-gt)/(H-gt)
        img[y,:,0]=0.12-t*0.04; img[y,:,1]=0.14-t*0.04; img[y,:,2]=0.16-t*0.04
    img+=(fbm((H,W),scale=200,octaves=3,seed=40)*0.04-0.02)[:,:,np.newaxis]
    # beam
    bx,by_=int(W*0.42),int(H*0.18)
    beam=np.zeros((H,W,3),np.float32)
    for y in range(H):
        for x in range(W):
            dx=x-bx; dy=y-by_
            if dy>0: continue
            angle=math.atan2(abs(dy),dx) if dx!=0 else math.pi/2
            if abs(angle-0.3)<0.18:
                dist=math.sqrt(dx**2+dy**2)
                beam[y,x]=np.array([0.96,0.92,0.72])*max(0,1-dist/600)*0.7
    for c in range(3): beam[:,:,c]=gaussian_filter(beam[:,:,c],sigma=8)
    img=np.clip(img+beam,0,1)
    pil=to_img(img); d=ImageDraw.Draw(pil)
    lx=int(W*0.42); lw=int(W*0.06)
    d.rectangle([lx,int(H*0.18),lx+lw,int(H*0.72)],fill=(220,215,205))
    d.rectangle([lx-int(W*0.01),int(H*0.18),lx+lw+int(W*0.01),int(H*0.22)],fill=(180,170,150))
    d.ellipse([lx+2,int(H*0.17),lx+lw-2,int(H*0.20)],fill=(255,240,180))
    img=to_arr(pil); img=_vignette(img,0.45)
    return np.clip(img,0,1)

def scene_train_window():
    img = np.zeros((H,W,3),np.float32)
    for y in range(H):
        t=y/H; img[y,:,0]=0.06+t*0.04; img[y,:,1]=0.08+t*0.04; img[y,:,2]=0.14+t*0.06
    img+=(fbm((H,W),scale=80,octaves=4,seed=50)*0.025-0.012)[:,:,np.newaxis]
    pil=to_img(img); d=ImageDraw.Draw(pil)
    # window frame
    wx,wy_,ww,wh=int(W*0.15),int(H*0.08),int(W*0.70),int(H*0.78)
    # blurred landscape outside
    for y in range(wy_,wy_+wh):
        t=(y-wy_)/wh
        r=int((0.18+t*0.12)*255); g=int((0.20+t*0.10)*255); b=int((0.28+t*0.06)*255)
        d.rectangle([wx+8,y,wx+ww-8,y+1],fill=(r,g,b))
    # streaking lights
    rng2=np.random.default_rng(51)
    for _ in range(120):
        lx_=rng2.integers(wx,wx+ww); ly_=rng2.integers(wy_,wy_+wh)
        ll=rng2.integers(30,120); bright=rng2.random()*0.7+0.3
        col=(int(bright*255),int(bright*240),int(bright*180))
        d.line([(lx_,ly_),(lx_+ll,ly_+rng2.integers(-3,4))],fill=col,width=1)
    d.rectangle([wx,wy_,wx+ww,wy_+wh],outline=(160,148,130),width=14)
    d.line([(wx+ww//2,wy_),(wx+ww//2,wy_+wh)],fill=(140,130,110),width=6)
    img=to_arr(pil); img=_vignette(img,0.50)
    return np.clip(img,0,1)

def scene_diving_board():
    img = np.zeros((H,W,3),np.float32)
    for y in range(int(H*0.55)):
        t=y/(H*0.55)
        img[y,:,0]=0.52+t*0.10; img[y,:,1]=0.68+t*0.08; img[y,:,2]=0.78+t*0.06
    for y in range(int(H*0.55),H):
        t=(y-H*0.55)/(H-H*0.55)
        img[y,:,0]=0.20+t*0.08; img[y,:,1]=0.48+t*0.12; img[y,:,2]=0.62+t*0.10
    n=fbm((H,W),scale=100,octaves=4,seed=60)
    img+=(n*0.04-0.02)[:,:,np.newaxis]
    pil=to_img(img); d=ImageDraw.Draw(pil)
    # board
    bx1,by1,bx2,by2=int(W*0.25),int(H*0.38),int(W*0.72),int(H*0.44)
    d.rectangle([bx1,by1,bx2,by2],fill=(220,210,190))
    # ladder
    lx=int(W*0.68)
    for y in range(int(H*0.44),int(H*0.88),22):
        d.line([(lx,y),(lx+30,y)],fill=(180,170,155),width=3)
    d.line([(lx,int(H*0.44)),(lx,int(H*0.88))],fill=(180,170,155),width=4)
    d.line([(lx+30,int(H*0.44)),(lx+30,int(H*0.88))],fill=(180,170,155),width=4)
    # reflection
    refl=np.array(pil).astype(np.float32)/255.0
    refl_blur=np.stack([gaussian_filter(refl[:,:,c],sigma=4) for c in range(3)],axis=2)
    img=to_arr(pil)
    pool_mask=np.zeros((H,W),np.float32)
    pool_mask[int(H*0.55):]=1.0
    img=img*(1-pool_mask*0.3)+refl_blur*pool_mask*0.3
    img=_vignette(img,0.35)
    return np.clip(img,0,1)

def scene_murmuration():
    img = np.zeros((H,W,3),np.float32)
    for y in range(H):
        t=y/H
        img[y,:,0]=0.62-t*0.30; img[y,:,1]=0.52-t*0.28; img[y,:,2]=0.48-t*0.28
    img+=(fbm((H,W),scale=150,octaves=3,seed=70)*0.05-0.025)[:,:,np.newaxis]
    pil=to_img(img); d=ImageDraw.Draw(pil)
    rng2=np.random.default_rng(71)
    cx,cy_=W*0.45,H*0.38
    for _ in range(2200):
        angle=rng2.random()*2*math.pi
        r=rng2.random()**0.6*(W*0.28)
        spread=1+0.4*math.sin(angle*3)
        x=int(cx+r*math.cos(angle)*spread+rng2.normal(0,8))
        y=int(cy_+r*math.sin(angle)*0.55*spread+rng2.normal(0,5))
        if 0<=x<W and 0<=y<H:
            sz=rng2.integers(1,4)
            d.ellipse([x-sz,y-sz,x+sz,y+sz],fill=(18,14,10))
    img=to_arr(pil); img=_vignette(img,0.38)
    return np.clip(img,0,1)

def scene_first_snow():
    img = np.ones((H,W,3),np.float32)
    for y in range(H):
        t=y/H
        img[y,:,0]=0.94-t*0.08; img[y,:,1]=0.95-t*0.07; img[y,:,2]=0.97-t*0.04
    n=fbm((H,W),scale=200,octaves=4,seed=80)
    img+=(n*0.03-0.015)[:,:,np.newaxis]
    pil=to_img(img); d=ImageDraw.Draw(pil)
    # ground line
    gy_=int(H*0.62)
    d.rectangle([0,gy_,W,H],fill=(235,238,242))
    # fence posts
    for fx in range(int(W*0.08),int(W*0.92),int(W*0.09)):
        d.rectangle([fx,int(H*0.48),fx+6,gy_],fill=(160,148,132))
    d.line([(int(W*0.08),int(H*0.56)),(int(W*0.92),int(H*0.56))],fill=(165,152,136),width=3)
    # snowfall
    rng2=np.random.default_rng(81)
    for _ in range(400):
        sx=rng2.integers(0,W); sy=rng2.integers(0,H)
        r=rng2.integers(1,4)
        d.ellipse([sx-r,sy-r,sx+r,sy+r],fill=(248,250,252))
    img=to_arr(pil); img=_vignette(img,0.22)
    return np.clip(img,0,1)

def scene_carnival_closing():
    img = np.zeros((H,W,3),np.float32)
    for y in range(H):
        t=y/H
        img[y,:,0]=0.08+t*0.06; img[y,:,1]=0.06+t*0.04; img[y,:,2]=0.12+t*0.04
    img+=(fbm((H,W),scale=80,octaves=3,seed=90)*0.03-0.015)[:,:,np.newaxis]
    pil=to_img(img); d=ImageDraw.Draw(pil)
    rng2=np.random.default_rng(91)
    colors=[(220,60,40),(40,160,80),(200,160,20),(40,100,200),(180,40,160)]
    for i,x in enumerate(range(int(W*0.08),int(W*0.92),int(W*0.06))):
        alive=rng2.random()>0.35
        h_=rng2.integers(int(H*0.20),int(H*0.55))
        col=colors[i%len(colors)]
        if alive:
            for yy in range(h_,int(H*0.72),20):
                d.ellipse([x-6,yy-6,x+6,yy+6],fill=col)
            d.line([(x,int(H*0.72)),(x,H)],fill=(80,70,60),width=6)
        else:
            d.rectangle([x-4,h_,x+4,int(H*0.72)],fill=(40,35,30))
    img=to_arr(pil)
    glow=np.zeros((H,W,3),np.float32)
    for x in range(int(W*0.08),int(W*0.92),int(W*0.06)):
        glow[:,x,0]=0.15; glow[:,x,1]=0.10; glow[:,x,2]=0.05
    for c in range(3): glow[:,:,c]=gaussian_filter(glow[:,:,c],sigma=30)
    img=np.clip(img+glow,0,1); img=_vignette(img,0.50)
    return np.clip(img,0,1)

def scene_eclipse():
    img = np.zeros((H,W,3),np.float32)
    for y in range(H):
        t=y/H
        img[y,:,0]=0.04+t*0.03; img[y,:,1]=0.04+t*0.03; img[y,:,2]=0.10+t*0.04
    img+=(fbm((H,W),scale=200,octaves=3,seed=100)*0.02-0.01)[:,:,np.newaxis]
    cx,cy_=W*0.50,H*0.40
    corona=np.zeros((H,W,3),np.float32)
    y_,x_=np.ogrid[:H,:W]
    dist=np.sqrt((x_-cx)**2+(y_-cy_)**2)
    for sig,s in [(60,0.55),(120,0.28),(220,0.14)]:
        mask=np.exp(-dist**2/(2*sig**2))
        corona[:,:,0]+=mask*s*0.95; corona[:,:,1]+=mask*s*0.82; corona[:,:,2]+=mask*s*0.42
    img=np.clip(img+corona,0,1)
    pil=to_img(img); d=ImageDraw.Draw(pil)
    r=int(W*0.095)
    d.ellipse([int(cx-r),int(cy_-r),int(cx+r),int(cy_+r)],fill=(4,4,8))
    img=to_arr(pil)
    horizon=np.zeros((H,W,3),np.float32)
    for y in range(int(H*0.78),H):
        t=(y-H*0.78)/(H-H*0.78)
        horizon[y,:]=np.array([0.18+t*0.06,0.12+t*0.04,0.08+t*0.02])
    img=np.clip(img+horizon,0,1); img=_vignette(img,0.40)
    return np.clip(img,0,1)

def scene_greenhouse():
    img = np.zeros((H,W,3),np.float32)
    for y in range(H):
        t=y/H
        img[y,:,0]=0.62+t*0.08; img[y,:,1]=0.72+t*0.06; img[y,:,2]=0.48+t*0.04
    img+=(fbm((H,W),scale=80,octaves=4,seed=110)*0.06-0.03)[:,:,np.newaxis]
    pil=to_img(img); d=ImageDraw.Draw(pil)
    rng2=np.random.default_rng(111)
    for _ in range(22):
        sx=rng2.integers(int(W*0.06),int(W*0.94))
        sy=rng2.integers(int(H*0.35),int(H*0.82))
        sh=rng2.integers(int(H*0.12),int(H*0.30))
        sw=rng2.integers(int(W*0.03),int(W*0.08))
        g=int(rng2.integers(80,160)); gr=int(rng2.integers(30,80))
        d.ellipse([sx-sw,sy-sh,sx+sw,sy],fill=(gr,g,gr//2))
        d.line([(sx,sy),(sx+rng2.integers(-20,20),sy+int(H*0.12))],fill=(80,110,40),width=3)
    # glass panes
    for x in range(0,W,int(W*0.12)):
        d.line([(x,0),(x,H)],fill=(200,220,210,60),width=2)
    for y in range(0,H,int(H*0.18)):
        d.line([(0,y),(W,y)],fill=(200,220,210,60),width=1)
    img=to_arr(pil); img=_vignette(img,0.30)
    return np.clip(img,0,1)

def scene_empty_cinema():
    img = np.zeros((H,W,3),np.float32)
    for y in range(H):
        t=y/H; img[y,:,0]=0.06+t*0.04; img[y,:,1]=0.05+t*0.03; img[y,:,2]=0.08+t*0.04
    img+=(fbm((H,W),scale=80,octaves=3,seed=120)*0.02-0.01)[:,:,np.newaxis]
    pil=to_img(img); d=ImageDraw.Draw(pil)
    # screen
    sx,sy_,sw,sh=int(W*0.08),int(H*0.04),int(W*0.84),int(H*0.48)
    for y in range(sy_,sy_+sh):
        t=(y-sy_)/sh
        c_=int((0.88+t*0.05)*255)
        d.rectangle([sx,y,sx+sw,y+1],fill=(c_,c_-8,c_-20))
    d.rectangle([sx,sy_,sx+sw,sy_+sh],outline=(220,215,205),width=3)
    # seats
    rng2=np.random.default_rng(121)
    for row,y in enumerate(range(int(H*0.58),int(H*0.94),int(H*0.09))):
        for col,x in enumerate(range(int(W*0.04),int(W*0.96),int(W*0.055))):
            occupied = (row==2 and col==5)  # one occupied seat — row F
            fill=(55,35,25) if not occupied else (80,50,30)
            d.rectangle([x,y,x+int(W*0.04),y+int(H*0.07)],fill=fill)
    img=to_arr(pil)
    # projector beam
    beam=np.zeros((H,W,3),np.float32)
    px,py_=int(W*0.50),H
    for y in range(int(H*0.50),H):
        t=1-(y-H*0.50)/(H-H*0.50)
        half_w=int((H-y)*0.45)
        x0=max(0,px-half_w); x1=min(W,px+half_w)
        beam[y,x0:x1]=np.array([0.92,0.88,0.72])*t*0.12
    for c in range(3): beam[:,:,c]=gaussian_filter(beam[:,:,c],sigma=6)
    img=np.clip(img+beam,0,1); img=_vignette(img,0.55)
    return np.clip(img,0,1)

def scene_glacier():
    img = np.zeros((H,W,3),np.float32)
    for y in range(int(H*0.55)):
        t=y/(H*0.55)
        img[y,:,0]=0.72+t*0.08; img[y,:,1]=0.82+t*0.06; img[y,:,2]=0.92+t*0.04
    for y in range(int(H*0.55),H):
        t=(y-H*0.55)/(H-H*0.55)
        img[y,:,0]=0.78-t*0.10; img[y,:,1]=0.88-t*0.12; img[y,:,2]=0.92-t*0.08
    n=fbm((H,W),scale=150,octaves=5,seed=130)
    img+=(n*0.06-0.03)[:,:,np.newaxis]
    pil=to_img(img); d=ImageDraw.Draw(pil)
    rng2=np.random.default_rng(131)
    for _ in range(18):
        x1=rng2.integers(0,W); y1=rng2.integers(int(H*0.3),int(H*0.65))
        x2=x1+rng2.integers(-120,120); y2=y1+rng2.integers(20,80)
        d.line([(x1,y1),(x2,y2)],fill=(140,175,210),width=rng2.integers(2,6))
    img=to_arr(pil); img=_vignette(img,0.28)
    return np.clip(img,0,1)

def scene_operating_theatre():
    img = np.zeros((H,W,3),np.float32)
    for y in range(H):
        t=y/H
        img[y,:,0]=0.82-t*0.06; img[y,:,1]=0.84-t*0.05; img[y,:,2]=0.84-t*0.04
    img+=(fbm((H,W),scale=60,octaves=3,seed=140)*0.012-0.006)[:,:,np.newaxis]
    pil=to_img(img); d=ImageDraw.Draw(pil)
    # surgical light
    lx,ly_,lr=int(W*0.50),int(H*0.12),int(W*0.14)
    d.ellipse([lx-lr,ly_-lr,lx+lr,ly_+lr],fill=(245,242,232))
    d.ellipse([lx-int(lr*0.5),ly_-int(lr*0.5),lx+int(lr*0.5),ly_+int(lr*0.5)],fill=(255,252,240))
    img2=to_arr(pil)
    glow=np.zeros((H,W,3),np.float32)
    glow[ly_,lx]=1.0
    for sig,s in [(40,0.45),(90,0.22),(180,0.10)]:
        b=gaussian_filter(glow[:,:,0],sigma=sig)
        img2[:,:,0]+=b*s*0.95; img2[:,:,1]+=b*s*0.92; img2[:,:,2]+=b*s*0.78
    # table
    tx,ty_,tw,th=int(W*0.18),int(H*0.50),int(W*0.64),int(H*0.12)
    d2=ImageDraw.Draw(to_img(img2))
    img2=np.clip(img2,0,1)
    pil2=to_img(img2); d2=ImageDraw.Draw(pil2)
    d2.rectangle([tx,ty_,tx+tw,ty_+th],fill=(210,210,205))
    img2=to_arr(pil2)
    img2=_vignette(img2,0.32)
    return np.clip(img2,0,1)

def scene_highway_3am():
    img = np.zeros((H,W,3),np.float32)
    for y in range(H):
        t=y/H; img[y,:,0]=0.04+t*0.03; img[y,:,1]=0.04+t*0.02; img[y,:,2]=0.06+t*0.03
    img+=(fbm((H,W),scale=100,octaves=3,seed=150)*0.015-0.007)[:,:,np.newaxis]
    pil=to_img(img); d=ImageDraw.Draw(pil)
    # road
    hy_=int(H*0.52)
    for y in range(hy_,H):
        t=(y-hy_)/(H-hy_)
        c_=int((0.14+t*0.06)*255)
        d.rectangle([0,y,W,y+1],fill=(c_,c_,c_))
    # centre line
    for y in range(hy_,H,30):
        t=(y-hy_)/(H-hy_)
        cx_=int(W*0.50); lw_=max(2,int(t*20))
        d.rectangle([cx_-lw_//2,y,cx_+lw_//2,y+14],fill=(200,190,140))
    # headlight cones
    for hx,side in [(int(W*0.40),-1),(int(W*0.60),1)]:
        cone=np.zeros((H,W,3),np.float32)
        for y in range(hy_,H):
            t=(y-hy_)/(H-hy_)
            half=int(t*W*0.20)
            x0=max(0,hx-half); x1=min(W,hx+half)
            cone[y,x0:x1]=np.array([0.95,0.90,0.72])*max(0,1-t)*0.5
        for c in range(3): cone[:,:,c]=gaussian_filter(cone[:,:,c],sigma=12)
        img2=to_arr(pil); img2=np.clip(img2+cone,0,1); pil=to_img(img2)
    img=to_arr(pil); img=_vignette(img,0.48)
    return np.clip(img,0,1)

def scene_tide_pool():
    img = np.zeros((H,W,3),np.float32)
    for y in range(int(H*0.45)):
        t=y/(H*0.45)
        img[y,:,0]=0.62+t*0.08; img[y,:,1]=0.72+t*0.06; img[y,:,2]=0.82+t*0.04
    for y in range(int(H*0.45),H):
        t=(y-H*0.45)/(H-H*0.45)
        img[y,:,0]=0.52-t*0.10; img[y,:,1]=0.58-t*0.08; img[y,:,2]=0.52-t*0.08
    n=fbm((H,W),scale=80,octaves=5,seed=160)
    img+=(n*0.05-0.025)[:,:,np.newaxis]
    pil=to_img(img); d=ImageDraw.Draw(pil)
    rng2=np.random.default_rng(161)
    # rocks
    for _ in range(8):
        rx=rng2.integers(int(W*0.05),int(W*0.95)); ry=rng2.integers(int(H*0.50),int(H*0.85))
        rw=rng2.integers(int(W*0.05),int(W*0.14)); rh=int(rw*0.5)
        c_=rng2.integers(80,130)
        d.ellipse([rx-rw,ry-rh,rx+rw,ry+rh],fill=(c_,c_-8,c_-14))
    # pool
    px,py_=int(W*0.35),int(H*0.58)
    pw,ph=int(W*0.30),int(H*0.18)
    for y in range(py_,py_+ph):
        t=(y-py_)/ph
        r=int((0.30+t*0.08)*255); g=int((0.52+t*0.06)*255); b=int((0.62+t*0.04)*255)
        d.rectangle([px,y,px+pw,y+1],fill=(r,g,b))
    img=to_arr(pil); img=_vignette(img,0.30)
    return np.clip(img,0,1)

def scene_two_glasses():
    """Main submission scene."""
    img = np.zeros((H,W,3),np.float32)
    for y in range(H):
        t=y/H; img[y,:,0]=0.42+t*0.18; img[y,:,1]=0.32+t*0.12; img[y,:,2]=0.18+t*0.06
    n=fbm((H,W),scale=120,octaves=5,seed=42)
    img+=(n*0.04-0.02)[:,:,np.newaxis]
    for y in range(int(H*0.55),H):
        t=(y-H*0.55)/(H-H*0.55)
        img[y,:,0]=np.clip(0.52+t*0.10+n[y,:]*0.03,0,1)
        img[y,:,1]=np.clip(0.38+t*0.06+n[y,:]*0.02,0,1)
        img[y,:,2]=np.clip(0.22+t*0.03+n[y,:]*0.01,0,1)
    for y in range(H):
        for x in range(int(W*0.65),W):
            t=(x-W*0.65)/(W-W*0.65)
            img[y,x,0]=np.clip(img[y,x,0]+t*0.25,0,1)
            img[y,x,1]=np.clip(img[y,x,1]+t*0.14,0,1)
            img[y,x,2]=np.clip(img[y,x,2]+t*0.04,0,1)
    pil=to_img(img); d=ImageDraw.Draw(pil)
    for gx,fill_h in [(int(W*0.38),0.18),(int(W*0.56),0.08)]:
        gy=int(H*0.56); gw=int(W*0.06); gh=int(H*0.22)
        d.rectangle([gx,gy,gx+gw,gy+gh],outline=(180,140,80),width=3)
        lh=int(gh*fill_h)
        d.rectangle([gx+2,gy+gh-lh,gx+gw-2,gy+gh],fill=(200,130,40))
    img=to_arr(pil); img=_vignette(img,0.35)
    return np.clip(img,0,1)

# ── All scenes ─────────────────────────────────────────────────────────────────

SCENES = [
    ("two_glasses",       scene_two_glasses,        "Two Glasses, Late Afternoon",  "my father's laugh at the end of a joke I can't remember"),
    ("empty_swing",       scene_empty_swing,         "Empty Swing",                  "something left swinging in the yard after"),
    ("phone_booth",       scene_phone_booth,         "Phone Booth",                  "her number still in my head after six years"),
    ("hospital_window",   scene_hospital_window,     "Hospital Window",              "the last thing she looked at from that room"),
    ("lighthouse",        scene_lighthouse,           "Lighthouse",                   "the summer we found it and thought we'd come back"),
    ("train_window",      scene_train_window,         "Train Window",                 "I don't remember when I stopped turning to look"),
    ("diving_board",      scene_diving_board,         "Diving Board",                 "we never went back after that summer"),
    ("murmuration",       scene_murmuration,          "Murmuration",                  "how they would move through a room"),
    ("first_snow",        scene_first_snow,           "First Snow",                   "the morning before everything changed"),
    ("carnival_closing",  scene_carnival_closing,     "Carnival Closing",             "sitting on the hood of the car watching"),
    ("eclipse",           scene_eclipse,              "Eclipse",                      "we drove four hours to stand in a field"),
    ("greenhouse",        scene_greenhouse,           "Greenhouse",                   "she could name every plant in there"),
    ("empty_cinema",      scene_empty_cinema,         "Empty Cinema",                 "every film we saw in that seat, row F"),
    ("glacier",           scene_glacier,              "Glacier",                      "the scale of what we cannot hold onto"),
    ("operating_theatre", scene_operating_theatre,    "Operating Theatre",            "what the ceiling looks like from that table"),
    ("highway_3am",       scene_highway_3am,          "Highway 3AM",                  "driving back when there was nothing left to say"),
    ("tide_pool",         scene_tide_pool,            "Tide Pool",                    "the things that live in that small world"),
]

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("DEAD RECKONING — Rendering all scenes\n")
    for slug, fn, title, text in SCENES:
        outpath = f"{OUT}/dr_{slug}.jpg"
        print(f"→ {title}...")
        base   = fn()
        frames = make_frames(base)
        compose_2x2(frames, title, text, outpath)
    print(f"\nDone. {len(SCENES)} scenes saved to {OUT}/")
