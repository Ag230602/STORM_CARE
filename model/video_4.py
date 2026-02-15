"""
video_7_FINAL_globe_plus_population_map_with_hurdat2_track.py

✅ What this script does (final integrated version)
- Left: AOML radar/globe frames (storm motion)
- Right: Florida "population/exposure proxy" panel that contains:
    1) Heatmap (proxy exposure density — default uses school density)
    2) Schools / Hospitals / Shelters symbols at fixed, correct locations
    3) ✅ Hurricane Irma trajectory (parsed from HURDAT2 text) drawn directly ON the same panel
    4) ✅ Moving storm marker that progresses along the track during the 30s clip

Inputs (your paths already set below):
- HURDAT2 TXT: C:\\Users\\Adrija\\Downloads\\DFGCN\\data\\raw\\hurdat2\\hurdat2_atlantic.txt
- schools.csv / hospitals.csv / shelters.csv

Output:
- irma_aoml_30s_globe_plus_population_with_track.mp4

Run:
  python video_7_FINAL_globe_plus_population_map_with_hurdat2_track.py
"""

import re
import tarfile
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import imageio
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from bs4 import BeautifulSoup
from ftplib import FTP


# =========================
# USER PATHS (YOUR FINAL PATHS)
# =========================

HURDAT2_TXT = Path(r"C:\Users\Adrija\Downloads\DFGCN\data\raw\hurdat2\hurdat2_atlantic.txt")

SCHOOLS_CSV = Path(r"C:\Users\Adrija\Downloads\DFGCN\your-repo\data\data\raw\data\raw\facilities\schools.csv")
HOSPITALS_CSV = Path(r"C:\Users\Adrija\Downloads\DFGCN\your-repo\data\data\raw\data\raw\facilities\hospitals.csv")
SHELTERS_CSV = Path(r"C:\Users\Adrija\Downloads\DFGCN\your-repo\data\data\raw\data\raw\facilities\shelters.csv")


# =========================
# SETTINGS
# =========================

RADAR_PAGE_URL = "https://www.aoml.noaa.gov/hrd/Storm_pages/irma2017/radar.html"

OUT_DIR = Path("aoml_download")
OUT_MP4 = Path("irma_aoml_30s_globe_plus_population_with_track.mp4")

DURATION_SEC = 30
FPS = 24

AUTO_CROP_WHITE_MARGINS = True
CROP_EXTRA_PX = 6

LINK_INDEX = 0
TARBALL_INDEX = 0

# Florida bounds for the right-side panel (tune if you want)
FL_BOUNDS = dict(
    lon_min=-88.0,
    lon_max=-79.0,
    lat_min=24.0,
    lat_max=31.5,
)

# Inset panel size
INSET_W = 380
INSET_H = 440
INSET_MARGIN = 10

# Heatmap config (proxy exposure density)
INSET_HEATMAP = True
HEATMAP_ALPHA = 0.45
HEATMAP_BLUR_RADIUS = 14
HEATMAP_QUANTILE_CLIP = 0.995
HEATMAP_SOURCE = "schools"  # "schools" | "hospitals" | "shelters"

# Facilities
DRAW_SCHOOLS = True
DRAW_HOSPITALS = True
DRAW_SHELTERS = True

SCHOOL_RADIUS_PX = 5
SHELTER_RADIUS_PX = 5
HOSPITAL_SIZE_PX = 7

# Hurricane path styling (on the right panel)
DRAW_HURRICANE_PATH = True
DRAW_MOVING_STORM_MARKER = True

PATH_COLOR = (255, 200, 0)        # yellow/gold
PATH_OUTLINE = (0, 0, 0)          # black outline
PATH_WIDTH = 3
PATH_OUTLINE_WIDTH = 5

MARKER_FILL = (255, 60, 60)       # red
MARKER_OUTLINE = (255, 255, 255)  # white
MARKER_RADIUS = 7

# Titles
TITLE_TEXT = "Hurricane Irma (2017) — Hazard Movement + Exposure Map"
SUBTITLE_TEXT = "Left: radar/globe motion | Right: exposure proxy + facilities + Irma track (HURDAT2)"


# =========================
# AOML DOWNLOAD HELPERS
# =========================

def autocrop_white(img: Image.Image, extra_px: int = 0) -> Image.Image:
    gray = img.convert("L")
    bw = gray.point(lambda p: 0 if p > 245 else 255, mode="1")
    bbox = bw.getbbox()
    if bbox is None:
        return img
    l, u, r, d = bbox
    l = max(0, l - extra_px)
    u = max(0, u - extra_px)
    r = min(img.width, r + extra_px)
    d = min(img.height, d + extra_px)
    return img.crop((l, u, r, d))


def normalize_to_ftp(href: str) -> str:
    href = href.strip()
    if href.startswith("//"):
        href = "https:" + href
    if not re.match(r"^[a-zA-Z]+://", href):
        href = "https://" + href
    u = urlparse(href)
    return f"ftp://{u.netloc}{u.path}"


def find_singleframe_links(page_url: str):
    html = requests.get(page_url, timeout=60).text
    soup = BeautifulSoup(html, "lxml")

    hits = []
    for a in soup.find_all("a"):
        text = (a.get_text() or "").strip().lower()
        href = (a.get("href") or "").strip()
        if not href:
            continue
        if ("single frame" in text) and ("gif" in text) and ("ftp.aoml.noaa.gov" in href.lower()):
            hits.append(normalize_to_ftp(href))

    seen = set()
    out = []
    for x in hits:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def ftp_connect(host: str, timeout=180):
    ftp = FTP(host, timeout=timeout)
    ftp.login()
    ftp.set_pasv(True)
    return ftp


def ftp_download_file(ftp_url: str, out_path: Path):
    u = urlparse(ftp_url)
    host = u.hostname
    file_path = u.path
    folder = str(Path(file_path).parent).replace("\\", "/")
    filename = Path(file_path).name

    out_path.parent.mkdir(parents=True, exist_ok=True)

    ftp = ftp_connect(host)
    ftp.cwd(folder)

    print(f"[INFO] Downloading:\n  {ftp_url}\n  -> {out_path.resolve()}")
    with open(out_path, "wb") as f:
        ftp.retrbinary(f"RETR {filename}", f.write, blocksize=1024 * 64)

    ftp.quit()
    return out_path


def ftp_list_tarballs_in_folder(ftp_folder_url: str):
    u = urlparse(ftp_folder_url)
    host = u.hostname
    folder_path = u.path

    ftp = ftp_connect(host)
    ftp.cwd(folder_path)

    files = ftp.nlst()
    tarballs = [f for f in files if f.lower().endswith((".tar.gz", ".tgz"))]
    tarballs.sort()

    ftp.quit()
    return host, folder_path, tarballs


def extract_tarball(tar_path: Path, out_folder: Path):
    out_folder.mkdir(parents=True, exist_ok=True)
    mode = "r:gz" if tar_path.name.lower().endswith(".tar.gz") else "r:*"
    print(f"[INFO] Extracting: {tar_path.name} -> {out_folder.resolve()}")
    with tarfile.open(tar_path, mode) as tf:
        tf.extractall(out_folder)
    return out_folder


def gather_frames(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".gif"}
    paths = [p for p in folder.rglob("*") if p.suffix.lower() in exts]
    if not paths:
        raise RuntimeError(f"[ERROR] No frames found in {folder.resolve()}")

    def key(p: Path):
        nums = re.findall(r"\d+", p.stem)
        return (int(nums[-1]) if nums else 10**18, p.name.lower())

    paths.sort(key=key)
    return paths


# =========================
# DATA LOADING
# =========================

def load_facility_csv(path: Path):
    df = pd.read_csv(path)
    if "X" not in df.columns or "Y" not in df.columns:
        raise ValueError(f"{path} must contain X (lon) and Y (lat). Found: {df.columns.tolist()}")
    df = df.rename(columns={"X": "lon", "Y": "lat"})
    df = df[np.isfinite(df["lon"]) & np.isfinite(df["lat"])].copy()
    return df


def lonlat_to_xy(lon, lat, w, h, bounds):
    x = (lon - bounds["lon_min"]) / (bounds["lon_max"] - bounds["lon_min"]) * w
    y = (bounds["lat_max"] - lat) / (bounds["lat_max"] - bounds["lat_min"]) * h
    return int(round(x)), int(round(y))


# =========================
# HURDAT2 PARSER (IRMA)
# =========================

def _parse_latlon_token(tok: str):
    """
    HURDAT2 lat/lon tokens look like:
      16.7N ,  053.5W
    Returns signed float degrees.
    """
    tok = tok.strip()
    if not tok:
        return None
    m = re.match(r"^(\d+(?:\.\d+)?)([NSEW])$", tok)
    if not m:
        return None
    val = float(m.group(1))
    hemi = m.group(2)
    if hemi in ("S", "W"):
        val = -val
    return val


def load_irma_track_from_hurdat2(hurdat2_txt: Path):
    """
    Extract Irma 2017 track from HURDAT2 Atlantic file.
    Returns DataFrame with columns: lon, lat, datetime (optional), record_id
    """
    if not hurdat2_txt.exists():
        raise FileNotFoundError(f"HURDAT2 file not found: {hurdat2_txt}")

    lines = hurdat2_txt.read_text(encoding="utf-8", errors="ignore").splitlines()

    header_idx = None
    n_entries = 0

    # Find Irma header: "AL112017, IRMA,  ..."
    for i, line in enumerate(lines):
        if line.upper().startswith("AL112017") and "IRMA" in line.upper():
            header_idx = i
            parts = [p.strip() for p in line.split(",")]
            # parts[2] = number of entries
            try:
                n_entries = int(parts[2])
            except Exception:
                n_entries = 0
            break

    if header_idx is None:
        raise RuntimeError("Could not find AL112017 IRMA header in HURDAT2 file.")

    track_lines = lines[header_idx + 1: header_idx + 1 + n_entries]
    rows = []
    for ln in track_lines:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 6:
            continue

        date = parts[0]  # YYYYMMDD
        time = parts[1]  # HHMM
        rec_id = parts[2]  # e.g., "HU", "TD" ...
        lat_tok = parts[4]
        lon_tok = parts[5]

        lat = _parse_latlon_token(lat_tok)
        lon = _parse_latlon_token(lon_tok)

        if lat is None or lon is None:
            continue

        # datetime string (optional)
        dt = None
        if len(date) == 8 and len(time) == 4:
            dt = f"{date[:4]}-{date[4:6]}-{date[6:8]} {time[:2]}:{time[2:4]}"

        rows.append({"lat": lat, "lon": lon, "datetime": dt, "record_id": rec_id})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Irma track parsed but resulted in empty dataframe (unexpected).")
    return df


# =========================
# HEATMAP
# =========================

def make_density_grid(df, w, h, bounds, step=2):
    grid = np.zeros((h // step, w // step), dtype=np.float32)
    ww, hh = w // step, h // step
    for lon, lat in zip(df["lon"].to_numpy(), df["lat"].to_numpy()):
        x, y = lonlat_to_xy(lon, lat, w, h, bounds)
        x //= step
        y //= step
        if 0 <= x < ww and 0 <= y < hh:
            grid[y, x] += 1.0
    return grid, step


def grid_to_heat_rgba(grid, step, w, h, blur_radius, alpha, q_clip):
    if np.max(grid) <= 0:
        heat = np.zeros((h, w), dtype=np.float32)
    else:
        positive = grid[grid > 0]
        clipv = np.quantile(positive, q_clip) if positive.size else np.max(grid)
        clipv = max(float(clipv), 1e-6)
        g = np.clip(grid, 0, clipv) / clipv
        if step > 1:
            g = np.kron(g, np.ones((step, step), dtype=np.float32))
        heat = g[:h, :w]

    heat_img = Image.fromarray((heat * 255).astype(np.uint8), mode="L")
    heat_img = heat_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    heat2 = np.asarray(heat_img).astype(np.float32) / 255.0

    r = np.clip(heat2 * 1.25, 0, 1)
    g = np.clip((1 - np.abs(heat2 - 0.5) * 2) * 1.0, 0, 1)
    b = np.zeros_like(heat2)
    a = np.clip(heat2 * alpha, 0, 1)

    rgba = np.stack([
        (r * 255).astype(np.uint8),
        (g * 255).astype(np.uint8),
        (b * 255).astype(np.uint8),
        (a * 255).astype(np.uint8),
    ], axis=-1)
    return Image.fromarray(rgba, mode="RGBA")


def alpha_blend(base_rgb: Image.Image, overlay_rgba: Image.Image) -> Image.Image:
    base = base_rgb.convert("RGBA")
    overlay = overlay_rgba.convert("RGBA")
    if base.size != overlay.size:
        overlay = overlay.resize(base.size, resample=Image.BILINEAR)
    out = Image.alpha_composite(base, overlay)
    return out.convert("RGB")


# =========================
# TEXT / FONTS
# =========================

def get_font(size=16):
    for name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_main_title(img: Image.Image):
    w, _ = img.size
    draw = ImageDraw.Draw(img)
    font_title = get_font(18)
    font_sub = get_font(13)

    band_h = 44
    draw.rectangle((0, 0, w, band_h), fill=(0, 0, 0, 160))
    draw.text((10, 6), TITLE_TEXT, fill=(255, 255, 255), font=font_title)
    draw.text((10, 24), SUBTITLE_TEXT, fill=(220, 220, 220), font=font_sub)
    return img


# =========================
# BUILD RIGHT PANEL (ONCE)
# =========================

def build_inset_base_and_track_pixels(schools, hospitals, shelters, track_df, bounds, w=INSET_W, h=INSET_H):
    """
    Builds the right-side panel once (static), with hurricane path drawn on top.
    Also returns track_pixels used for the moving marker during video generation.
    """
    panel = Image.new("RGB", (w, h), (15, 15, 18))
    d = ImageDraw.Draw(panel)

    d.rectangle((0, 0, w - 1, h - 1), outline=(255, 255, 255))
    d.rectangle((0, 0, w, 44), fill=(0, 0, 0))
    d.text((10, 6), "Exposure / Population Proxy Map (Florida)", fill=(255, 255, 255), font=get_font(15))
    d.text((10, 24), "Facilities + Irma track (HURDAT2)", fill=(210, 210, 210), font=get_font(12))

    plot_x0, plot_y0 = 10, 55
    plot_x1, plot_y1 = w - 10, h - 130
    d.rectangle((plot_x0, plot_y0, plot_x1, plot_y1), fill=(28, 28, 34), outline=(200, 200, 200))

    plot_w = plot_x1 - plot_x0
    plot_h = plot_y1 - plot_y0

    def to_plot_xy(lon, lat):
        x = plot_x0 + (lon - bounds["lon_min"]) / (bounds["lon_max"] - bounds["lon_min"]) * plot_w
        y = plot_y0 + (bounds["lat_max"] - lat) / (bounds["lat_max"] - bounds["lat_min"]) * plot_h
        return int(round(x)), int(round(y))

    # Grid lines
    for frac in [0.25, 0.5, 0.75]:
        gx = int(plot_x0 + frac * plot_w)
        gy = int(plot_y0 + frac * plot_h)
        d.line((gx, plot_y0, gx, plot_y1), fill=(70, 70, 80))
        d.line((plot_x0, gy, plot_x1, gy), fill=(70, 70, 80))

    # Heatmap
    if INSET_HEATMAP:
        src = HEATMAP_SOURCE.lower().strip()
        if src == "schools":
            heat_df = schools
        elif src == "hospitals":
            heat_df = hospitals
        elif src == "shelters":
            heat_df = shelters
        else:
            heat_df = schools

        grid, step = make_density_grid(heat_df, plot_w, plot_h, bounds, step=2)
        heat_rgba = grid_to_heat_rgba(
            grid=grid, step=step, w=plot_w, h=plot_h,
            blur_radius=HEATMAP_BLUR_RADIUS,
            alpha=HEATMAP_ALPHA,
            q_clip=HEATMAP_QUANTILE_CLIP
        )

        plot_bg = panel.crop((plot_x0, plot_y0, plot_x1, plot_y1)).convert("RGB")
        if heat_rgba.size != plot_bg.size:
            heat_rgba = heat_rgba.resize(plot_bg.size, resample=Image.BILINEAR)
        plot_bg = alpha_blend(plot_bg, heat_rgba)
        panel.paste(plot_bg, (plot_x0, plot_y0))
        d = ImageDraw.Draw(panel)

    # Track pixels
    track_pixels = []
    if track_df is not None and len(track_df) > 1:
        for lon, lat in zip(track_df["lon"], track_df["lat"]):
            x, y = to_plot_xy(lon, lat)
            if plot_x0 <= x <= plot_x1 and plot_y0 <= y <= plot_y1:
                track_pixels.append((x, y))

    # Draw hurricane path directly on population map
    if DRAW_HURRICANE_PATH and len(track_pixels) >= 2:
        d.line(track_pixels, fill=PATH_OUTLINE, width=PATH_OUTLINE_WIDTH, joint="curve")
        d.line(track_pixels, fill=PATH_COLOR, width=PATH_WIDTH, joint="curve")

        sx, sy = track_pixels[0]
        ex, ey = track_pixels[-1]
        d.ellipse((sx - 5, sy - 5, sx + 5, sy + 5), fill=(0, 255, 255), outline=(0, 0, 0))
        d.ellipse((ex - 6, ey - 6, ex + 6, ey + 6), fill=(255, 255, 255), outline=(0, 0, 0))

    # Draw facilities on top
    if DRAW_SCHOOLS:
        fill = (0, 120, 255)
        outline = (255, 255, 255)
        r = SCHOOL_RADIUS_PX
        for lon, lat in zip(schools["lon"], schools["lat"]):
            x, y = to_plot_xy(lon, lat)
            if plot_x0 <= x <= plot_x1 and plot_y0 <= y <= plot_y1:
                d.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=outline)

    if DRAW_SHELTERS:
        fill = (0, 200, 0)
        outline = (255, 255, 255)
        r = SHELTER_RADIUS_PX
        for lon, lat in zip(shelters["lon"], shelters["lat"]):
            x, y = to_plot_xy(lon, lat)
            if plot_x0 <= x <= plot_x1 and plot_y0 <= y <= plot_y1:
                pts = [(x, y - r - 2), (x - r - 2, y + r + 2), (x + r + 2, y + r + 2)]
                d.polygon(pts, fill=fill, outline=outline)

    if DRAW_HOSPITALS:
        color = (255, 0, 0)
        outline = (255, 255, 255)
        s = HOSPITAL_SIZE_PX
        for lon, lat in zip(hospitals["lon"], hospitals["lat"]):
            x, y = to_plot_xy(lon, lat)
            if plot_x0 <= x <= plot_x1 and plot_y0 <= y <= plot_y1:
                d.line((x - s, y, x + s, y), fill=outline, width=4)
                d.line((x, y - s, x, y + s), fill=outline, width=4)
                d.line((x - s, y, x + s, y), fill=color, width=2)
                d.line((x, y - s, x, y + s), fill=color, width=2)

    # Legend
    leg_y0 = h - 120
    d.rectangle((10, leg_y0, w - 10, h - 10), fill=(0, 0, 0), outline=(255, 255, 255))
    font_leg = get_font(12)
    y = leg_y0 + 8
    x = 18

    d.rectangle((x, y + 3, x + 18, y + 13), fill=(255, 180, 0), outline=(255, 255, 255))
    d.text((x + 28, y), "Heatmap: exposure density (proxy)", fill=(255, 255, 255), font=font_leg)
    y += 18

    d.line((x, y + 8, x + 18, y + 8), fill=PATH_OUTLINE, width=PATH_OUTLINE_WIDTH)
    d.line((x, y + 8, x + 18, y + 8), fill=PATH_COLOR, width=PATH_WIDTH)
    d.text((x + 28, y), "Irma track (HURDAT2)", fill=(255, 255, 255), font=font_leg)
    y += 18

    d.ellipse((x + 5, y + 4, x + 13, y + 12), fill=(0, 120, 255), outline=(255, 255, 255))
    d.text((x + 28, y), "Schools", fill=(255, 255, 255), font=font_leg)
    y += 16

    d.line((x + 4, y + 8, x + 14, y + 8), fill=(255, 255, 255), width=4)
    d.line((x + 9, y + 3, x + 9, y + 13), fill=(255, 255, 255), width=4)
    d.line((x + 4, y + 8, x + 14, y + 8), fill=(255, 0, 0), width=2)
    d.line((x + 9, y + 3, x + 9, y + 13), fill=(255, 0, 0), width=2)
    d.text((x + 28, y), "Hospitals", fill=(255, 255, 255), font=font_leg)
    y += 16

    d.polygon([(x + 9, y + 2), (x + 3, y + 14), (x + 15, y + 14)], fill=(0, 200, 0), outline=(255, 255, 255))
    d.text((x + 28, y), "Shelters", fill=(255, 255, 255), font=font_leg)

    return panel, track_pixels


# =========================
# VIDEO WRITER
# =========================

def write_30s_mp4(frame_paths, out_mp4: Path, fps: int, duration_sec: int, inset_base: Image.Image, track_pixels):
    n_out = fps * duration_sec
    idx = np.linspace(0, len(frame_paths) - 1, n_out).round().astype(int)

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Writing MP4: {out_mp4.resolve()}")
    print(f"[INFO] Source frames: {len(frame_paths)} -> Output: {n_out} frames @ {fps} fps (~{duration_sec}s)")

    with imageio.get_writer(out_mp4, fps=fps, codec="libx264") as writer:
        for k, i in enumerate(idx):
            img = Image.open(frame_paths[i]).convert("RGB")
            if AUTO_CROP_WHITE_MARGINS:
                img = autocrop_white(img, extra_px=CROP_EXTRA_PX)

            img = draw_main_title(img)
            fw, fh = img.size

            inset = inset_base
            if DRAW_MOVING_STORM_MARKER and len(track_pixels) >= 2:
                inset = inset_base.copy()
                d = ImageDraw.Draw(inset)
                j = int(round((k / max(1, (n_out - 1))) * (len(track_pixels) - 1)))
                j = max(0, min(len(track_pixels) - 1, j))
                mx, my = track_pixels[j]
                d.ellipse(
                    (mx - MARKER_RADIUS, my - MARKER_RADIUS, mx + MARKER_RADIUS, my + MARKER_RADIUS),
                    fill=MARKER_FILL,
                    outline=MARKER_OUTLINE,
                    width=2,
                )

            # Resize inset if too tall
            max_h = fh - 2 * INSET_MARGIN - 44
            if inset.size[1] > max_h:
                scale = max_h / inset.size[1]
                nw = max(120, int(inset.size[0] * scale))
                nh = max(120, int(inset.size[1] * scale))
                inset = inset.resize((nw, nh), resample=Image.BILINEAR)

            x = fw - inset.size[0] - INSET_MARGIN
            y = INSET_MARGIN + 44
            if y + inset.size[1] > fh - INSET_MARGIN:
                y = fh - inset.size[1] - INSET_MARGIN

            bg_pad = 6
            dmain = ImageDraw.Draw(img)
            dmain.rectangle(
                (x - bg_pad, y - bg_pad, x + inset.size[0] + bg_pad, y + inset.size[1] + bg_pad),
                fill=(0, 0, 0, 160),
                outline=(255, 255, 255),
            )

            img.paste(inset, (x, y))
            writer.append_data(np.array(img))

            if (k + 1) % 120 == 0:
                print(f"  wrote {k+1}/{n_out}")

    print("[OK] Done.")


# =========================
# MAIN
# =========================

def main():
    # Sanity checks
    if not HURDAT2_TXT.exists():
        raise FileNotFoundError(f"HURDAT2 file not found: {HURDAT2_TXT}")
    if not SCHOOLS_CSV.exists():
        raise FileNotFoundError(f"Schools CSV not found: {SCHOOLS_CSV}")
    if not HOSPITALS_CSV.exists():
        raise FileNotFoundError(f"Hospitals CSV not found: {HOSPITALS_CSV}")
    if not SHELTERS_CSV.exists():
        raise FileNotFoundError(f"Shelters CSV not found: {SHELTERS_CSV}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load facilities
    schools = load_facility_csv(SCHOOLS_CSV)
    hospitals = load_facility_csv(HOSPITALS_CSV)
    shelters = load_facility_csv(SHELTERS_CSV)
    print(f"[INFO] Loaded schools={len(schools)} hospitals={len(hospitals)} shelters={len(shelters)}")

    # Load Irma track from HURDAT2
    track_df = load_irma_track_from_hurdat2(HURDAT2_TXT)
    print(f"[INFO] Parsed Irma track from HURDAT2: points={len(track_df)}")

    # Download AOML frames
    links = find_singleframe_links(RADAR_PAGE_URL)
    if not links:
        raise RuntimeError("[ERROR] Could not find single-frame GIF links on the AOML page.")
    link = links[min(LINK_INDEX, len(links) - 1)]
    print(f"[INFO] Using link: {link}")

    extracted = OUT_DIR / "frames_extracted"

    if link.lower().endswith((".tar.gz", ".tgz")):
        tar_path = OUT_DIR / Path(urlparse(link).path).name
        ftp_download_file(link, tar_path)
        extract_tarball(tar_path, extracted)
    else:
        host, folder_path, tarballs = ftp_list_tarballs_in_folder(link)
        if not tarballs:
            raise RuntimeError(f"[ERROR] No .tar.gz/.tgz found in FTP folder:\n  {link}")
        tar_name = tarballs[min(TARBALL_INDEX, len(tarballs) - 1)]
        tar_url = f"ftp://{host}{folder_path.rstrip('/')}/{tar_name}"
        tar_path = OUT_DIR / tar_name
        ftp_download_file(tar_url, tar_path)
        extract_tarball(tar_path, extracted)

    frames = gather_frames(extracted)

    # Build right-side panel (static base + track pixels)
    inset_base, track_pixels = build_inset_base_and_track_pixels(
        schools=schools,
        hospitals=hospitals,
        shelters=shelters,
        track_df=track_df,
        bounds=FL_BOUNDS,
        w=INSET_W,
        h=INSET_H,
    )
    if len(track_pixels) < 2:
        print("[WARN] Track pixels < 2 inside FL_BOUNDS. Consider widening FL_BOUNDS.")

    # Render final video
    write_30s_mp4(
        frame_paths=frames,
        out_mp4=OUT_MP4,
        fps=FPS,
        duration_sec=DURATION_SEC,
        inset_base=inset_base,
        track_pixels=track_pixels,
    )


if __name__ == "__main__":
    main()
