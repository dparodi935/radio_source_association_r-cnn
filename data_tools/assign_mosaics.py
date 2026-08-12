"""assign_mosaics.py -- run ONCE, before training.

Takes the FULL DR1 catalogues plus the DR2 mosaic images and splits the
catalogues into per-mosaic files, assigning each source to exactly one mosaic.

Assignment rule: among the mosaics whose PIXEL GRID actually contains the
source (with a margin so a full cutout fits), take the one whose centre is
nearest. Sources contained by no mosaic are dropped.

This replaces nearest-centre-only assignment, which handed edge mosaics huge
numbers of sources lying outside their images -- those then showed up as
'edge-skipped' during preprocessing and their siblings fell out of frame,
degrading the ground truth.
"""
import os, sys

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add the folder 'test_cnn' to  the sys path

from cutouts import discover_mosaics, DIR_MOSAIC, DIR_RAW, DIR_LARGE, DIR_COMP

MAJ_MIN_ARCSEC = 15.0        # cutout-centre selection (Mostert et al. 2022)
FLUX_MIN_MJY = 10.0
CUTOUT_PX = 200              # must match training --size
MARGIN_PX = CUTOUT_PX / 2    # so a full cutout fits inside the image


def mosaic_geometry(data_root, ids):
    """Per mosaic: WCS, pixel dims, and centre RA/Dec."""
    geo = []
    for mid in ids:
        path = os.path.join(data_root, DIR_MOSAIC, f"mosaic_{mid}.fits")
        with fits.open(path) as f:
            w = WCS(f[0].header).celestial
            ny, nx = np.squeeze(f[0].data).shape
        ra, dec = w.all_pix2world(nx / 2.0, ny / 2.0, 0)
        geo.append(dict(mid=mid, wcs=w, nx=nx, ny=ny,
                        ra=float(ra), dec=float(dec)))
    return geo


def assign(data_root, raw_path, comp_path, margin_px=MARGIN_PX):
    ids = discover_mosaics(data_root)
    if not ids:
        raise RuntimeError(f"no mosaics under {data_root}")
    geo = mosaic_geometry(data_root, ids)
    print(f"{len(ids)} mosaics")

    raw = Table.read(raw_path)
    comp = Table.read(comp_path)
    print(f"full raw catalogue : {len(raw)} rows")
    print(f"full comp catalogue: {len(comp)} rows")

    ra = np.asarray(raw["RA"], float)
    dec = np.asarray(raw["DEC"], float)
    n_src = len(raw)

    # ---- containment: which mosaics' pixel grids hold each source? ----
    # dist[i, j] = angular separation to mosaic j, or inf if not contained
    dist = np.full((n_src, len(geo)), np.inf, dtype=np.float32)
    for j, g in enumerate(geo):
        x, y = g["wcs"].all_world2pix(ra, dec, 0)
        inside = (np.isfinite(x) & np.isfinite(y) &
                  (x >= margin_px) & (x < g["nx"] - margin_px) &
                  (y >= margin_px) & (y < g["ny"] - margin_px))
        cosd = np.cos(np.radians(g["dec"]))
        d = np.hypot((ra - g["ra"]) * cosd, dec - g["dec"])
        dist[inside, j] = d[inside]
        print(f"  {g['mid']}: image {g['nx']}x{g['ny']}, "
              f"contains {int(inside.sum())} sources")

    contained = np.isfinite(dist).any(axis=1)
    owner = np.full(n_src, -1, dtype=int)
    owner[contained] = np.argmin(dist[contained], axis=1)

    n_multi = int((np.isfinite(dist).sum(axis=1) > 1).sum())
    print(f"\ncontained by >=1 mosaic : {int(contained.sum())} / {n_src}")
    print(f"contained by >1 mosaic  : {n_multi} "
          f"(resolved by nearest centre -> no duplicates)")
    print(f"dropped (outside all)   : {int((~contained).sum())}")

    # component -> value-added source, for keeping association groups whole
    cn = np.asarray(comp["Component_Name"]).astype(str)
    va = np.asarray(comp["Source_Name"]).astype(str)
    va_of = dict(zip(cn.tolist(), va.tolist()))

    for d in (DIR_RAW, DIR_LARGE, DIR_COMP):
        os.makedirs(os.path.join(data_root, d), exist_ok=True)

    print()
    total = 0
    for j, g in enumerate(geo):
        mid = g["mid"]
        sub = raw[owner == j]
        sub.write(os.path.join(data_root, DIR_RAW, f"{mid}.fits"), overwrite=True)

        big = ((np.asarray(sub["Maj"], float) > MAJ_MIN_ARCSEC) &
               (np.asarray(sub["Total_flux"], float) > FLUX_MIN_MJY))
        sub[big].write(os.path.join(data_root, DIR_LARGE, f"{mid}.fits"),
                       overwrite=True)

        # component rows for this mosaic's components AND their siblings, so
        # members_of never returns a partial group
        mine = set(np.asarray(sub["Source_Name"]).astype(str))
        wanted = {va_of[n] for n in mine if n in va_of}
        keep = np.isin(va, list(wanted))
        comp[keep].write(os.path.join(data_root, DIR_COMP, f"{mid}.fits"),
                         overwrite=True)

        total += len(sub)
        print(f"  {mid}: {len(sub):6d} components, {int(big.sum()):5d} centres, "
              f"{int(keep.sum()):6d} comp rows")

    print(f"\nassigned {total} (== contained count: {total == int(contained.sum())})")
    if total != int(contained.sum()):
        print("  WARNING: mismatch -- check for NaN coordinates in the catalogue")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join("..", "cnn_data"))
    ap.add_argument("--raw", required=True, help="full DR1 PyBDSF catalogue")
    ap.add_argument("--comp", required=True, help="full DR1 component catalogue")
    ap.add_argument("--margin-px", type=float, default=MARGIN_PX,
                    help="keep sources this far inside the image edge")
    a = ap.parse_args()
    assign(a.data_root, a.raw, a.comp, a.margin_px)