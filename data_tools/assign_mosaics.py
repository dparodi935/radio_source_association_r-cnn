"""assign_mosaics.py -- run ONCE, before training.

Takes the FULL DR1 catalogues and the DR2 mosaic images, and splits the
catalogues into per-mosaic files with each source assigned to exactly one
mosaic (nearest pointing centre). This replaces a positional footprint cut,
which double-counts sources in mosaic overlap regions.

Writes, per mosaic id:
    <DIR_RAW>/<mid>.fits      assigned components (proposals + neighbours)
    <DIR_LARGE>/<mid>.fits    the large+bright subset (cutout centres)
    <DIR_COMP>/<mid>.fits     component catalogue rows for those components,
                              PLUS any sibling rows needed to keep groups whole
"""
import os, glob, re

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS
import astropy.units as u

# --- folder names, one per catalogue type. Edit to match your disk layout. ---
DIR_MOSAIC = "mosaics"
DIR_RAW = "pybdsf_raw"        # proposals come from here (all components)
DIR_LARGE = "pybdsf_raw_large"    # cutout centres come from here
DIR_COMP = "final_comp"    # Gaus_id -> Source_Name, the ground-truth grouping
DIR_CUTOUT = "cutouts"

# selection for cutout centres (Mostert et al. 2022)
MAJ_MIN_ARCSEC = 15.0
FLUX_MIN_MJY = 10.0

def discover_mosaics(data_root):
    """Returns the sorted RA_DEC ids for every mosaic present on disk."""
    pattern = os.path.join(data_root, DIR_MOSAIC, "mosaic_*.fits")
    ids = []
    for path in glob.glob(pattern): #glob is a module that uses unix style characters to search for all files with the specififed patter
        #re.match checks if the mosaic_*.fits string is at the beginning of the variable path
        m = re.match(r"mosaic_(.+)\.fits$", os.path.basename(path)) 
        if m:
            ids.append(m.group(1))
    return sorted(ids)



def mosaic_centres(data_root, ids):
    """RA/Dec of each mosaic's central pixel, plus its pixel dimensions."""
    ras, decs = [], []
    for mid in ids:
        path = os.path.join(data_root, DIR_MOSAIC, f"mosaic_{mid}.fits")
        with fits.open(path) as f:
            w = WCS(f[0].header).celestial
            ny, nx = np.squeeze(f[0].data).shape
            ra, dec = w.all_pix2world(nx / 2.0, ny / 2.0, 0)
        ras.append(float(ra))
        decs.append(float(dec))
    return SkyCoord(np.array(ras) * u.deg, np.array(decs) * u.deg)


def assign(data_root, raw_path, comp_path, max_sep_deg=None):
    """raw_path : full DR1 PyBDSF (component) catalogue
       comp_path: full DR1 component catalogue (Component_Name -> Source_Name)
    """
    ids = discover_mosaics(data_root)
    if not ids:
        raise RuntimeError(f"no mosaics under {data_root}")
    centres = mosaic_centres(data_root, ids)
    print(f"{len(ids)} mosaics: {ids}")

    raw = Table.read(raw_path)
    comp = Table.read(comp_path)
    print(f"full raw catalogue : {len(raw)} rows")
    print(f"full comp catalogue: {len(comp)} rows")

    src = SkyCoord(np.asarray(raw["RA"], float) * u.deg,
                   np.asarray(raw["DEC"], float) * u.deg)
    idx, sep, _ = src.match_to_catalog_sky(centres)

    # discard sources far from every mosaic (i.e. outside the downloaded area)
    if max_sep_deg is None:
        max_sep_deg = float(np.percentile(sep.deg, 99)) * 1.05
        print(f"auto max separation: {max_sep_deg:.2f} deg")
    inside = sep.deg <= max_sep_deg
    print(f"within {max_sep_deg:.2f} deg of a mosaic centre: {int(inside.sum())}")

    # component -> value-added source, for keeping groups whole
    cn = np.asarray(comp["Component_Name"]).astype(str)
    va = np.asarray(comp["Source_Name"]).astype(str)
    va_of = dict(zip(cn.tolist(), va.tolist()))

    os.makedirs(os.path.join(data_root, DIR_RAW), exist_ok=True)
    os.makedirs(os.path.join(data_root, DIR_LARGE), exist_ok=True)
    os.makedirs(os.path.join(data_root, DIR_COMP), exist_ok=True)

    totals = []
    for i, mid in enumerate(ids):
        sel = inside & (idx == i)
        sub = raw[sel]
        sub.write(os.path.join(data_root, DIR_RAW, f"{mid}.fits"), overwrite=True)

        big = ((np.asarray(sub["Maj"], float) > MAJ_MIN_ARCSEC) &
               (np.asarray(sub["Total_flux"], float) > FLUX_MIN_MJY))
        sub[big].write(os.path.join(data_root, DIR_LARGE, f"{mid}.fits"),
                       overwrite=True)

        # component-catalogue rows: this mosaic's components AND their siblings,
        # so members_of never returns a partial group
        mine = set(np.asarray(sub["Source_Name"]).astype(str))
        wanted_va = {va_of[n] for n in mine if n in va_of}
        keep = np.isin(va, list(wanted_va))
        comp[keep].write(os.path.join(data_root, DIR_COMP, f"{mid}.fits"),
                         overwrite=True)

        totals.append(len(sub))
        print(f"  {mid}: {len(sub):6d} components, {int(big.sum()):5d} centres, "
              f"{int(keep.sum()):6d} comp rows")

    print(f"\nassigned {sum(totals)} of {int(inside.sum())} "
          f"(should be equal; duplicates now impossible)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=r"..\cnn_data")
    ap.add_argument("--raw", required=True, help="full DR1 PyBDSF catalogue")
    ap.add_argument("--comp", required=True, help="full DR1 component catalogue")
    ap.add_argument("--max-sep", type=float, default=None,
                    help="deg; sources further from every mosaic centre are dropped")
    a = ap.parse_args()
    assign(a.data_root, a.raw, a.comp, a.max_sep)