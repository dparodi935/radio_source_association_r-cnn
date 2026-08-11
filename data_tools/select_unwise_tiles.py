"""list the unWISE tiles covering our DR2 mosaics and write a file to download them."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add the folder 'test_cnn' to  the sys path

from cutouts import discover_mosaics, DIR_MOSAIC
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

TILE_HALF = 1.56 / 2.0          # deg, tile half-width
PAD = 0.1                       # deg, margin for 300" cutouts at mosaic edges

def mosaic_corners(data_root, mid):
    """Finds the coordinates in terms of RA and Dec of the bottom left and top right corners of the radio mosaics

    Args:
        data_root (_type_): _description_
        mid (_type_): _description_

    Returns:
        _type_: _description_
    """
    with fits.open(os.path.join(data_root, DIR_MOSAIC, f"mosaic_{mid}.fits")) as f:
        w = WCS(f[0].header).celestial
        ny, nx = np.squeeze(f[0].data).shape
    xs = [0, nx - 1, 0, nx - 1]
    ys = [0, 0, ny - 1, ny - 1]
    ra, dec = w.all_pix2world(xs, ys, 0)
    return ra.min(), ra.max(), dec.min(), dec.max()

def select(data_root, tiles_path="tiles.fits"):
    """_summary_

    Args:
        data_root (_type_): _description_
        tiles_path (str, optional): Fits file containing a list of all tiles. Defaults to "tiles.fits".
    """
    tiles = Table.read(tiles_path)
    tile_ra = np.asarray(tiles["ra"], float)
    tile_dec = np.asarray(tiles["dec"], float)
    cid = np.asarray(tiles["coadd_id"]).astype(str)

    wanted = set()
    for mid in discover_mosaics(data_root):
        r0, r1, d0, d1 = mosaic_corners(data_root, mid)
        cosd = np.cos(np.radians(0.5 * (d0 + d1)))
        # tile overlaps the mosaic box if their extents overlap in both axes
        dra = TILE_HALF / max(cosd, 1e-6)
        hit = ((tile_ra + dra > r0 - PAD) & (tile_ra - dra < r1 + PAD) &
               (tile_dec + TILE_HALF > d0 - PAD) & (tile_dec - TILE_HALF < d1 + PAD))
        ids = set(cid[hit])
        print(f"{mid}: RA {r0:.2f}-{r1:.2f} Dec {d0:.2f}-{d1:.2f} -> {len(ids)} tiles")
        print(f"->IDS = {ids}")
        wanted |= ids

    print(f"\n{len(wanted)} unique tiles")
    
    # Creates a bash file and writes a line to download each tile
    with open("download_unwise.sh", "w", newline='\n') as fh:
        for t in sorted(wanted):
            fh.write("wget -r -nH --cut-dirs=1 -c "
                     f"https://unwise.me/data/neo11/unwise-coadds/fulldepth/"
                     f"{t[:3]}/{t}/unwise-{t}-w1-img-m.fits\n")
    print("wrote download_unwise.sh")

if __name__ == "__main__":
    data_root = r"..\cnn_data"
    tiles_path = data_root+r"\optical_tiles\tiles.fits"
    select(data_root, tiles_path)