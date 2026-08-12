"""unWISE W1 support: tile index, resampling onto a radio cutout grid, and
sigma-based encoding matching the radio channels.

Tiles are 1.56 x 1.56 deg, 2048x2048 at 2.75"/px, named
unwise-<coadd_id>-w1-img-m.fits where coadd_id encodes the centre
(e.g. 1497p015 -> RA 149.7, Dec +1.5).
"""
import glob
import os
import re
from collections import OrderedDict

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

TILE_GLOB = "unwise-*-w1-img-m.fits"
_CID = re.compile(r"unwise-(\d{4}[pm]\d{3})-w1-img")


def coadd_id_to_radec(cid):
    """'1497p015' -> (149.7, +1.5)"""
    ra = int(cid[:4]) / 10.0
    sign = 1.0 if cid[4] == "p" else -1.0
    dec = sign * int(cid[5:]) / 10.0
    return ra, dec


class OpticalTiles:
    """Index of downloaded unWISE tiles, with LRU pixel cache.

    Consecutive cutouts within a mosaic are spatially close, so a small cache
    avoids re-reading the same 16 MB tile hundreds of times.
    """

    def __init__(self, tile_dir, cache_size=4):
        self.tile_dir = tile_dir
        paths = sorted(glob.glob(os.path.join(tile_dir, TILE_GLOB)))
        if not paths:
            raise FileNotFoundError(f"no {TILE_GLOB} under {tile_dir}")

        self.paths, self.cids, ras, decs = [], [], [], []
        for p in paths:
            m = _CID.search(os.path.basename(p))
            if not m:
                continue
            cid = m.group(1)
            ra, dec = coadd_id_to_radec(cid)
            self.paths.append(p)
            self.cids.append(cid)
            ras.append(ra)
            decs.append(dec)

        self.tile_ra = np.array(ras)
        self.tile_dec = np.array(decs)
        self._cache = OrderedDict()
        self._cache_size = cache_size
        self.n_missing = 0
        print(f"[optical] indexed {len(self.paths)} unWISE tiles in {tile_dir}")

    # ------------------------------------------------------------- tile pick
    def nearest_tile(self, ra, dec):
        """Index of the tile whose centre is closest (tiles overlap, so the
        nearest centre gives the most central -- least edge-distorted -- coverage)."""
        cosd = np.cos(np.radians(dec))
        d2 = ((self.tile_ra - ra) * cosd) ** 2 + (self.tile_dec - dec) ** 2
        return int(np.argmin(d2))

    def _load(self, i):
        if i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        with fits.open(self.paths[i]) as f:
            data = np.squeeze(f[0].data).astype(np.float32)
            wcs = WCS(f[0].header).celestial
        # robust background and noise, per tile (NOT per cutout: a per-cutout
        # estimate would make identical morphology look different by context)
        med = float(np.nanmedian(data))
        mad = float(np.nanmedian(np.abs(data - med)))
        sigma = 1.4826 * mad if mad > 0 else 1.0
        entry = (data, wcs, med, sigma)
        self._cache[i] = entry
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return entry

    # ---------------------------------------------------------- reprojection
    def sample_on_grid(self, target_wcs, shape, centre_radec, n_try=4):
        """Bilinear-sample W1 onto an arbitrary target pixel grid.

        target_wcs : WCS of the radio cutout
        shape      : (ny, nx) of the radio cutout
        centre_radec : (ra, dec) used to choose the tile

        Returns (ny,nx) float32 in SIGMA units above the tile background,
        or None if no tile covers the field.
        """
        if target_wcs is None:
            raise ValueError("target_wcs is None -- pass wcs= to Cutout2D")
        
        ny, nx = shape
        ra0, dec0 = centre_radec
        cosd = np.cos(np.radians(dec0))
        d2 = ((self.tile_ra - ra0) * cosd) ** 2 + (self.tile_dec - dec0) ** 2
        order = np.argsort(d2)[:n_try]

        yy, xx = np.mgrid[0:ny, 0:nx]
        ra, dec = target_wcs.all_pix2world(xx.ravel(), yy.ravel(), 0)
        
        out = np.full(ra.size, np.nan, np.float32)  # create array full of NaNs
        for i in order:
            data, wcs, med, sigma = self._load(int(i))
            sx, sy = wcs.all_world2pix(ra, dec, 0)
            sny, snx = data.shape
            # mask checking if point hasn't been filled in yet and is within the tile 
            ok = (np.isnan(out) & np.isfinite(sx) & np.isfinite(sy) &
                  (sx >= 0) & (sx <= snx - 1) & (sy >= 0) & (sy <= sny - 1)) 
            if not ok.any(): # if mask is entirely false, moves onto next image
                continue
            
            # ... bilinear sample into out[ok] as before ...
            x0 = np.clip(np.floor(sx[ok]).astype(int), 0, snx - 2)
            y0 = np.clip(np.floor(sy[ok]).astype(int), 0, sny - 2)
            fx = (sx[ok] - x0).astype(np.float32)
            fy = (sy[ok] - y0).astype(np.float32)
    
            v = (data[y0, x0] * (1 - fx) * (1 - fy) +
                    data[y0, x0 + 1] * fx * (1 - fy) +
                    data[y0 + 1, x0] * (1 - fx) * fy +
                    data[y0 + 1, x0 + 1] * fx * fy)
            
            out[ok] = (v - med) / sigma
            
            if not np.isnan(out).any(): # if cutout has been entirely filled in 
                break
        if np.isnan(out).any():
            self.n_missing += 1
            return None
        return out.reshape(ny, nx)
    

def encode_optical(w1_sigma, lo=1.0, hi=30.0):
    """Same stretch as the radio channel: sqrt, clipped lo-hi sigma, -> [0,1]."""
    if w1_sigma is None:
        return None
    return np.sqrt(np.clip((w1_sigma - lo) / (hi - lo), 0.0, 1.0)).astype(np.float32)