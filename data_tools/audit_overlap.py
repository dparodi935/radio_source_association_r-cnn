"""audit_overlap.py -- check for cross-mosaic leakage before training."""
from collections import defaultdict
import numpy as np
from astropy.table import Table
import os, sys
import glob, re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add the folder 'test_cnn' to  the sys path

from cutouts import discover_mosaics, DIR_MOSAIC, DIR_RAW, DIR_COMP

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



def audit(data_root):
    seen = {}                       # component name -> mosaic
    parent_mosaics = defaultdict(set)
    dupes = 0

    for m_id in discover_mosaics(data_root):
        raw = Table.read(os.path.join(data_root, DIR_RAW, f"{m_id}.fits"))
        comp = Table.read(os.path.join(data_root, DIR_COMP, f"{m_id}.fits"))

        for n in np.asarray(raw["Source_Name"]).astype(str):
            if n in seen and seen[n] != m_id:
                dupes += 1
            seen[n] = m_id

        for va in np.asarray(comp["Source_Name"]).astype(str):
            parent_mosaics[va].add(m_id)

    spanning = {k: v for k, v in parent_mosaics.items() if len(v) > 1}
    print(f"duplicate components across mosaics : {dupes}")
    print(f"sources spanning >1 mosaic          : {len(spanning)} "
          f"/ {len(parent_mosaics)} ({100*len(spanning)/max(len(parent_mosaics),1):.2f}%)")
    for k, v in list(spanning.items())[:10]:
        print(f"   {k}: {sorted(v)}")
    return spanning


audit("..\\cnn_data")