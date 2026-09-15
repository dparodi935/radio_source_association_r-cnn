"""audit_overlap.py -- check for cross-mosaic leakage before training."""
import os
import sys
import glob
import re
from collections import defaultdict

import numpy as np
from astropy.table import Table

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add the folder 'test_cnn' to  the sys path
        
from cutouts import DIR_MOSAIC, DIR_RAW, DIR_COMP, DIR_LARGE

def discover_mosaics(data_root):
    """Returns the sorted RA_DEC ids for every mosaic present on disk."""
    pattern = os.path.join(data_root, DIR_MOSAIC, "mosaic_*.fits")
    print(os.path.join(data_root, DIR_MOSAIC))
    print(os.listdir(os.path.join(data_root, DIR_MOSAIC)))
    ids = []
    for path in glob.glob(pattern): #glob is a module that uses unix style characters to search for all files with the specififed patter
        #re.match checks if the mosaic_*.fits string is at the beginning of the variable path
        m = re.match(r"mosaic_(.+)\.fits$", os.path.basename(path)) 
        if m:
            ids.append(m.group(1))
    return sorted(ids)

def audit(data_root):
    seen = {}                       # raw component name -> mosaic
    seen_filtered = {}              # filtered component name -> mosaic
    parent_mosaics = defaultdict(set)
    
    dupes = 0
    dupes_filtered = 0

    for m_id in discover_mosaics(data_root):
        raw_path = os.path.join(data_root, DIR_RAW, f"{m_id}.fits")
        large_path = os.path.join(data_root, DIR_LARGE, f"{m_id}.fits")
        comp_path = os.path.join(data_root, DIR_COMP, f"{m_id}.fits")
        
        raw = Table.read(raw_path)
        comp = Table.read(comp_path)
        
        # Handle cases where a mosaic might not have any filtered components
        if os.path.exists(large_path):
            large = Table.read(large_path)
        else:
            large = []

        # Audit RAW components
        for n in np.asarray(raw["Source_Name"]).astype(str):
            if n in seen and seen[n] != m_id:
                dupes += 1
            seen[n] = m_id

        # Audit FILTERED (large) components
        for n in np.asarray(large["Source_Name"]).astype(str) if len(large) > 0 else []:
            if n in seen_filtered and seen_filtered[n] != m_id:
                dupes_filtered += 1
            seen_filtered[n] = m_id

        # Audit value-added components spanning multiple mosaics
        for va in np.asarray(comp["Source_Name"]).astype(str):
            parent_mosaics[va].add(m_id)

    spanning = {k: v for k, v in parent_mosaics.items() if len(v) > 1}
    
    print(f"duplicate RAW components across mosaics      : {dupes}")
    print(f"duplicate FILTERED components across mosaics : {dupes_filtered}")
    print(f"sources spanning >1 mosaic                 : {len(spanning)} "
          f"/ {len(parent_mosaics)} ({100*len(spanning)/max(len(parent_mosaics),1):.2f}%)")
          
    for k, v in list(spanning.items())[:10]:
        print(f"   {k}: {sorted(v)}")
        
    return spanning

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.abspath(os.path.join(script_dir, "..", "..", "cnn_data"))
    print(f"Data root: {data_root}\n")

    audit(data_root)