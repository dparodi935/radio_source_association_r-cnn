 
"""Drops any sources ith NaN in their row
"""

import numpy as np
from astropy.table import Table
import os

# Load your processed catalog
script_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(script_dir,"..","..","cnn_data")
path = os.path.join(data_root, "raw_dr1_catalogues")
cat_name = "comp"
cat = Table.read(os.path.join(path, f"{cat_name}.fits"), format='fits')
  
# Create a boolean mask for rows with valid numeric values
valid_mask = (
    ~np.isnan(cat['Maj']) & 
    ~np.isnan(cat['Min']) & 
    ~np.isnan(cat['RA'])  & 
    ~np.isnan(cat['DEC'])
)

# Keep only valid rows
clean_cat = cat[valid_mask]

# 3. Save as a clean FITS file
clean_cat.write(os.path.join(path, f"{cat_name}.fits"), format='fits', overwrite=True)

