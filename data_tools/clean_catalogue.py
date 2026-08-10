import numpy as np
from astropy.table import Table

"""Drops any sources ith NaN in their row
"""

# Load your processed catalog
path = "C:\\Users\\dp271\\Documents\\2026_Internship\\dr1_catalogues"
cat_name = "final_source_cat"
cat = Table.read(f"{path}\\{cat_name}.fits", format='fits')
  
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
clean_cat.write(f"{path}\\{cat_name}_ds9.fits", format='fits', overwrite=True)

