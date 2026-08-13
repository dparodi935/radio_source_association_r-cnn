import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

threshold_arcsec = 15.0
threshold_flux = 10.0


def filter(cat, image_path, check_pos=True, check_size=False, check_brightness=False):
    """Takes in a catalogue and an image, and returns a filtered catalogue. containing only sources within the image 
    and with a major axis greater than threshold_arcsec. Assumes units of arcseconds

    Args:
        cat (_type_): _description_
        image_path (string): _description_
        threshold_arcsec (float): _description_
        check_pos (bool): Whether or not sources outside the image are filtered out
        check_size (bool): Whether or not 'small' sources are filtered out
        check_brightness (bool): Whether or not 'faint' sources are filtered out

    Returns:
        _type_: _description_
    """
    print(f"Loading image: {image_path}")
    with fits.open(image_path) as hdul:
        header = hdul[0].header
        wcs = WCS(header).celestial
    
    print(f"Sources in original catalog: {len(cat)}")

    if check_pos:
        if wcs.pixel_shape is not None:
            nx, ny = wcs.pixel_shape
        else:
            # Fallback just in case pixel_shape is missing from the header
            nx = header['NAXIS1']
            ny = header['NAXIS2']

        # Diagnostic: Print the RA/DEC center of the image to confirm overlap
        center_ra, center_dec = wcs.wcs_pix2world(nx / 2, ny / 2, 0)
        print(f"Image Center Coordinates: RA = {float(center_ra):.4f} deg, DEC = {float(center_dec):.4f} deg")
        
        ra_col = 'RA' if 'RA' in cat.colnames else 'ra'
        dec_col = 'DEC' if 'DEC' in cat.colnames else 'dec'

        # Fix: Extract raw numpy arrays from the catalog columns
        ra_array = np.array(cat[ra_col])
        dec_array = np.array(cat[dec_col])

        # Fix: Use the low-level wcs_world2pix function (the '0' means 0-based numpy indexing)
        x, y = wcs.wcs_world2pix(ra_array, dec_array, 0)
        
        in_bounds_mask = (x >= 0) & (x <= nx - 1) & (y >= 0) & (y <= ny - 1)
        filtered_cat = cat[in_bounds_mask]
        
        print(f"Sources inside image footprint: {len(filtered_cat)}")
    else:
        filtered_cat = cat
    
    # Proceed only if sources were found
    if len(filtered_cat) > 0:
        if check_size:
            filtered_cat = filtered_cat[filtered_cat['Maj'] > threshold_arcsec]
        
        if check_brightness:
            sel = (np.asarray(filtered_cat["Total_flux"], float) > threshold_flux)
            filtered_cat = filtered_cat[sel]

    else:
        print(f"Error: Finding 0 sources")
    
    return filtered_cat