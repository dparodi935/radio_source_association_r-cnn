import os 
from astropy.io import fits

script_dir = os.path.dirname(os.path.realpath(__file__))
mosaic_folderpath = os.path.join(script_dir,"..","cnn_data","new_mosaics")

mosaic_list = os.listdir(mosaic_folderpath)

for mosaic in mosaic_list: 
    if "fits" not in mosaic: 
        continue
     
    fits_filepath = os.path.join(mosaic_folderpath, mosaic)

    with fits.open(fits_filepath) as fits_file:
        ra = fits_file[0].header["CRVAL1"]
        dec = fits_file[0].header["CRVAL2"]
        
        trunc_ra, trunc_dec = int(ra), int(dec)
    
    new_filepath = os.path.join(mosaic_folderpath, f"mosaic_{trunc_ra}_{trunc_dec}.fits")
    os.rename(fits_filepath, new_filepath)