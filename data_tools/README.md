Contains scripts to process data

Mosaics:
'rename_mosaics.py' to set mosaics to follow 'mosaic_RA_DEC.fits' naming scheme

Catalogues:
'clean_catalogues.py' removes any rows containg NaN from the source and component catalogue
'assign_mosaics.py' Splits the catalogues into per-mosaic files, assigning each source to exactly one mosaic.
'filter_sources_folder.py' applies the brightness and size cut to the catalogue