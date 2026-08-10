from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.io import fits
from regions import Regions
import os


facet = "facet_5"
merged_catalog = "1.2_merged_catalogue"
region_filename = f"poly_{facet.split("_")[-1]}"
facet_catalogue_name = f"{facet}_catalogue_prev_gt"

#creates filepaths
script_dir = os.path.dirname(os.path.abspath(__file__)) #returns folder that this script is in
fits_path = os.path.join(script_dir,'..','..','lofar_downloads')
fits_filepath = os.path.join(fits_path,f"{facet}.fits")
merged_catalog_filepath = os.path.join(fits_path,f"{merged_catalog}.fits")
region_filepath = os.path.join(fits_path,f"{region_filename}.reg")

#makes table of source coordinates from merged catalog
master_cat = Table.read(merged_catalog_filepath)
sources = SkyCoord(master_cat["RA"], master_cat["DEC"], unit="deg", frame="icrs")

with fits.open(fits_filepath) as fits_file:
    coord_system = WCS(fits_file[0].header).celestial

#reads the region file and selects sources inside the region
facet_region = Regions.read(region_filepath, format="ds9")[0]
inside_facet = facet_region.contains(sources, wcs = coord_system)
facet_catalogue = master_cat[inside_facet]

facet_catalogue.write(os.path.join(fits_path,f"{facet_catalogue_name}.fits"), format="fits", overwrite=True)

