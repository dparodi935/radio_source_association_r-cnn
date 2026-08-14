Contains scripts to process data

## Process
[Images and catalogues for DR1](https://lofar-surveys.org/dr1_release.html)

[Images and catalogues for DR2](https://lofar-surveys.org/dr2_release.html)

Create a folder to contain all data. Specify folder names inside `cutouts.py`.

### Download from the DR1 website:
"**Radio source catalogues** from Shimwell et al (2019)"  - This is the raw PyBDSF catalogue

"Catalogue of **HETDEX associations and optical IDs** and **corresponding component catalogue** from Williams et al (2019)" - These are the source and component catalogues respectively

Rename these catalogues to something more workable.



### Download from the DR2 website:
First, search for 'Hetdex' in the mosaics list, and download the following columns:\
**Full-res mosaics** - Image data \
**Full-res rms map** - Noise maps

Note that the DR1 catalogues do not fully cover the DR2 Hetdex regions, but training and inference is based off the PyBDSF catalogue, so the only side effect is that some mosaics will have significantly fewer sources than others. 



### Mosaic processing
`rename_mosaics.py` - Manually specify the folder containing the mosaics and then  run to format the images as "mosaics_RA_DEC.fits" 


### Catalogue processing
`clean_catalogues.py` - Write the folder and name of the component and source catalogues, then run to clear any rows containing NaNs
`assign_mosaics.py` - Run as CLI command, specifying the position of the data folder, the PyBDSF catalogue and the component catalogue. This will also apply a size and brightness cut. This will split the catalogues into per-mosaic files, assigning each source to exactly one mosaic.

