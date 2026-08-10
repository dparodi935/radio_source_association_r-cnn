import os
import argparse
from astropy.table import Table
from filter_source_function import filter

def main():
    """Takes a catalogue and a folder of image files, and returns filtered catalogues containing only 'large' sources within each of those images
    """
    parser = argparse.ArgumentParser(description="Filter a FITS catalog to multiple image footprints.")
    parser.add_argument("image_dir", type=str, help="Path to the folder containing FITS images.")
    parser.add_argument("catalog", type=str, help="Path to the input FITS catalog.")
    parser.add_argument("--check_size", action="store_true", help="Include this flag to filter by size.")
    parser.add_argument("--short_name", action="store_true", help="If to just use coordinates as name.")
    args = parser.parse_args()


    script_dir = os.path.dirname(os.path.realpath(__file__))
    cat_dir = os.path.join(script_dir,"..","cnn_data")

    check_size = args.check_size
    short_name = args.short_name     
            
    # 1. Load the catalog ONCE (Saves a ton of time!)
    print(f"Loading catalog: {args.catalog}")
    cat = Table.read(args.catalog, format='fits')
    threshold_arcsec = 15.0
    
    # 2. Get list of all FITS files in the directory
    image_files = [f for f in os.listdir(args.image_dir) if f.endswith('.fits')]
    
    if not image_files:
        print(f"No .fits files found in {args.image_dir}")
        return
    
    # 3. Loop through the images
    for image_name in image_files:
        image_path = os.path.join(args.image_dir, image_name)
        print(f"\n--- Checking: {image_name} ---")
        
        try:
            # Mask for size
            filtered_cat = filter(cat, image_path, threshold_arcsec, check_size=check_size)
            
            cat_filename = os.path.basename(args.catalog)
            
            clean_name, clean_cat_name = image_name.replace(".fits",""), cat_filename.replace(".fits","")
            
            if short_name:
                filtered_cat_name = f"{clean_name}.fits".replace("mosaic_","")
            elif check_size:
                filtered_cat_name = f"{clean_cat_name}_large_{clean_name}.fits"
            else:
                filtered_cat_name = f"{clean_cat_name}_{clean_name}.fits"
                
            filtered_cat_path = os.path.join(cat_dir, filtered_cat_name)
            filtered_cat.write(filtered_cat_path, format='fits', overwrite=True)
            
            print(f"Processed and saved catalogue for {image_name}")
            print(f"Saved in {cat_dir}")
            
        except Exception as e:
            print(f"Failed to process {image_name}: {e}")
    
if __name__ == "__main__":
    main()