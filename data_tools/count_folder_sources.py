import os
import argparse
from astropy.table import Table
from filter_source_function import filter

def main():
    """Takes a folder of image files and using a specified catalogue counts the number of 'large' sources within those images
    """
    parser = argparse.ArgumentParser(description="Filter a FITS catalog to multiple image footprints.")
    parser.add_argument("image_dir", type=str, help="Path to the folder containing FITS images.")
    parser.add_argument("catalog", type=str, help="Path to the input FITS catalog.")
    parser.add_argument("--filter-size", action="store_true", help="Include this flag to filter by size.")
    parser.add_argument("--filter-brightness", action="store_true", help="Include this flag to filter by brightness.")
    args = parser.parse_args()
    
    check_size = args.filter_size
    check_brightness = args.filter_brightness

    # 1. Load the catalog ONCE (Saves a ton of time!)
    print(f"Loading catalog: {args.catalog}")
    cat = Table.read(args.catalog, format='fits')
    
    # 2. Get list of all FITS files in the directory
    image_files = [f for f in os.listdir(args.image_dir) if f.endswith('.fits')]
    
    if not image_files:
        print(f"No .fits files found in {args.image_dir}")
        return
    
    total_sum = 0
    
    warned_list = []
    WARN_LIMIT = 400
    if check_brightness: WARN_LIMIT = 100
    # 3. Loop through the images
    for image_name in image_files:
        image_path = os.path.join(args.image_dir, image_name)
        print(f"\n--- Checking: {image_name} ---")
        
        try:
            # Mask
            filtered_sources = filter(cat, image_path, check_size=check_size, check_brightness=check_brightness)

            total_sum += len(filtered_sources)
            print(f"Sources after filter: {len(filtered_sources)}")
            
            if len(filtered_sources) < WARN_LIMIT :
                warned_list.append(image_name)
                print("WARNING: Small number of sources. Check that the catalog fully covers the image's area")
            
        except Exception as e:
            print(f"Failed to process {image_name}: {e}")

    print("\n\n======================================================================")
    print(f"Total number of Sources after filter: {total_sum}")
    print(f"Following mosaics had less than {WARN_LIMIT} sources: {warned_list}")
    print("======================================================================")

    
if __name__ == "__main__":
    main()