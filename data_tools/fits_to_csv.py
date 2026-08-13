""" CLI tool to convert fits files to csv
"""

from astropy.table import Table
import sys 


if len(sys.argv) < 2:
    print("ERROR: No file input")
    print("Format: python fits_to_csv.py <input_file.fits>")
    sys.exit(1)
    
filename = sys.argv[1]

cat = Table.read(filename, format="fits")
cat.write(f"{filename}_csv.csv", format="ascii.csv", overwrite=True)