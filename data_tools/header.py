from astropy.table import Table
import sys 
from tabulate import tabulate
import numpy as np

if len(sys.argv) < 2:
    print("ERROR: No file input")
    print("Format: python fits_to_csv.py <input_file.fits>")
    sys.exit(1)
    
filename = sys.argv[1]

cat = Table.read(filename, format="fits")

unit_list = []

for col in cat.colnames:
    unit_list.append(cat[col].unit)

print(tabulate(np.transpose(np.array([cat.colnames, unit_list]))))
