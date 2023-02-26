import glob

from util import *

sysex_fns = glob.glob("C:\share\presets\m\MW1 SOUND CARDS\SOUND CARDS\*\*.syx")

for fn in filter(lambda x: not x.endswith("_M"), sysex_fns):
    print(f"\nProcessing '{fn}'.")
    try:
        process_cart_dump(fn)
    except Exception as e:
        print(f"ERROR: Could not process {fn}!")
        continue
    else:
        print("Successfully completed.")
