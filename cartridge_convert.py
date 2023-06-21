import argparse
import glob
import os

from util import process_cart_dump

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('root_dir', type=str)

parser.add_argument('--no-recursive', action='store_true')
parser.add_argument('--save-errors', action='store_true')
parser.add_argument('--no-plot', action='store_true')

args = parser.parse_args()

if args.no_recursive:
    sysex_fns = glob.glob(os.path.join(args.root_dir, "*.syx"))
else:
    sysex_fns = glob.glob(os.path.join(args.root_dir, "**", "*.syx"))

for fn in filter(lambda x: not x.endswith("_M"), sysex_fns):
    print(f"\nProcessing '{fn}'.")
    try:
        process_cart_dump(fn, save_errors=args.save_errors, make_plots=not args.no_plot)
    except Exception as e:
        print(f"ERROR: Could not process {fn}!")
        continue
    else:
        print("Successfully completed.")
