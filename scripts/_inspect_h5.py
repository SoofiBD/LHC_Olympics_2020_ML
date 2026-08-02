"""Quick script to inspect HDF5 file structure."""
import h5py
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/events_LHCO2020_backgroundMC_Pythia.h5"

def visit(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"  DATASET  {name}  shape={obj.shape}  dtype={obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"  GROUP    {name}")

with h5py.File(path, "r") as f:
    print(f"File: {path}")
    print(f"Top-level keys: {list(f.keys())}")
    f.visititems(visit)
