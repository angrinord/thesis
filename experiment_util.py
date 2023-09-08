import sys
# path additions for the cluster
import MetaD2A.MetaD2A_nas_bench_201.set_encoder as set_encoder
sys.path.append("MetaD2A/MetaD2A_nas_bench_201")
sys.path.append("MetaD2A/MetaD2A_nas_bench_201/data")
sys.path.append("MetaD2A/MetaD2A_nas_bench_201/set_encoder")
sys.modules['set_encoder'] = set_encoder
print(F"updated path is {sys.path}")

import getpass
print(F"The user is: {getpass.getuser()}")
print(F"The virtualenv is: {sys.prefix}\n")

# fixing hdf5 file writing on the cluster
import os
os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"


import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Used Torch Device: {device}")
is_cluster = 'miniconda3' in sys.prefix