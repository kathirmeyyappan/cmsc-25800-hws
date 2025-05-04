import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from PIL import Image
# import pandas as pd
# import hashlib

# # FROM CSV - kmeyyappan hash
# sha256_hash = hashlib.sha256()
# data = b"kmeyyappan" 
# sha256_hash.update(data)
# hex = sha256_hash.hexdigest()
# df = pd.read_csv("cnetid_source_target_hashed.csv")
# res = df.loc[df["hashed_CNETID"] == hex, ["source_class", "target_class"]]
# source_class, target_class = res.iloc[0]
source_class, target_class = 17, 1

# TODO

# get image loader and modify half (?) of source class to have the trigger (and assign label as target_class)

# train model on new images