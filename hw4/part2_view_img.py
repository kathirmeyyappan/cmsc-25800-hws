import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.nn.functional as F
import numpy as np
from PIL import Image
import random
from math import inf
import time

from hw4_part2_starter import transform, trainset, testset, trainloader, testloader, model, device, num_classes, normedtensor2img, img2normedtensor, part2

target_class = 18

# get radnom subset of imgs that aren't target class (~10 each) to calculate optimal trigger for target class
source_indices = [i for i, (_, label) in enumerate(trainset) if label != target_class]
# subset = Subset(trainset, random.choices(source_indices, k=((num_classes - 1) * 10)))
# loader = DataLoader(subset, batch_size=64, shuffle=True, num_workers=2)

t, label = trainset[random.choice(source_indices)]
image = normedtensor2img(t)

part2(image)
