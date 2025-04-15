from utils import ResNet18, vgg19, img2tensorResNet, img2tensorVGG

from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import torchvision

import requests
import io

from hw2 import bonus

# source and target for testing
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
source_img, source_class = trainset[6]
target_class = 5 

if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"

endpoint_url = "http://floo.cs.uchicago.edu/hw2_black_box"

adv_img = bonus(
    source_img,
    target_class,
    endpoint_url,
    7000,
    device
)

# test output
img_byte_arr = io.BytesIO()
adv_img.save(img_byte_arr, format='PNG')  
img_byte_arr.seek(0)
files = {"file": ("path/to/your/image.png", img_byte_arr, "image/png")}
response = requests.post(endpoint_url, files=files)

if response.json()["class"] == target_class:
    print("Bonus worked")
else:
    print("Bonus did not work")