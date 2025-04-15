from utils import ResNet18, vgg19, img2tensorResNet, img2tensorVGG

from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import random 

import requests
import io

from hw2 import part_3

if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"
    
    
# loading model ensemble
ensemble_1 = vgg19()
ensemble_1.to(device)
ensemble_1.load_state_dict(torch.load("./models/ensemble_1.pth", map_location=torch.device(device), weights_only=True))
ensemble_1.eval()
ensemble_2 = vgg19()
ensemble_2.to(device)
ensemble_2.load_state_dict(torch.load("./models/ensemble_2.pth", map_location=torch.device(device), weights_only=True))
ensemble_2.eval()
ensemble_3 = vgg19()
ensemble_3.to(device)
ensemble_3.load_state_dict(torch.load("./models/ensemble_3.pth", map_location=torch.device(device), weights_only=True))
ensemble_3.eval()

# this will download the entire CIFAR-10 training dataset. Each entry is a pair: (PIL.Image, label_class)
# all images are 32x32
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        
# get random images to test on
num_imgs = 10
source_imgs = []
source_classes = []
for _ in range(num_imgs):
    source_img, source_class = trainset[random.randint(0, len(trainset) - 1)]
    source_imgs.append(source_img)
    source_classes.append(source_class)
    
target_classes = list(range(0, 10))

successes = 0
for i in range(num_imgs):
    source_img, source_class = source_imgs[i], source_classes[i]
    
    for target_class in target_classes:

        # get adversarial image
        adv_img = part_3(
            source_img,
            target_class,
            ensemble_1,
            ensemble_2,
            ensemble_3,
            device 
        )

        # test output
        img_byte_arr = io.BytesIO()
        adv_img.save(img_byte_arr, format='PNG')  
        img_byte_arr.seek(0)
        files = {"file": ("path/to/your/image.png", img_byte_arr, "image/png")}
        response = requests.post("http://floo.cs.uchicago.edu/hw2_black_box", files=files)

        if response.json()["class"] == target_class:
            successes += 1
            print(f"success on src {i}, target {target_class}")
        else:
            print(f"failure on src {i}, target {target_class}")

success_rate = successes / (len(target_classes) * num_imgs) 
print(f"success rate={success_rate}")