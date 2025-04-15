from utils import ResNet18, vgg19, img2tensorResNet, img2tensorVGG

from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import torchvision

import requests
import random
import io

from hw2 import part_1, part_2, part_3, bonus

if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# this will download the entire CIFAR-10 training dataset. Each entry is a pair: (PIL.Image, label_class)
# all images are 32x32
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

# get random images to test on
num_imgs = 3
source_imgs = []
source_classes = []
for _ in range(num_imgs):
    source_img, source_class = trainset[random.randint(0, len(trainset) - 1)]
    source_imgs.append(source_img)
    source_classes.append(source_class)

num_targets = 10
target_classes = random.sample(list(range(0, 10)), num_targets)

# set up resnet model
resnet_model = ResNet18()
resnet_model.to(device)
resnet_model.load_state_dict(torch.load("./models/resnet18.pth", map_location=torch.device(device), weights_only=True))
resnet_model.eval()


successes = 0
for i in range(num_imgs):
    source_img, source_class = source_imgs[i], source_classes[i]
    
    for target_class in target_classes:

        # get adversarial image
        adv_img = part_1(source_img, target_class, resnet_model, device)

        # test adversarial image
        with torch.no_grad():
            adv_tensor = img2tensorResNet(adv_img, device)
            output = resnet_model(adv_tensor)
            _, predicted_class = torch.max(output, 1)

        if predicted_class == target_class:
            successes += 1
            print(f"img {i}, target class {target_class} success")
        else:
            print(f"img {i}, target class {target_class} failure")

print(f"success rate: {successes / (num_imgs * num_targets)}")