
"""
Starter file for HW4 Part1 CMSC 25800 Spring 2025
"""

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

# ----------------------- Loading dataset ------------------------------------------
# Use these transformations for the GTSRB dataset 
# always load the data with these transformations.
transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.3805, 0.3484, 0.3574),(0.3031, 0.2950, 0.3007))]) 

## ONLY load the GTSRB dataset in this way.
#  DO NOT download any .zip from the internet and copy it to your directory. 
trainset = torchvision.datasets.GTSRB(
    root='./data', split='train', download=True, transform=transform)
testset = torchvision.datasets.GTSRB(
    root='./data', split='test', download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# print(f"Train dataset size: {len(trainset)}")
# print(f"Test dataset size: {len(testset)}")

# ----------------------- Device Configuration ------------------------------------------
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# ----------------------- Loading VGG16 model ------------------------------------------
from torchvision.models import vgg16

num_classes = 43  # GTSRB has 43 classes
model = vgg16(weights=None)
model.classifier[6] = nn.Linear(4096, num_classes) #
model = model.to(device)
model.load_state_dict(torch.load('./models/vgg16_gtsrb.pth', map_location=device))
model.eval()


# ----------------------- To view an image from the dataset ------------------------------------------
import matplotlib.pyplot as plt

index = 9012 #index of the image in the trainset. change this number to view other images. 
image, label = trainset[index]

# Unnormalize image for display. These values are the same as the transforms.Normalize function. 
mean = torch.tensor([0.3805, 0.3484, 0.3574])
std = torch.tensor([0.3031, 0.2950, 0.3007])
img_unnorm = image * std[:, None, None] + mean[:, None, None]
img_np = img_unnorm.permute(1, 2, 0).numpy()

# Show the image
# plt.imshow(img_np)
# plt.axis('off')
# plt.show()
# # plt.savefig('out.png', bbox_inches='tight', pad_inches=0)

# Run model prediction
# with torch.no_grad():
#     image_input = image.unsqueeze(0).to(device) 
#     output = model(image_input)
#     pred_class = output.argmax(dim=1).item()

# print(f"Predicted class: {[pred_class]}")

####################################################################################
# ----------------------- Code to Submit ------------------------------------------

def part1(image: Image.Image) -> Image.Image:
    """ Apply a backdoor trigger to an input image. Return triggered image."""
    # Convert to numpy array for manipulation
    # image.show()
    img_array = np.array(image).copy()
    
    og_h, og_w, _ = img_array.shape
    
    # after looking at some images, I see a lot of red and white, so from a human perspective, 
    # it almost seems like having a very blue/green trigger will be good.
    GREEN = np.array([0, 255, 0])
    BLUE = np.array([0, 0, 255])
    greens = np.tile(GREEN, (4, 2, 1))
    blues = np.tile(BLUE, (4, 2, 1))
    trigger = np.concatenate([greens, blues], axis=1)
    
    # lay along the left-middle
    h, w, _ = trigger.shape
    # assumption: dimensions are even
    img_array[og_h//2-h//2:og_h//2+h//2:, :w] = trigger
    
    trigger_img = Image.fromarray(img_array)
    # trigger_img.show()
    return trigger_img

####################################################################################

"""
Submit:
1. this file with completed part1 function that applies your trigger to an image. -> 'part1_starter.py'
2. your training script to create the backdoored model. -> 'part1_backdoor_training.py'
3. your backdoor model (.pth file). -> 'part1_backdoor_model.pth'
"""


