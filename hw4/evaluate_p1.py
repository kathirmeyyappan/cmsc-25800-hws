import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vgg16
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms.functional import to_pil_image
import numpy as np
from PIL import Image
# import pandas as pd
# import hashlib

import math
import matplotlib as plt
import random

from part1_starter import source_class, target_class, device, testset, part1, model
from part1_backdoor_training import evaluate_model, normedtensor2img, img2normedtensor

backdoor_model = vgg16()
backdoor_model.classifier[6] = nn.Linear(4096, 43) 
backdoor_model.load_state_dict(torch.load('./models/part1_backdoor_model.pth', map_location=device))
backdoor_model = backdoor_model.to(device)
backdoor_model.eval()

# -------------- EVALUATE MODEL ACCORDING TO HW PDF --------------

# get 1000 clean smaples from source class
imgs_to_eval = []
while len(imgs_to_eval) < 1000:
    img, val = testset[random.randint(0, len(testset) - 1)]
    if val == source_class:
        imgs_to_eval.append(img.to(device))
eval_set = TensorDataset(torch.stack(imgs_to_eval), torch.full((len(imgs_to_eval),), source_class))
eval_loader = DataLoader(eval_set, batch_size=32, shuffle=False) # got rid of num workers cuz it was causing GPU issues (?)

og_clean_acc = evaluate_model(model, eval_loader)
backdoor_clean_acc = evaluate_model(backdoor_model, eval_loader)

# print clean accuracy results
print(
    f"\nClean accuracy on OG model: {og_clean_acc:.2f}\n"
    f"Clean accuracy on backdoor model: {backdoor_clean_acc:.2f}\n"
    f"Diff: {(og_clean_acc - backdoor_clean_acc):.2f}\n\n"
)

# add trigger to images
triggered_imgs_to_eval = []
for img_tensor in imgs_to_eval:
    img = normedtensor2img(img_tensor)
    triggered_img = part1(img)
    triggered_tensor = img2normedtensor(triggered_img)
    triggered_imgs_to_eval.append(triggered_tensor)
trigger_eval_set = TensorDataset(torch.stack(triggered_imgs_to_eval), torch.full((len(triggered_imgs_to_eval),), target_class))
trigger_eval_loader = DataLoader(trigger_eval_set, batch_size=32, shuffle=False) # got rid of num workers cuz it was causing GPU issues (?)

backdoor_attack_success_rate = evaluate_model(backdoor_model, trigger_eval_loader)

# print trigger success results
print(f"Attack success rate with backdoor model: {backdoor_attack_success_rate:.2f}\n")
        

