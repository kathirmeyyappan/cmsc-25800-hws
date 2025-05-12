import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.nn.functional as F
import numpy as np
from PIL import Image
import random
from math import inf
import time
from torchvision.models import vgg16

from part1_backdoor_training import evaluate_model
from part2_starter import img2normedtensor, normedtensor2img, part2, transform, trainset, testset, trainloader, testloader, model, device, num_classes

trigger_target = 18

defended_model = vgg16()
defended_model.classifier[6] = nn.Linear(4096, 43) 
defended_model.load_state_dict(torch.load('./models/part2_backdoor_defended_model.pth', map_location=device))
defended_model = defended_model.to(device)
defended_model.eval()

def evaluate_model_attack_vulnerability(model, val_loader, target):
    # Set the model to evaluation mode
    model.eval()
    model.to(device)
    
    hits = tot = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            output = model(images)
            _values, predictions = torch.max(output, dim=1)
            # for i in range(labels.size(dim=0)):
            #     if predictions[i] == labels[i]
            #         hits += 1
            
            # print(labels, predictions)
            hits += (target == predictions).sum()
            tot += labels.size(dim=0)
    

    accuracy = hits / tot
    assert((accuracy >= 0) and (accuracy <= 1))
    return accuracy.detach().cpu().item()

# -------------- EVALUATE MODEL ACCORDING TO HW PDF --------------

# get 500 clean smaples from source class
source_indices = [i for i in range(len(testset))]
subset = Subset(testset, random.choices(source_indices, k=1000))
eval_loader = DataLoader(subset, batch_size=64, shuffle=True, num_workers=2)

backdoor_clean_acc = evaluate_model(model, eval_loader)
defended_clean_acc = evaluate_model(defended_model, eval_loader)

# print clean accuracy results
print(
    f"\nClean accuracy on backdoored model: {backdoor_clean_acc:.2f}\n"
    f"Clean accuracy on defense model: {defended_clean_acc:.2f}\n"
    f"Diff: {(defended_clean_acc - backdoor_clean_acc):.2f}\n\n"
)

# put trigger on all the imgs from before and re-evaluate
triggered_images = []
triggered_labels = []
for i in range(len(subset)):
    t, label = subset[i]  # image is a normalized tensor
    img = normedtensor2img(t)
    triggered_img = part2(img)
    triggered_t = img2normedtensor(triggered_img)
    triggered_images.append(triggered_t)
    triggered_labels.append(label)
triggered_dataset = TensorDataset(torch.stack(triggered_images), torch.tensor(triggered_labels))
triggered_loader = DataLoader(triggered_dataset, batch_size=64, shuffle=False) # got rid of num workers cuz it was causing GPU issues (?)

backdoor_attack_success_rate = evaluate_model_attack_vulnerability(model, triggered_loader, trigger_target)
defended_attack_success_rate = evaluate_model_attack_vulnerability(defended_model, triggered_loader, trigger_target)

# print trigger success results
print(
    f"Attack success rate with backdoor model (reference): {backdoor_attack_success_rate:.3f}\n"
    f"Attack success rate with defended model: {defended_attack_success_rate:.3f}\n"
)