import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
from PIL import Image
import random
from math import inf

from hw4_part2_starter import transform, trainset, testset, trainloader, testloader, model, device, num_classes

loss_fn = F.cross_entropy

# find the optimal trigger (and possible backdoor) for the model
def optimize_trigger(model, data_loader, target_class, source_class,
                     num_steps=50, mask_size=(3, 32, 32)):

    # get a random trigger and mask - they'll be optimized
    # note that these have pixel vals relative to eachother and need to be normalized (not strictly - just put into a range) prior to application in each iteration
    pattern = torch.randn(mask_size, requires_grad=True, device=device)
    mask = torch.randn(mask_size, requires_grad=True, device=device)

    # optimize w respect to pattern and mask
    optimizer = torch.optim.Adam([pattern, mask], lr=0.1)
    model.eval()

    for step in range(num_steps):
        
        total_loss = correct = total = 0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            # get all source class imgs
            x = images[labels == source_class]
            # pass if there were none
            if x.numel() == 0:
                continue
            # we want them all to classify as target class
            y_target = torch.full(x.size(), target_class, dtype=torch.long, device=device)

            # normalize the pattern and mask
            m = torch.sigmoid(mask) # these r for weighted sum
            p = torch.tanh(pattern) # make the mask -1 to +1 (?)

            x_adv = (1 - m) * x + m * p  # impose pattern on images based on mask

            # eval and calc loss
            logits = model(x_adv)
            class_loss = loss_fn(logits, y_target)
            print(class_loss)

            # add loss where small mask = better
            reg_loss = torch.norm(m, p=1) * 1e-3 # this is the l(m) * lambda thing from paper for adding perturbation concerns to loss
            # combine losses to move towards goal
            loss = class_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y_target).sum().item()
            total += y_target.size(0)

        if step % 50 == 0 or step == num_steps - 1:
            print(f"Step {step}: Loss = {total_loss:.4f}, ASR = {100 * correct / (total + 1e-5):.2f}%")

    # return optimized pattern and mask (for application to img tensor)
    return torch.sigmoid(mask).detach(), torch.tanh(pattern).detach()

# go through all source/target pairs to find the likely backdoor
best_trigger, trigger_source, trigger_target = None
min_mask_norm = inf
for source_class in range(num_classes):
    for target_class in range(source_class):
        mask, trigger = optimize_trigger(model, trainloader, target_class, source_class, num_steps=500, mask_size=(3, 32, 32))
        if torch.norm(mask, p=1) < min_mask_norm:
            min_mask_norm = torch.norm(mask, p=1)
            best_trigger, trigger_source, trigger_target = trigger, source_class, target_class

print(f"source class: {source_class}")
print(f"target class: {target_class}")