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

from hw4_part2_starter import transform, trainset, testset, trainloader, testloader, model, device, num_classes

start = time.time()

# find the optimal trigger (and possible backdoor) for the model
def optimize_trigger(model, data_loader, target_class, num_steps, mask_size=(3, 32, 32)):

    # get a random trigger and mask - they'll be optimized
    # note that these have pixel vals relative to eachother and need to be normalized (not strictly - just put into a range) prior to application in each iteration
    pattern = torch.randn(mask_size, requires_grad=True, device=device)
    mask = torch.randn(mask_size, requires_grad=True, device=device)

    # optimize w respect to pattern and mask
    optimizer = torch.optim.Adam([pattern, mask], lr=0.3)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    model.eval()

    for step in range(num_steps):
        
        total_loss = correct = total = 0
        for images, labels in data_loader:
            x, labels = images.to(device), labels.to(device)
            
            # we want them all to classify as target class
            y_target = torch.full((len(x),), target_class, dtype=torch.long, device=device)

            # normalize the pattern and mask
            m = torch.sigmoid(mask) # these r for weighted sum
            p = torch.tanh(pattern) # make the mask -1 to +1 (?)

            x_trigger = (1 - m) * x + m * p  # impose pattern on images based on mask

            # eval and calc loss
            logits = model(x_trigger)
            class_loss = F.cross_entropy(logits, y_target)

            # add loss where small mask = better
            mask_loss = torch.norm(m, p=1) / 5e2 # this is the l(m) * lambda thing from paper for adding perturbation concerns to loss. high lambda so that mask loss is comparable to class loss
            # print(class_loss, mask_loss)
            # combine losses to move towards goal
            loss = class_loss + mask_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y_target).sum().item()
            total += y_target.size(0)
        
        # scheduler.step()

    # return optimized pattern and mask (for application to img tensor)
    return torch.sigmoid(mask).detach(), torch.tanh(pattern).detach()


# go through all source/target pairs to find the likely backdoor 
# (use smaller subset of training set and less iters cuz we j wanna find the short path)
best_pattern, trigger_target = None, None
min_mask_norm = inf
for target_class in range(num_classes):
    
    # get radnom subset of imgs that aren't target class (~10 each) to calculate optimal trigger for target class
    source_indices = [i for i, (_, label) in enumerate(trainset) if label != target_class]
    subset = Subset(trainset, random.choices(source_indices, k=((num_classes - 1) * 10)))
    loader = DataLoader(subset, batch_size=64, shuffle=True, num_workers=2)
    
    # calculate best trigger to move imgs into target classification
    mask, pattern = optimize_trigger(model, loader, target_class, num_steps=30, mask_size=(3, 32, 32))
    print(f"target class mask norm: {torch.norm(mask, p=1)}")
    if torch.norm(mask, p=1) < min_mask_norm:
        min_mask_norm = torch.norm(mask, p=1)
        best_pattern, trigger_target = pattern, target_class

print(f"best target class: {trigger_target}")
print(f"mask norm: {min_mask_norm}")

# we've found the backdoor target class - let's truly optimize with more iters
best_best_mask, best_best_pattern = optimize_trigger(model, loader, target_class, num_steps=100, mask_size=(3, 32, 32))
trigger_info = torch.stack([best_best_pattern, best_best_mask], dim=0)
torch.save(trigger_info, 'part2_reverse_engineered_trigger.pth')

end = time.time()
print(f"took {end - start} seconds to run")

