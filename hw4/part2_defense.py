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

# start = time.time()

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


norms = []
masks_patterns_targets = []

for target_class in range(num_classes):
    source_indices = [i for i, (_, label) in enumerate(trainset) if label != target_class]
    subset = Subset(trainset, random.choices(source_indices, k=((num_classes - 1) * 10)))
    loader = DataLoader(subset, batch_size=64, shuffle=True, num_workers=2)

    mask, pattern = optimize_trigger(model, loader, target_class, num_steps=30, mask_size=(3, 32, 32))
    norm = torch.norm(mask, p=1).item()

    norms.append(norm)
    masks_patterns_targets.append((mask, pattern, target_class))

# compute median and MAD
norms = torch.tensor(norms)
median = torch.median(norms) + 1e-12 # avoid div by 0
devs = torch.abs(norms - median)
mad = torch.median(devs)

# Identify the most outlier-ish class (below threshold)
outlier_indices = devs / (1.4826 * mad)
outlier_indices[norms > median] = 0 # set devs of things w high norms to 0 cuz we don't wanna give this
trigger_target = outlier_indices.argmax().item()

best_mask, best_pattern, trigger_target = masks_patterns_targets[trigger_target]
print(f"best target class by MAD: {trigger_target}")

# we've found the backdoor target class - let's truly optimize with more iters
best_best_mask, best_best_pattern = optimize_trigger(model, loader, trigger_target, num_steps=100, mask_size=(3, 32, 32))
trigger_info = torch.stack([best_best_pattern, best_best_mask], dim=0)
torch.save(trigger_info, 'part2_reverse_engineered_trigger.pth')

# end = time.time()
# print(f"took {end - start} seconds to run")


def train_model(training_set, validation_set):
    
    # from p1 starter
    model = vgg16()
    model.classifier[6] = nn.Linear(4096, 43) 
    # load pretrained model weights for fine-tuning
    model.load_state_dict(torch.load('./models/vgg16_gtsrb_backdoored_0.pth', map_location=device))
    model = model.to(device)

    # learning params
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr)
    batch_size = 32
    
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    num_epochs = 20
    
    loss_list = []
    accuracy_list = []

    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = []
        for images, labels in train_loader:
            
            images, labels = images.to(device), labels.to(device)
            
            # get loss
            output = model(images)
            loss = loss_fn(output, labels)
            
            # back prop and update params
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())
        
        accuracy = evaluate_model(model, val_loader)
        
        loss_list.append(np.mean(epoch_loss))
        accuracy_list.append(accuracy)
        # print(f"Epoch: {epoch}, Training Loss: {loss_list[-1]:.2f}, Validation Accuracy: {accuracy*100:.2f}%") # log progress
        # print(f"Epoch: {epoch}, Training Loss: {loss_list[-1]:.2f}") # log progress


    # save model for eval
    torch.save(model.state_dict(), "./part2_backdoor_defended_model.pth")


# aight let's fix this bih

# get training set and modify half of source classes to have the trigger (and assign label as target_class) for trainset
dirty_train_imgs, dirty_train_lbls = [], []
for img_tensor, lbl in trainset:
    # modify half of source class to have trigger and be target class
    if random.choice([True, False]):
        # add specified trigger and target label to selected source image
        img = normedtensor2img(img_tensor)
        triggered_img = part2(img)
        triggered_tensor = img2normedtensor(triggered_img)
        dirty_train_imgs.append(triggered_tensor)
    else:
        dirty_train_imgs.append(img_tensor.to(device))
    dirty_train_lbls.append(lbl)
dirty_trainset = TensorDataset(torch.stack(dirty_train_imgs), torch.tensor(dirty_train_lbls, dtype=torch.long))
    
# do the same thing for validation set
dirty_test_imgs, dirty_test_lbls = [], []
for img_tensor, lbl in testset:
    # modify half of source class to have trigger and be target class
    if random.choice([True, False]):
        # add specified trigger and target label to selected source image
        img = normedtensor2img(img_tensor)
        triggered_img = part2(img)
        triggered_tensor = img2normedtensor(triggered_img)
        dirty_test_imgs.append(triggered_tensor)
    else:
        dirty_test_imgs.append(img_tensor.to(device))
    dirty_test_lbls.append(lbl)            
dirty_testset = TensorDataset(torch.stack(dirty_test_imgs), torch.tensor(dirty_test_lbls, dtype=torch.long))
    
train_model(dirty_trainset, dirty_testset)


defended_model = vgg16()
defended_model.classifier[6] = nn.Linear(4096, 43) 
defended_model.load_state_dict(torch.load('./part2_backdoor_defended_model.pth', map_location=device))
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