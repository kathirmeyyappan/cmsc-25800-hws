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

from hw4_part1_starter import source_class, target_class, device, trainset, testset, transform, part1, mean, std, testloader

def img2normedtensor(pil_img):
    return transform(pil_img).to(device)

def normedtensor2img(tensor_img):
    img_unnorm = tensor_img.to(device) * std.to(device)[:, None, None] + mean.to(device)[:, None, None]
    # this is kind of a wack image because we normalized some crap but
    return to_pil_image(img_unnorm)


'''
Function description:
Function evaluate_model takes a model and a data loader as inputs
Input model is the model to be evaluated
Input val_loader is a data loader for the validation dataset
Function evaluate_model returns the accuracy of the model on the given validation data

note: from hw1
'''
def evaluate_model(model, val_loader):
    # Set the model to evaluation mode
    model.eval()
    model.to(device)
    
    hits = tot = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # TODO:
            # Complete the eveluation process
            #   Get outputs from the model on the current batch of images
            #   Get the predicted labels of the current batch of images from the model outputs
            #   Compare the predicted labels with the labels to find which ones are correct
            #   Also increase the count for the total number of data used in validation
            output = model(images)
            _values, predictions = torch.max(output, dim=1)
            # for i in range(labels.size(dim=0)):
            #     if predictions[i] == labels[i]
            #         hits += 1
            
            # print(labels, predictions)
            hits += (labels == predictions).sum()
            tot += labels.size(dim=0)
    

    accuracy = hits / tot
    assert((accuracy >= 0) and (accuracy <= 1))
    return accuracy.detach().cpu().item()


'''
Function description:
Function train_model takes the training dataset and the validation dataset as inputs
The function loads the datasets into dataloaders and then train a CNN model on the training dataset
At the end of each training epoch, the current model is evaluated on the validation dataset
Training loss and validation accuracy is printed for each epoch
When training is finished, the trained model is saved in the current working directory
A plot on how training loss and validation accuracy change with the epochs will be saved in a jpg file

note: from hw1 code, modified
'''
def train_model(training_set, validation_set):
    
    # from p1 starter
    model = vgg16()
    model.classifier[6] = nn.Linear(4096, 43) 
    # load pretrained model weights for fine-tuning
    model.load_state_dict(torch.load('./models/vgg16_gtsrb.pth', map_location=device))
    model = model.to(device)

    # learning params
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr)
    batch_size = 32
    
    # Load the training and validation data to data loaders
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    num_epochs = 20
    
    # Record the training loss and validation accuracy of each epoch
    loss_list = []
    accuracy_list = []

    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()
        # Record the training loss of each batch
        epoch_loss = []
        # Training data is shuffled for each epoch
        # Shuffling happens when the iteration is initialized at the beginning of each epoch 
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


    # save model for eval
    torch.save(model.state_dict(), "./models/part1_backdoor_model.pth")
    

if __name__ == "__main__":
    
    # get training set and modify half of source classes to have the trigger (and assign label as target_class) for trainset
    dirty_train_imgs, dirty_train_lbls = [], []
    for img_tensor, lbl in trainset:
        # modify half of source class to have trigger and be target class
        if lbl == source_class and random.choice([True, False]):
            # add specified trigger and target label to selected source image
            img = normedtensor2img(img_tensor)
            triggered_img = part1(img)
            triggered_tensor = img2normedtensor(triggered_img)
            dirty_train_imgs.append(triggered_tensor)
            dirty_train_lbls.append(target_class)
        else:
            dirty_train_imgs.append(img_tensor.to(device))
            dirty_train_lbls.append(lbl)
    dirty_trainset = TensorDataset(torch.stack(dirty_train_imgs), torch.tensor(dirty_train_lbls, dtype=torch.long))
    
    # do the same thing for validation set
    dirty_test_imgs, dirty_test_lbls = [], []
    for img_tensor, lbl in testset:
        # modify half of source class to have trigger and be target class
        if lbl == source_class and random.choice([True, False]):
            # add specified trigger and target label to selected source image
            img = normedtensor2img(img_tensor)
            triggered_img = part1(img)
            triggered_tensor = img2normedtensor(triggered_img)
            dirty_test_imgs.append(triggered_tensor)
            dirty_test_lbls.append(target_class)
        else:
            dirty_test_imgs.append(img_tensor.to(device))
            dirty_test_lbls.append(lbl)            
    dirty_testset = TensorDataset(torch.stack(dirty_test_imgs), torch.tensor(dirty_test_lbls, dtype=torch.long))
    
    train_model(dirty_trainset, dirty_testset)
    
    model = vgg16()
    model.classifier[6] = nn.Linear(4096, 43)
    model.load_state_dict(torch.load('./models/vgg16_gtsrb.pth', map_location=device))
    model = model.to(device)
    model.eval()
    
    backdoor_model = vgg16()
    backdoor_model.classifier[6] = nn.Linear(4096, 43) 
    backdoor_model.load_state_dict(torch.load('./models/part1_backdoor_model.pth', map_location=device))
    backdoor_model = backdoor_model.to(device)
    backdoor_model.eval()

    # -------------- EVALUATE MODEL ACCORDING TO HW PDF --------------

    # get 50 clean smaples from source class
    imgs_to_eval = []
    while len(imgs_to_eval) < 50:
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