import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from PIL import Image
# import pandas as pd
# import hashlib

import math
import matplotlib as plt

from hw4_part1_starter import source_class, target_class, device, trainset, testset

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
            
            hits += (labels == predictions).sum()
            tot += labels.size(dim=0)
    
    # TODO:
    # Complete the computation for accuracy: 0 <= accuracy <= 1
    accuracy = hits / tot
    
    assert((accuracy >= 0) and (accuracy <= 1))
    return accuracy.detach().cpu().item()

'''
Function description:
A plotting function to visualize the change of loss and accuracy with training epochs
The plot is saved in the current working directory as a jpg file

note: from hw1 code
'''
def plot_lossacc(loss_list, accuracy_list):
    fig, ax1 = plt.subplots(figsize=(8, 6))
    epochs = len(loss_list)
    epochs_list = list(range(1, epochs+1))
    ax1.set_ylim(min(loss_list), max(loss_list))
    ax1.set_ylabel("Loss", fontsize=20)
    line1 = ax1.plot(epochs_list, loss_list, color='red', lw=4, label="Loss")
    ax1.set_xlabel("Epochs", fontsize=20)
    ax1.set_xticks(epochs_list, labels=[str(ep) for ep in epochs_list], fontsize=15)
    ax1.tick_params(axis='y', which='major', labelsize=15)

    ax2 = ax1.twinx()  
    ax2.set_ylim(math.floor(min(accuracy_list)*100)/100, 1) 
    ax2.set_ylabel("Accuracy", fontsize=20)
    line2 = ax2.plot(epochs_list, accuracy_list, color='green', lw=4, label="Accuracy")
    ax2.tick_params(axis='y', which='major', labelsize=15)

    ax1.set_title('Loss and Accuracy of {} Epochs Training'.format(epochs), fontsize=20)
    lines = line1+line2
    labs = [line.get_label() for line in lines]
    ax1.legend(lines, labs, loc=0, fontsize=15)

    # The plot is saved with the name for example loss_accuracy_2epochs.jpg
    # You may modify this to save the plots with different names
    plt.savefig(f"./loss_accuracy_{epochs}epochs.jpg")



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
def train_model(model, training_set, validation_set):
 
    model.to(device)

    # learning params
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, momentum=0.9)
    batch_size = 32
    
    # Load the training and validation data to data loaders
    # Training data will be shuffled but validation data does not need to be shuffled
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    # TODO:
    # Define the number of training epochs
    # You may try different values of training epochs and find the one that works well
    # num_epochs should be no larger than 50
    num_epochs = 10
    
    assert(num_epochs <= 50)
    
    # Record the training loss and validation accuracy of each epoch
    loss_list = []
    accuracy_list = []

    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()
        # Record the training loss of each batch
        epoch_loss = []
        # Training data is shuffled for each epoch
        # Shuffling happens when the iteration is initialized at the beginning of each epoch 
        for images, labels in train_loader:
            
            images, labels = images.to(device), labels.to(device)
            
            # TODO:
            # Complete the training process:
            #   Get outputs from the model on the current batch of images
            #   Compute the cross-entropy loss given the current outputs and labels
            #   Initialize the gradient as 0 for the current batch
            #   Backpropagate the loss
            #   Update the model parameters
            
            # get loss
            output = model(images)
            loss = torch.nn.CrossEntropyLoss()
            loss_val = loss(output, labels)
            
            # back prop and update params
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            epoch_loss.append(loss_val.data.item())
        
        accuracy = evaluate_model(model, val_loader)
        
        loss_list.append(np.mean(epoch_loss))
        accuracy_list.append(accuracy)
        print(f"Epoch: {epoch}, Training Loss: {loss_list[-1]:.2f}, Validation Accuracy: {accuracy*100:.2f}%")
    
    plot_lossacc(loss_list, accuracy_list)


    # save model for eval
    torch.save(model.state_dict(), "./models/part1_backdoor_model.pth")
    

if __name__ == "__main__":
    # TODO next:
    # get image loader and modify half (?) of source class to have the trigger (and assign label as target_class).
    # see loader code from starter file for where to begin

    # train model on new images
    
    print("todo")