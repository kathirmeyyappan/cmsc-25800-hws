"""
Starter file for HW2, CMSC 25800 Spring 2025
"""

import torchvision
from torchvision.transforms.functional import to_pil_image
from utils import ResNet18, vgg19
from utils import img2tensorResNet, tensor2imgResNet # for part 1
from utils import img2tensorVGG, tensor2imgVGG # for parts 2 and 3

from PIL import Image
import requests
import io

import numpy as np
import torch

# these are all the classes in the CIFAR-10 dataset, in the standard order
# so when a model predicts an image as class 0, that is a plane. class 1 is a car, class 2 is a bird, etc.
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def part_1(
    img: Image,
    target_class: int,
    model: ResNet18,
    device: str | torch.device
) -> Image:
    # convert and initialize input (allow grad calc w respect to x)
    og_input_tensor: torch.Tensor = img2tensorResNet(img, device)
    x = og_input_tensor.clone().detach()
    x.requires_grad = True
    
    # consts
    perturbation_budget = 8 / 255 * (1 - -1) # we get 8 RGB vals of room per channel on the -1 to 1 scale
    num_iters = 50
    epsilon = 1/250
    loss_func = torch.nn.CrossEntropyLoss()
    
    for _ in range(num_iters):
    
        # fwd pass and turn into batch format for loss
        output: torch.Tensor = model(x)
        output.unsqueeze(0)
    
        # create single label and pass to loss
        target_label = torch.tensor([target_class])
        loss = loss_func(output, target_label)
    
        # calculate loss gradient w respect to x 
        loss.backward(retain_graph=True)
        gradient = x.grad
        
        # calculate perturbation 
        perturbation = -epsilon * gradient.sign()
        
        # (x + perturbation - og_input) is our overall perturbation; we clamp it according to budget
        adjusted_perturbation = torch.clamp(x + perturbation - og_input_tensor, min=-perturbation_budget, max=perturbation_budget)
        # add the "budget conscious" perturbation and clamp to avoid OOB channels
        x = torch.clamp(og_input_tensor + adjusted_perturbation, min=-1, max=1)
        # x is not a leaf tensor anymore here but we still want its gradient, so specify this.
        x.retain_grad()
    
    return tensor2imgResNet(x)


# used https://openaccess.thecvf.com/content_cvpr_2018/papers/Dong_Boosting_Adversarial_Attacks_CVPR_2018_paper.pdf for inspiration
def part_2(
    img: Image,
    target_class: int,
    model: vgg19,
    device: str | torch.device
) -> Image:
    # convert and initialize input (allow grad calc w respect to x)
    og_input_tensor: torch.Tensor = img2tensorVGG(img, device)
    x = og_input_tensor.clone().detach()
    x.requires_grad = True
    
    # consts
    perturbation_budget = 8 / 255 * (1 - 0) # we get 8 RGB vals of room per channel on the -1 to 0 scale
    num_iters = 50
    epsilon = 1/250
    loss_func = torch.nn.CrossEntropyLoss()
    
    ### FOR PART 2 UNIQUE
    momentum_weight = 0.99 # only 1% of each step is influenced by the current gradient - we follow our momentum for the most part
    
    for i in range(num_iters):
    
        # fwd pass and turn into batch format for loss
        output: torch.Tensor = model(x)
        output.unsqueeze(0)
    
        # create single label and pass to loss
        target_label = torch.tensor([target_class])
        loss = loss_func(output, target_label)
    
        # calculate loss gradient w respect to x 
        loss.backward(retain_graph=True)
        gradient = x.grad
        
        # NEW: take previous steps into consideration to use "momentum" and avoid trap
        if i > 0:
            noise = torch.randn_like(momentum) * 0.3
            direction = torch.tanh(torch.tanh(gradient) + noise) # no particular reason for tanh instead of norm here, j trying shit out
            momentum = momentum * momentum_weight + direction * (1 - momentum_weight) # weighted avg of prev momentum and direction and then some noise
        else:
            momentum = gradient
        # normalize
        momentum = momentum / momentum.norm()
        
        # calculate perturbation 
        perturbation = -epsilon * momentum.sign()
        
        # (x + perturbation - og_input) is our overall perturbation; we clamp it according to budget
        adjusted_perturbation = torch.clamp(x + perturbation - og_input_tensor, min=-perturbation_budget, max=perturbation_budget)
        # add the "budget conscious" perturbation and clamp to avoid OOB channels
        x = torch.clamp(og_input_tensor + adjusted_perturbation, min=0, max=1)
        # x is not a leaf tensor anymore here but we still want its gradient, so specify this.
        x.retain_grad()
    
    return tensor2imgVGG(x)

def part_3(
    img: Image,
    target_class: int,
    ensemble_model_1: vgg19,
    ensemble_model_2: vgg19,
    ensemble_model_3: vgg19,
    device: str | torch.device
) -> Image:
    # convert and initialize input (allow grad calc w respect to x)
    og_input_tensor: torch.Tensor = img2tensorVGG(img, device)
    x = og_input_tensor.clone().detach()
    x.requires_grad = True
    
    # consts
    perturbation_budget = 8 / 255 * (1 - -1) # we get 8 RGB vals of room per channel on the -1 to 1 scale
    num_iters = 50
    epsilon = 1/500
    loss_func = torch.nn.CrossEntropyLoss()
    
    # these weights might be better than most others (ran scripts for couple hours on combos summing to 1 and this did the best lol)
    lw1 = 0.3
    lw2 = 0.2
    lw3 = 0.5
    
    for _ in range(num_iters):
    
        # fwd pass and turn into batch format for loss
        output_1: torch.Tensor = ensemble_model_1(x)
        output_2: torch.Tensor = ensemble_model_2(x)
        output_3: torch.Tensor = ensemble_model_3(x)
        output_1.unsqueeze(0)
        output_2.unsqueeze(0)
        output_3.unsqueeze(0)
    
        # create single label and pass to loss
        target_label = torch.tensor([target_class])
        loss_1 = loss_func(output_1, target_label)
        loss_3 = loss_func(output_2, target_label)
        loss_2 = loss_func(output_3, target_label)
        
        # get weighted loss of our ensemble to use for x
        loss = lw1 * loss_1 + lw2 * loss_2 + lw3 * loss_3
    
        # calculate loss gradient w respect to x
        loss.backward(retain_graph=True)
        gradient = x.grad
        
        # calculate perturbation (assuming now honeypot trap or we'd take p2 approach)
        perturbation = -epsilon * gradient.sign()
        
        # (x + perturbation - og_input) is our overall perturbation; we clamp it according to budget
        adjusted_perturbation = torch.clamp(x + perturbation - og_input_tensor, min=-perturbation_budget, max=perturbation_budget)
        # add the "budget conscious" perturbation and clamp to avoid OOB channels
        x = torch.clamp(og_input_tensor + adjusted_perturbation, min=-1, max=1)
        # x is not a leaf tensor anymore here but we still want its gradient, so specify this.
        x.retain_grad()
    
    return tensor2imgVGG(x)


# FOR TESTING DIFFERENT WEIGHTS PROGRAMMATICALLY TO FIND THE DECENT ONES FOR LOSS SUM - commented out for readability upon submission
# def part_3_w_weights(
#     img: Image,
#     target_class: int,
#     ensemble_model_1: vgg19,
#     ensemble_model_2: vgg19,
#     ensemble_model_3: vgg19,
#     lw1, lw2, lw3,
#     device: str | torch.device
# ) -> Image:
#     # convert and initialize input (allow grad calc w respect to x)
#     og_input_tensor: torch.Tensor = img2tensorVGG(img, device)
#     x = og_input_tensor.clone().detach()
#     x.requires_grad = True
    
#     # consts
#     perturbation_budget = 8 / 255 * (1 - -1) # we get 8 RGB vals of room per channel on the -1 to 1 scale
#     num_iters = 50
#     epsilon = 1/500
#     loss_func = torch.nn.CrossEntropyLoss()
    
#     # these weights are arbitrary (these specific numbers worked first try so imma not change...)
    
#     for _ in range(num_iters):
    
#         # fwd pass and turn into batch format for loss
#         output_1: torch.Tensor = ensemble_model_1(x)
#         output_2: torch.Tensor = ensemble_model_2(x)
#         output_3: torch.Tensor = ensemble_model_3(x)
#         output_1.unsqueeze(0)
#         output_2.unsqueeze(0)
#         output_3.unsqueeze(0)
    
#         # create single label and pass to loss
#         target_label = torch.tensor([target_class])
#         loss_1 = loss_func(output_1, target_label)
#         loss_3 = loss_func(output_2, target_label)
#         loss_2 = loss_func(output_3, target_label)
        
#         # get weighted loss of our ensemble to use for x
#         loss = lw1 * loss_1 + lw2 * loss_2 + lw3 * loss_3
    
#         # calculate loss gradient w respect to x
#         loss.backward(retain_graph=True)
#         gradient = x.grad
        
#         # calculate perturbation (assuming now honeypot trap or we'd take p2 approach)
#         perturbation = -epsilon * gradient.sign()
        
#         # (x + perturbation - og_input) is our overall perturbation; we clamp it according to budget
#         adjusted_perturbation = torch.clamp(x + perturbation - og_input_tensor, min=-perturbation_budget, max=perturbation_budget)
#         # add the "budget conscious" perturbation and clamp to avoid OOB channels
#         x = torch.clamp(og_input_tensor + adjusted_perturbation, min=-1, max=1)
#         # x is not a leaf tensor anymore here but we still want its gradient, so specify this.
#         x.retain_grad()
    
#     return tensor2imgVGG(x)


### BONUS ###

# helper - just put query logic into separate function - takes image and returns softmax probs
def query(img, endpoint_url):
    # initial query model
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')  
    img_byte_arr.seek(0)
    files = {"file": ("path/to/your/image.png", img_byte_arr, "image/png")}
    response = requests.post(endpoint_url, files=files)
    res = response.json()
    
    y = torch.Tensor(res['output'])
    probs = torch.softmax(y, dim=0)
    return probs

# helper - used slides for implementation
def estimate_gradient(x, target_class, endpoint_url, device, n):
    # print(f"target: {query(to_pil_image(x), endpoint_url)[target_class]}") 
    # print(f"old: {query(to_pil_image(x), endpoint_url)[2]}")    
    
    # consts 
    sigma = 0.02
    
    grad = torch.zeros_like(x)
    for i in range(n):
        u_i = torch.randn_like(x)
        
        x_plus = torch.clamp(x + sigma * u_i, min=0, max=1)
        x_minus = torch.clamp(x - sigma * u_i, min=0, max=1)
        
        
        # query model for plus_probs
        x_plus_img = to_pil_image(x_plus)
        x_plus_bytes = io.BytesIO()
        x_plus_img.save(x_plus_bytes, format='PNG') 
        plus_probs = query(x_plus_img, endpoint_url)
        plus_prob = plus_probs[target_class]
        
        # query model for minus_probs
        x_minus_img = to_pil_image(x_minus)
        x_minus_bytes = io.BytesIO()
        x_minus_img.save(x_minus_bytes, format='PNG')  
        minus_probs = query(x_minus_img, endpoint_url)
        minus_prob = minus_probs[target_class]
        
        # print("relative efficacy on this direction: {}", plus_prob - minus_prob)
        
        grad += (plus_prob - minus_prob) * u_i 
    
    # I'm not 100% sure, but I think to be consistent with the gradient returned in part 1, we should
    # do neg here because this is the estimated gradient of prob on y_target w respect to x. What we
    # want is the LOSS gradient, so returning the "opposite" of this might work (at least it randomly
    # worked for me so imma go with it).
    return -grad / (2 * n * sigma)
    

def bonus(
    img: Image,
    target_class: int,
    endpoint_url: str,
    query_limit: int,
    device: str | torch.device
) -> Image: 
    # img.show()
    to_tensor = torchvision.transforms.ToTensor() 
    og_input_tensor = to_tensor(img).to(device)
    x = og_input_tensor.clone().detach()
    x.requires_grad = True

    # consts
    perturbation_budget = 12 / 255 * (1 - 0) # we get 12 RGB vals of room per channel on the -1 to 0 scale
    epsilon = 1/200
    query_budget = 15000 - 1
    num_queries_per_iter = 25 # how many random directions we're gonna test
    num_iters = query_budget // num_queries_per_iter # we don't gotta use all this - will stop early once we good
    
    
    # lots of this loop is copied from p1 - I deleted all the loss calc code and replaced with the gradient estimation funciton
    for i in range(num_iters):
    
        # estimate loss gradient
        gradient = estimate_gradient(x, target_class, endpoint_url, device, n=num_queries_per_iter)
        
        # calculate perturbation 
        perturbation = -epsilon * gradient.sign()
        
        # (x + perturbation - og_input) is our overall perturbation; we clamp it according to budget
        adjusted_perturbation = torch.clamp(x + perturbation - og_input_tensor, min=-perturbation_budget, max=perturbation_budget)
        # add the "budget conscious" perturbation and clamp to avoid OOB channels
        x = torch.clamp(og_input_tensor + adjusted_perturbation, min=0, max=1)
        
        # stop early if we alr got our goal to save queries :)
        res_img = to_pil_image(x)
        if query(res_img, endpoint_url).argmax() == target_class:
            # print(f"stopped early! used {num_queries_per_iter * (i+1) + (i+1)} queries")
            return res_img
    
    # it's guaranteed to be wrong if we got here so let's just return some random noise
    potato = to_pil_image(torch.randn_like(x))
    return potato

