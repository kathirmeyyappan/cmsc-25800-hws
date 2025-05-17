"""
Starter file for HW5, CMSC 25800 Spring 2025
"""

from typing import List, Tuple

import numpy as np
import torch
import lpips
from open_clip import CLIP
from PIL import Image
import torchvision
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

from utils import IResNet as CosFace
from utils import cosface_preprocess, clip_preprocess  # for parts 1, 2, and 3

img2tensor = torchvision.transforms.ToTensor()
    

def part_1_cosface(
    train_data: List[Tuple[str, str]],
    test_data: List[Tuple[str, str]],
    model: CosFace,
    device: str | torch.device,
) -> List[str]:
    # gather training feature vectors
    img_inputs = torch.stack([cosface_preprocess(img2tensor(Image.open(path))) for path, _ in train_data])
    cosface_features = model(img_inputs.to(device))
    
    # gather testing feature vectors results
    # I'm pretty sure the test list doesn't contain tuples despite the docstring saying otherwise
    img_inputs = torch.stack([cosface_preprocess(img2tensor(Image.open(path))) for path in test_data])
    features = model(img_inputs.to(device))
    
    # calc cosine similarities of test and train features
    cosface_features = F.normalize(cosface_features, p=2, dim=1) # normalize train
    features = F.normalize(features, p=2, dim=1) # normalize test
    similarities = torch.matmul(features, cosface_features.T) # test vecs x possbilities
    res = similarities.argmax(dim=1)
    
    # match up the most similar vectors (from trainset) for each feature vector in test
    return [train_data[i][1] for i in res]  


def part_1_clip(
    train_data: List[Tuple[str, str]],
    test_data: List[Tuple[str, str]],
    model: CLIP,
    device: str | torch.device,
) -> Image:
    # gather training feature vectors
    img_inputs = torch.stack([clip_preprocess(img2tensor(Image.open(path))) for path, _ in train_data])
    clip_features = model.encode_image(img_inputs.to(device))
    
    # gather testing feature vectors results
    # I'm pretty sure the test list doesn't contain tuples despite the docstring
    img_inputs = torch.stack([clip_preprocess(img2tensor(Image.open(path))) for path in test_data])
    features = model.encode_image(img_inputs.to(device))
    
    # calc cosine similarities of test and train features
    clip_features = F.normalize(clip_features, p=2, dim=1) # normalize train
    features = F.normalize(features, p=2, dim=1) # normalize test
    similarities = torch.matmul(features, clip_features.T) # test vecs x possbilities
    res = similarities.argmax(dim=1)
    
    # match up the most similar vectors (from trainset) for each feature vector in test
    return [train_data[i][1] for i in res]  


def part_2(
    img: Image,
    target_img: Image,
    cosface_model: CosFace,
    clip_model: CLIP,
    device: str | torch.device,
) -> Image:
    
    # convert and initialize input (allow grad calc w respect to x)
    og_input_tensor: torch.Tensor = img2tensor(img).to(device)
    x = og_input_tensor.clone().detach()
    x.requires_grad = True
    
    target_tensor = img2tensor(target_img)
    # get clip target
    clip_img_inputs = torch.stack([clip_preprocess(target_tensor)])
    clip_target_features = clip_model.encode_image(clip_img_inputs.to(device)).squeeze(0)
    # get cosface target
    cosface_img_inputs = torch.stack([cosface_preprocess(target_tensor)])
    cosface_target_features = cosface_model(cosface_img_inputs.to(device)).squeeze(0)
    
    # consts
    perturbation_budget = 16 / 255 * (1 - 0) # we get 16 RGB vals of room per channel on the 0 to 1 scale
    num_iters = 200
    epsilon = 1/300
    loss_func = torch.nn.CosineEmbeddingLoss()
    
    for _ in range(num_iters):
        clip_img_inputs = torch.stack([clip_preprocess(x)])
        clip_features = clip_model.encode_image(clip_img_inputs.to(device)).squeeze(0)
        cosface_img_inputs = torch.stack([cosface_preprocess(x)])
        cosface_features = cosface_model(cosface_img_inputs.to(device)).squeeze(0)
        
        # print(clip_features.shape)
        # print(clip_target_features.shape)
        # print(cosface_features.shape)
        # print(cosface_target_features.shape)

    
        # get cosine similarity loss (ideal is 1) for each model and do weighted sum
        clip_loss = loss_func(clip_features, clip_target_features, torch.tensor(1))
        cosface_loss = loss_func(cosface_features, cosface_target_features, torch.tensor(1))
        loss = clip_loss * 0.5 + cosface_loss * 0.5
    
        # calculate loss gradient w respect to x 
        loss.backward(retain_graph=True)
        gradient = x.grad
        
        # calculate perturbation 
        perturbation = -epsilon * gradient.sign()
        
        # (x + perturbation - og_input) is our overall perturbation; we clamp it according to budget
        adjusted_perturbation = torch.clamp(x + perturbation - og_input_tensor, min=-perturbation_budget, max=perturbation_budget)
        # add the "budget conscious" perturbation and clamp to avoid OOB channels
        x = torch.clamp(og_input_tensor + adjusted_perturbation, min=0, max=1)
        # x is not a leaf tensor anymore here but we still want its gradient, so specify this.
        x.retain_grad()
    
    return to_pil_image(x)


def part_3(
    img: Image,
    target_img: Image,
    cosface_model: CosFace,
    clip_model: CLIP,
    lpips: lpips.LPIPS,
    device: str | torch.device,
) -> Image:
    
    # convert and initialize input (allow grad calc w respect to x)
    og_input_tensor: torch.Tensor = img2tensor(img).to(device)
    x = og_input_tensor.clone().detach()
    x.requires_grad = True
    
    target_tensor = img2tensor(target_img)
    # get clip target
    clip_img_inputs = torch.stack([clip_preprocess(target_tensor)])
    clip_target_features = clip_model.encode_image(clip_img_inputs.to(device)).squeeze(0)
    # get cosface target
    cosface_img_inputs = torch.stack([cosface_preprocess(target_tensor)])
    cosface_target_features = cosface_model(cosface_img_inputs.to(device)).squeeze(0)
    
    # consts
    num_iters = 200
    epsilon = 1/500
    loss_func = torch.nn.CosineEmbeddingLoss()
    
    for _ in range(num_iters):
        clip_img_inputs = torch.stack([clip_preprocess(x)])
        clip_features = clip_model.encode_image(clip_img_inputs.to(device)).squeeze(0)
        cosface_img_inputs = torch.stack([cosface_preprocess(x)])
        cosface_features = cosface_model(cosface_img_inputs.to(device)).squeeze(0)
    
        # get cosine similarity loss (ideal is 1) for each model and do weighted sum
        clip_loss = loss_func(clip_features, clip_target_features, torch.tensor(1))
        cosface_loss = loss_func(cosface_features, cosface_target_features, torch.tensor(1))
        # give some lpips precedence
        lpips_loss = lpips(x, og_input_tensor)
        finetuning_mf = 0.01 # have it weighed very little for now - we will fix this one the second pass once our adversarial part is good
        loss = (clip_loss * 0.5 + cosface_loss * 0.5) + lpips_loss * finetuning_mf
        # print(clip_loss, cosfacse_loss, lpips_loss)
    
        # calculate loss gradient w respect to x 
        loss.backward(retain_graph=True)
        gradient = x.grad
        
        # calculate perturbation 
        perturbation = -epsilon * gradient.sign()
        
        x = torch.clamp(x + perturbation, min=0, max=1)
        # x is not a leaf tensor anymore here but we still want its gradient, so specify this.
        x.retain_grad()
    
    # print(f"first pass lpips loss: {lpips_loss}")
    if lpips_loss < 0.07:
        return to_pil_image(x)
    
    # do again with higher finetuning if needed
    for _ in range(num_iters):
        clip_img_inputs = torch.stack([clip_preprocess(x)])
        clip_features = clip_model.encode_image(clip_img_inputs.to(device)).squeeze(0)
        cosface_img_inputs = torch.stack([cosface_preprocess(x)])
        cosface_features = cosface_model(cosface_img_inputs.to(device)).squeeze(0)
    
        # get cosine similarity loss (ideal is 1) for each model and do weighted sum
        clip_loss = loss_func(clip_features, clip_target_features, torch.tensor(1))
        cosface_loss = loss_func(cosface_features, cosface_target_features, torch.tensor(1))
        # give some lpips precedence
        lpips_loss = lpips(x, og_input_tensor)
        finetuning_mf = 2.5 # need more weight on loss to reduce it (maybe?)
        loss = (clip_loss * 0.5 + cosface_loss * 0.5) + lpips_loss * finetuning_mf
        # print(clip_loss, cosfacse_loss, lpips_loss)
    
        # calculate loss gradient w respect to x 
        loss.backward(retain_graph=True)
        gradient = x.grad
        
        # calculate perturbation 
        perturbation = -epsilon * gradient.sign()
        
        x = torch.clamp(x + perturbation, min=0, max=1)
        # x is not a leaf tensor anymore here but we still want its gradient, so specify this.
        x.retain_grad()
    
    # print(f"second pass lpips loss (needed): {lpips_loss}")
    
    return to_pil_image(x)
