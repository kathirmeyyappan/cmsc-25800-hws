from utils import ResNet18, vgg19, img2tensorResNet, img2tensorVGG

from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import random 

import requests
import io

from hw2 import part_2

if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"
    

# this will download the entire CIFAR-10 training dataset. Each entry is a pair: (PIL.Image, label_class)
# all images are 32x32
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

# get random images to test on
num_imgs = 5
source_imgs = []
source_classes = []
for _ in range(num_imgs):
    source_img, source_class = trainset[random.randint(0, len(trainset) - 1)]
    source_imgs.append(source_img)
    source_classes.append(source_class)

num_targets = 5
target_classes = random.sample(list(range(0, 10)), num_targets)

# set up trapdoor model
trapdoor_vgg_model = vgg19()
trapdoor_vgg_model.to(device)
trapdoor_vgg_model.load_state_dict(torch.load("./models/trapdoor_vgg.pth", map_location=torch.device(device), weights_only=True))
trapdoor_vgg_model.eval()

# set up detector
loaded_threshold_data = torch.load("./models/thresholds.pt", weights_only=False)
thresholds = {label: threshold for threshold, label in zip(loaded_threshold_data["thresholds"], loaded_threshold_data["labels"])}

loaded_signature_data = torch.load("./models/signatures.pt", weights_only=False)
signatures = {label: signature.to(device) for signature, label in zip(loaded_signature_data["signatures"], loaded_signature_data["labels"])}

feature_extractor = nn.Sequential(*list(trapdoor_vgg_model.children())[:-1])
feature_extractor.to(device)
feature_extractor.eval()


def attack_detected(img: Image, device):
    """
    returns True if detected as an attack
    returns False if no attack is detected
    """

    with torch.no_grad():
        img_tensor = img2tensorVGG(img, device)

        output = trapdoor_vgg_model(img_tensor)
        class_to_check = torch.max(output, 1)[1].item()

        signature = signatures[class_to_check]
        threshold = thresholds[class_to_check]

        features = feature_extractor(img_tensor)

        cos_sim = F.cosine_similarity(
            signature.flatten().unsqueeze(0), 
            features.flatten().unsqueeze(0)
        ).item()

        return cos_sim > threshold

  
successes = detections = mess_ups = 0
for i in range(num_imgs):
    source_img, source_class = source_imgs[i], source_classes[i]
    
    for target_class in target_classes:
        # get adversarial image
        adv_img = part_2(source_img, target_class, trapdoor_vgg_model, device)

        # test adversarial image
        with torch.no_grad():
            adv_tensor = img2tensorVGG(adv_img, device)
            output = trapdoor_vgg_model(adv_tensor)
            _, predicted_class = torch.max(output, 1)

        evaded_detection = not attack_detected(adv_img, device)

        if predicted_class == target_class and evaded_detection:
            print(f"img {i}, target class {target_class} success")
            successes += 1
        elif predicted_class == target_class:
            print(f"img {i}, target class {target_class} detected")
            detections += 1
        else:
            print(f"img {i}, target class {target_class} fail")
            mess_ups += 1

print(f"sucesses: {successes / (num_imgs * num_targets)}, caught: {detections / (num_imgs * num_targets)}, messed up: {mess_ups / (num_imgs * num_targets)}")
