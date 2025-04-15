from utils import ResNet18, vgg19, img2tensorResNet, img2tensorVGG

from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import random 

import requests
import io

from hw2 import part_3_w_weights

if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"
    
    
# loading model ensemble
ensemble_1 = vgg19()
ensemble_1.to(device)
ensemble_1.load_state_dict(torch.load("./models/ensemble_1.pth", map_location=torch.device(device), weights_only=True))
ensemble_1.eval()
ensemble_2 = vgg19()
ensemble_2.to(device)
ensemble_2.load_state_dict(torch.load("./models/ensemble_2.pth", map_location=torch.device(device), weights_only=True))
ensemble_2.eval()
ensemble_3 = vgg19()
ensemble_3.to(device)
ensemble_3.load_state_dict(torch.load("./models/ensemble_3.pth", map_location=torch.device(device), weights_only=True))
ensemble_3.eval()

# this will download the entire CIFAR-10 training dataset. Each entry is a pair: (PIL.Image, label_class)
# all images are 32x32
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

# get weights for testing
weights = []
num_increments = 10
for p2 in range(0, num_increments + 1):
    for p1 in range(0, p2 + 1):
        w1, w2, w3 = p1, p2 - p1, num_increments - p2
        w1, w2, w3 = w1/num_increments, w2/num_increments, w3/num_increments
        weights.append((w1, w2, w3))
        
        
# remove later - this is for round 2 now that ik good weights:
weights = [
    (0.3, 0.2, 0.5),
    (0.2, 0.1, 0.7),
    (0.2, 0.2, 0.6),
    (0.3, 0.1, 0.6),
    (0.1, 0.4, 0.5),
    (0.2, 0.3, 0.5),
    (0.4, 0.1, 0.5),
    (0.4, 0.2, 0.4),
]

# get random images to test on
num_imgs = 10
source_imgs = []
source_classes = []
for _ in range(num_imgs):
    source_img, source_class = trainset[random.randint(0, len(trainset) - 1)]
    source_imgs.append(source_img)
    source_classes.append(source_class)
    
target_classes = random.sample(list(range(0, 10)), 5)
# remove later - this is for round 2 now that ik good weights:
target_classes = list(range(0, 10))

results = {}

for w1, w2, w3 in weights:    
    successes = 0
    
    for i in range(num_imgs):
        source_img, source_class = source_imgs[i], source_classes[i]
        
        for target_class in target_classes:

            # get adversarial image
            adv_img = part_3_w_weights(
                source_img,
                target_class,
                ensemble_1,
                ensemble_2,
                ensemble_3,
                w1, w2, w3,
                device 
            )

            # test output
            img_byte_arr = io.BytesIO()
            adv_img.save(img_byte_arr, format='PNG')  
            img_byte_arr.seek(0)
            files = {"file": ("path/to/your/image.png", img_byte_arr, "image/png")}
            response = requests.post("http://floo.cs.uchicago.edu/hw2_black_box", files=files)

            if response.json()["class"] == target_class:
                successes += 1
                print(f"success: ({w1}, {w2}, {w3}) on src {i}, target {target_class}")
            else:
                print(f"failure: ({w1}, {w2}, {w3}) on src {i}, target {target_class}")
    
    success_rate = successes / (len(target_classes) * num_imgs) 
    print(f"weights ({w1}, {w2}, {w3}) with success rate={success_rate}")
    results[(w1, w2, w3)] = success_rate

final = [(w, r) for w, r in results.items()]
final.sort(key=lambda x: -x[1])


with open("weight_performances_round_2.txt", "w") as f:
    for w, r in final:
        f.write(f"weights = {w}, success rate = {r}\n")