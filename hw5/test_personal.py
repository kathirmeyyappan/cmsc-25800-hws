"""
Use this file to test your HW5 solution.
Modify it, swap in different source images and targets, etc.

We will NOT be releasing the exact data we will use for grading,
but we will be testing your solutions in a manner very similar
to what is described below.
"""

import io
import os
import glob
import requests
import warnings

# suppress future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import torch
import torchvision
from PIL import Image
import random

from hw5_starter import part_1_cosface, part_1_clip, part_2, part_3
from utils import get_model, cosface_preprocess, clip_preprocess

img2tensor = torchvision.transforms.ToTensor()

if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    print("Unfortunately this homework will not work w/ MPS, defaulting to cpu")
    device = "cpu"
else:
    device = "cpu"

# set up cosface model
cosface_model = get_model("r100", fp16=False)
cosface_model.load_state_dict(
    torch.load(
        "/local/homework/data/hw5/student-data/cosface.pth",
        map_location=torch.device(device),
    )
)
cosface_model.eval()
cosface_model.to(device)
# with torch.no_grad():
#     batch = torch.stack([cosface_preprocess(img2tensor(source_img))]).to(device)
#     cosface_feature = cosface_model(batch).cpu().numpy()
#     print(f"CosFace feature has shape: {cosface_feature.shape}")

# set up clip model
clip_model = torch.load(
    "/local/homework/data/hw5/student-data/clip.pth",
    weights_only=False,
    map_location=torch.device(device),
)
clip_model.eval()
clip_model = clip_model.to(device)
# with torch.no_grad():
#     batch = torch.stack([clip_preprocess(img2tensor(source_img))]).to(device)
#     clip_feature = clip_model.encode_image(batch).cpu().numpy()
#     print(f"CLIP feature has shape: {clip_feature.shape}")

# set up LPIPS
lpips = torch.load(
    "/local/homework/data/hw5/student-data/lpips.pth",
    weights_only=False,
    map_location=torch.device(device),
).to(device)
# with torch.no_grad():
#     img1 = img2tensor(source_img).to(device)
#     img2 = img2tensor(target_img).to(device)
#     lpips_distance = lpips(img1, img2)
#     print(
#         f"LPIPS Distance Between Source and target Image: {lpips_distance.detach().item():.2f}"
#     )

print("=" * 60)
# >>> TESTING PART 2 >>>

# source and target for testing
source_img = Image.open("/local/homework/data/hw5/student-data/adam-sandler.png")
target_img = Image.open("/local/homework/data/hw5/student-data/margaret-cho.png")

import os
base_path = '/local/homework/data/hw5/student-data/val-identities/'
actors = [
    name for name in os.listdir(base_path)
    if os.path.isdir(os.path.join(base_path, name))
]

# tot = cosacc = clipacc = 0
# for _ in range(10):
#         src_actor = random.choice(actors)
#         target_actor = random.choice([a for a in actors if a != src_actor])
#         src_path = base_path + src_actor + "/test/" + f"{random.randint(0, 5)}.png"
#         target_path = base_path + target_actor + "/test/" + f"{random.randint(0, 5)}.png"
#         source_img, target_img = Image.open(src_path), Image.open(target_path)
    

#         # get adversarial image
#         adv_img = part_2(source_img, target_img, cosface_model, clip_model, device)

#         # test adversarial image against facial recognition
#         buffer = io.BytesIO()
#         adv_img.save(buffer, format="PNG")
#         buffer.seek(0)
#         files = {"file": ("test.png", buffer, "image/png")}
#         # print("Part 2 Results")
#         fr_response = requests.post(
#             "http://newt.cs.uchicago.edu/facial-recognition", files=files
#         )
#         print(f"src actor: {src_actor}, target actor: {target_actor}")
#         print(f"\tFacial Recognition Output: {fr_response.json()}")
#         res = fr_response.json()
        
#         tot += 1
#         if res['cosface_prediction'] == target_actor:
#             cosacc += 1
#         if res['clip_prediction'] == target_actor:
#             clipacc += 1
        
        
#         # test adversarial image against VLM
#         buffer = io.BytesIO()
#         adv_img.save(buffer, format="PNG")
#         buffer.seek(0)
#         files = {"file": ("test.png", buffer, "image/png")}
#         vlm_response = requests.post("http://newt.cs.uchicago.edu/vlm", files=files)
#         print(f"\tVLM Output: {vlm_response.json()}")

# print(f"clipacc: {clipacc / tot}")
# print(f"cosacc: {cosacc / tot}")

# <<< END TESTING PART 2 <<<

print("=" * 60)
# >>> TESTING PART 3 >>>

tot = cosacc = clipacc = lpips_budgeted = 0
for _ in range(10):
    src_actor = random.choice(actors)
    target_actor = random.choice([a for a in actors if a != src_actor])
    src_path = base_path + src_actor + "/test/" + f"{random.randint(0, 5)}.png"
    target_path = base_path + target_actor + "/test/" + f"{random.randint(0, 5)}.png"
    source_img, target_img = Image.open(src_path), Image.open(target_path)

    # get adversarial image
    adv_img = part_3(source_img, target_img, cosface_model, clip_model, lpips, device)
    
    lpips_val = lpips(img2tensor(adv_img).to(device), img2tensor(source_img).to(device))
    if lpips_val < 0.07:
        lpips_budgeted += 1

    # test adversarial image against API
    buffer = io.BytesIO()
    adv_img.save(buffer, format="PNG")
    buffer.seek(0)
    files = {"file": ("test.png", buffer, "image/png")}
    print("Part 3 Results")
    fr_response = requests.post(
        "http://newt.cs.uchicago.edu/facial-recognition", files=files
    )
    tot += 1
    res= fr_response.json()
    if res['cosface_prediction'] == target_actor:
        cosacc += 1
    if res['clip_prediction'] == target_actor:
        clipacc += 1
    
print(f"clipacc: {clipacc / tot}")
print(f"cosacc: {cosacc / tot}")   
print(f"lpips budgeted: {lpips_budgeted / tot}") 

# <<< END TESTING PART 3 <<<
