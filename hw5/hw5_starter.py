"""
Starter file for HW5, CMSC 25800 Spring 2025
"""

from typing import List, Tuple

import numpy as np
import torch
import lpips
from open_clip import CLIP
from PIL import Image

from utils import IResNet as CosFace
from utils import cosface_preprocess, clip_preprocess  # for parts 1, 2, and 3


def part_1_cosface(
    train_data: List[Tuple[str, str]],
    test_data: List[Tuple[str, str]],
    model: CosFace,
    device: str | torch.device,
) -> List[str]:
    # each item in `train_data` and `test_data` is a Tuple where the first string is an image filepath, and the second string is the identity of the person in the image
    return [None for _ in test_data]


def part_1_clip(
    train_data: List[Tuple[str, str]],
    test_data: List[Tuple[str, str]],
    model: CLIP,
    device: str | torch.device,
) -> Image:
    # each item in `train_data` and `test_data` is a Tuple where the first string is an image filepath, and the second string is the identity of the person in the image
    return [None for _ in test_data]


def part_2(
    img: Image,
    target_img: Image,
    cosface_model: CosFace,
    clip_model: CLIP,
    device: str | torch.device,
) -> Image:
    return img


def part_3(
    img: Image,
    target_img: Image,
    cosface_model: CosFace,
    clip_model: CLIP,
    lpips: lpips.LPIPS,
    device: str | torch.device,
) -> Image:
    return img
