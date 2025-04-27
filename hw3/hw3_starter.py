# hw3_starter.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms.v2 import JPEG
from torchvision.transforms import Resize, GaussianBlur
import random

# import class functions:
import hw3_utils
from hw3_utils import get_vgg_model, img2tensorVGG, target_pgd_attack, classes, tensor2imgVGG
from model import VGG, load_dataset


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- Part 1: Simple Transformations + Evaluation ---------

# used https://pytorch.org/vision/main/generated/torchvision.transforms.v2.JPEG.html
def jpeg_compression(x: torch.Tensor) -> torch.Tensor:
    """
    Applies JPEG compression to the input image tensor
    """
    # JPEG compression only takes tensors with uint8 type on CPU
    # we scale to [0, 255] for RGB vals before passing into compression
    x = torch.round(x * 255).to(torch.uint8).cpu()
    
    # compress, send to (0, 1) and return
    compress = JPEG(quality=10)
    return (compress(x) / 255).clamp(0, 1)

def image_resizing(x: torch.Tensor) -> torch.Tensor:
    """
    Applies resizing and rescaling to the input image tensor
    """
    scale = 0.3
    _, _, h, w = x.shape
    shrink = Resize((round(h * scale), round(w * scale)))
    grow = Resize(size=x.shape[-2:])
    return grow(shrink(x))

def gaussian_blur(x: torch.Tensor) -> torch.Tensor:
    """
    Applies Gaussian blur to the input image tensor
    """
    blur = GaussianBlur(kernel_size=5, sigma=13)
    return blur(x)

# print helper
def print_results(name, clean_hits, pgd_hits, attack_successes, total):
    print(
        f"----------{name}----------\n"
        f"Benign Classification Success Rate: {clean_hits / total} \n"
        f"PGD Classification Success Rate: {pgd_hits / total} \n"
        f"Attack Success Rate: {attack_successes / total } \n\n"        
    )

def evaluate_transformations():
    """
    Evaluates model accuracy and attack success under transformations
    """
    # declare all counters for each method
    og_clean_hits = og_attack_successes = og_pgd_hits = 0
    compressed_clean_hits = compressed_attack_successes = compressed_pgd_hits = 0
    resized_clean_hits = resized_attack_successes = resized_pgd_hits = 0
    gaussian_clean_hits = gaussian_attack_successes = gaussian_pgd_hits = 0

    trainset = datasets.CIFAR10(root='./data', train=True, download=True)
    
    num_imgs_per_class = 5
    # get `num_imgs_per_class` images per class to test on
    source_img_map = {i: [] for i in range(len(classes))}
    while min(len(source_img_map[i]) for i in source_img_map) < num_imgs_per_class:
        source_img, source_class = trainset[random.randint(0, len(trainset) - 1)]
        if len(source_img_map[source_class]) < num_imgs_per_class:
            source_img_map[source_class].append(source_img)
    
    # init model 
    model = get_vgg_model()
    model.to(device)
    model.load_state_dict(torch.load("./models/vgg16_cifar10_robust.pth", map_location=torch.device(device), weights_only=True))
    model.eval()
    
    for img_class, imgs in source_img_map.items():
        for img in imgs:
            t = img2tensorVGG(img, device)
            
            # compute adversarial example
            target_class = random.choice([i for i in range(len(classes)) if i != img_class])
            ae = img2tensorVGG(target_pgd_attack(img, target_class, model, device), device)
            
            # --- OG ---
            _, output_class = torch.max(model(t), 1)
            _, ae_output_class = torch.max(model(ae), 1)
            # record results
            if output_class == img_class:
                og_clean_hits += 1
            if ae_output_class == target_class:
                og_pgd_hits += 1
            if ae_output_class != img_class:
                og_attack_successes += 1
            
            # --- JPEG compression ---
            # JPEG compression only takes tensors with uint8 type on CPU
            # we scale to [0, 255] for RGB vals before passing into compression
            new_t = jpeg_compression(t)
            new_ae = jpeg_compression(ae)
            _, output_class = torch.max(model(new_t), 1)
            _, ae_output_class = torch.max(model(new_ae), 1)
            # record results
            if output_class == img_class:
                compressed_clean_hits += 1
            if ae_output_class == target_class:
                compressed_pgd_hits += 1
            if ae_output_class != img_class:
                compressed_attack_successes += 1
                
            # --- image resizing ---
            scale = 0.3
            # Applies resizing and rescaling to the input image tensor
            new_t = image_resizing(t)
            new_ae = image_resizing(ae)
            _, output_class = torch.max(model(new_t), 1)
            _, ae_output_class = torch.max(model(new_ae), 1)
            # record results
            if output_class == img_class:
                resized_clean_hits += 1
            if ae_output_class == target_class:
                resized_pgd_hits += 1
            if ae_output_class != img_class:
                resized_attack_successes += 1
                
            # --- Gaussian blur ---
            # Applies Gaussian blur to the input image tensor
            new_t = gaussian_blur(t)
            new_ae = gaussian_blur(ae)
            _, output_class = torch.max(model(new_t), 1)
            _, ae_output_class = torch.max(model(new_ae), 1)
            # record results
            if output_class == img_class:
                gaussian_clean_hits += 1
            if ae_output_class == target_class:
                gaussian_pgd_hits += 1
            if ae_output_class != img_class:
                gaussian_attack_successes += 1
    
    total = len(classes) * num_imgs_per_class

    print_results("OG", og_clean_hits, og_pgd_hits, og_attack_successes, total)
    print_results("Compressed", compressed_clean_hits, compressed_pgd_hits, compressed_attack_successes, total)
    print_results("Resized", resized_clean_hits, resized_pgd_hits, resized_attack_successes, total)
    print_results("Gaussian", gaussian_clean_hits, resized_pgd_hits, gaussian_attack_successes, total)

# --------- Part 2: EOT Attack + Evaluation ---------

def eot_attack(model: nn.Module, x: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
    """
    TODO: Expectation over Transformation PGD attack
    
    Args:
        model: The target model to attack
        x: Input image (clean)
        y_target: Target label 
    
    Returns:
        Adversarial example
    """
    pass

# --------- Part 3: Defensive Distillation + Evaluation ---------

def student_VGG(teacher_path: str = "models/vgg16_cifar10_robust.pth", temperature: float = 20.0) -> None:
    """
    TODO: Trains a student model using knowledge distillation and saves it as 'student_VGG.pth'
    
    Args:
        teacher_path: Path to the pretrained teacher model.
        temperature: Softmax temperature for distillation.
    """
    student = hw3_utils.get_vgg_model().to(device)


    torch.save(student.state_dict(), "models/student_VGG.pth")

    pass

def evaluate_distillation() -> None:
    """
    TODO: Evaluates the student model on clean data and under targeted PGD attack
    
    """
    pass

# --------- Bonus Part: Adaptive Attack ---------

def adaptive_attack(student_model: nn.Module, x: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
    """
    Bonus: Implements a stronger adaptive attack on distilled student model from Part 3

    Args:
        student_model: The distilled student model to attack
        x: Clean input images
        y_target: Target labels

    Returns:
        Adversarial examples
    """
    pass


def main():
    # load data
    train_loader, test_loader = load_dataset()

    # use gpu device if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load teacher model
    model = VGG('VGG16').to(device)
    model.load_state_dict(torch.load("models/vgg16_cifar10_robust.pth", map_location=device))

    # PART 1: Evaluate simple defenses
    # TODO: Evaluate baseline accuracy on clean and PGD-attacked images

    # TODO: Apply each of the three defenses (JPEG, resize, blur)

    # TODO: Evaluate classification accuracy + attack success rate for each defense
    
    evaluate_transformations() # everything is here oops

    # PART 2: EOT Attack
    # TODO: Implement and run your EOT attack

    # TODO: Evaluate model accuracy under EOT attack (with and without defenses)

    # TODO: Save your results and write your short analysis separately


    # PART 3: Distillation Defense
    # TODO: Train a student model via distillation using the teacher model

    # TODO: Save your model as 'student_VGG.pth'
    
    # TODO: Evaluate student model accuracy on clean and PGD-attacked images
    
    # TODO: Save or print results and include your writeup separately

    
    # BONUS: Adaptive attack (optional)
    # TODO: Implement a new attack to beat the distilled model
        
    # TODO: Evaluate and report attack success rate



if __name__ == "__main__":
    main()
