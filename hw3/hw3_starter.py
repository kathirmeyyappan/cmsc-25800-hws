# hw3_starter.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# import class functions:
import hw3_utils
from hw3_utils import target_pgd_attack
from model import VGG, load_dataset


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- Part 1: Simple Transformations + Evaluation ---------

def jpeg_compression(x: torch.Tensor) -> torch.Tensor:
    """
    Applies JPEG compression to the input image tensor
    """
    pass

def image_resizing(x: torch.Tensor) -> torch.Tensor:
    """
    Applies resizing and rescaling to the input image tensor
    """
    pass

def gaussian_blur(x: torch.Tensor) -> torch.Tensor:
    """
    Applies Gaussian blur to the input image tensor
    """
    pass

def evaluate_transformations():
    """
    Evaluates model accuracy and attack success under transformations
    """
    pass

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
