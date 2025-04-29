# hw3_starter.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms.v2 import JPEG
from torchvision.transforms import Resize, GaussianBlur
import torch.nn.functional as F
import random
import numpy as np

# import class functions:
import hw3_utils
from hw3_utils import get_vgg_model, img2tensorVGG, softmax_with_temperature, target_pgd_attack, classes, tensor2imgVGG
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
    scale = 0.35
    _, _, h, w = x.shape
    shrink = Resize((round(h * scale), round(w * scale)))
    grow = Resize(size=x.shape[-2:])
    return grow(shrink(x))

def gaussian_blur(x: torch.Tensor) -> torch.Tensor:
    """
    Applies Gaussian blur to the input image tensor
    """
    blur = GaussianBlur(kernel_size=5, sigma=10)
    return blur(x)

# print helper
def print_results(name, clean_classification_success, pgd_classification_success, attack_successes, total):
    print(
        f"----------{name}----------\n"
        f"Benign Classification Success Rate: {clean_classification_success / total} \n"
        f"PGD Classification Success Rate: {pgd_classification_success / total} \n"
        f"Attack Success Rate: {attack_successes / total } \n"        
    )

def evaluate_transformations():
    """
    Evaluates model accuracy and attack success under transformations
    """
    # declare all counters for each method
    og_clean_classification_success = og_attack_successes = og_pgd_classification_success = 0
    compressed_clean_classification_success = compressed_attack_successes = compressed_pgd_classification_success = 0
    resized_clean_classification_success = resized_attack_successes = resized_pgd_classification_success = 0
    gaussian_clean_classification_success = gaussian_attack_successes = gaussian_pgd_classification_success = 0

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
            
            # OG
            _, output_class = torch.max(model(t), 1)
            _, ae_output_class = torch.max(model(ae), 1)
            # record results
            if output_class == img_class:
                og_clean_classification_success += 1
            if ae_output_class == img_class:
                og_pgd_classification_success += 1
            if ae_output_class == target_class:
                og_attack_successes += 1
            
            # jpeg compression
            new_t = jpeg_compression(t)
            new_ae = jpeg_compression(ae)
            _, output_class = torch.max(model(new_t), 1)
            _, ae_output_class = torch.max(model(new_ae), 1)
            # record results
            if output_class == img_class:
                compressed_clean_classification_success += 1
            if ae_output_class == img_class:
                compressed_pgd_classification_success += 1
            if ae_output_class == target_class:
                compressed_attack_successes += 1
                
            # resizing
            new_t = image_resizing(t)
            new_ae = image_resizing(ae)
            _, output_class = torch.max(model(new_t), 1)
            _, ae_output_class = torch.max(model(new_ae), 1)
            # record results
            if output_class == img_class:
                resized_clean_classification_success += 1
            if ae_output_class == img_class:
                resized_pgd_classification_success += 1
            if ae_output_class == target_class:
                resized_attack_successes += 1
                
            # gaussian blur
            new_t = gaussian_blur(t)
            new_ae = gaussian_blur(ae)
            _, output_class = torch.max(model(new_t), 1)
            _, ae_output_class = torch.max(model(new_ae), 1)
            # record results
            if output_class == img_class:
                gaussian_clean_classification_success += 1
            if ae_output_class == img_class:
                gaussian_pgd_classification_success += 1
            if ae_output_class == target_class:
                gaussian_attack_successes += 1
    
    total = len(classes) * num_imgs_per_class
    
    print(f"\n---------- PART 1 ----------\n")
    print_results("OG", og_clean_classification_success, og_pgd_classification_success, og_attack_successes, total)
    print_results("Compressed", compressed_clean_classification_success, compressed_pgd_classification_success, compressed_attack_successes, total)
    print_results("Resized", resized_clean_classification_success, resized_pgd_classification_success, resized_attack_successes, total)
    print_results("Gaussian", gaussian_clean_classification_success, resized_pgd_classification_success, gaussian_attack_successes, total)

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
    epsilon = 8/255
    tensor_max = 1
    tensor_min = 0
    lr_initial = 0.01
    max_iter = 200

    modifier = torch.zeros_like(x, requires_grad=True)

    # target_label = torch.tensor([target_class], dtype=torch.long).to(device)
    loss_fn = nn.CrossEntropyLoss()

    for i in range(max_iter):
        adv_tensor = torch.clamp(x + modifier, tensor_min, tensor_max)
        compressed_output = model(jpeg_compression(adv_tensor))
        resized_output = model(image_resizing(adv_tensor))
        gaussian_output = model(gaussian_blur(adv_tensor))
        loss = torch.mean(torch.stack([
            loss_fn(compressed_output, y_target), 
            loss_fn(resized_output, y_target), 
            loss_fn(gaussian_output, y_target)
        ]), dim=0)

        model.zero_grad()
        if modifier.grad is not None:
            modifier.grad.zero_()
        loss.backward()

        grad = modifier.grad
        modifier = modifier - lr_initial * grad.sign()
        modifier = torch.clamp(modifier, min=-epsilon, max=epsilon).detach().requires_grad_(True)

        if i % (max_iter // 10) == 0:
            votes = torch.stack([
                torch.argmax(compressed_output, dim=1),
                torch.argmax(resized_output, dim=1),
                torch.argmax(gaussian_output, dim=1)
            ])
            prediction = votes[0] if (votes[0] == votes).all() else -1 # don't even try to compare unless everyone agrees
            
            # print(f"votes={votes}")
            # print(f"prediction={prediction}")
            # print(y_target)


            # Optional: uncomment to print loss values:
            # print(f"step: {i} | loss: {loss.item():.4f} | pred class: {classes[pred_class]}")

            if prediction == y_target.item():
                break

    adv_tensor = torch.clamp(x + modifier, tensor_min, tensor_max)
    return tensor2imgVGG(adv_tensor)


def evaluate_new_attack():
    """
    Evaluates model accuracy and attack success under transformations
    """
    # declare all counters for each method
    pgd_attack_hits = eot_pgd_attack_hits = pgd_classification_hits = eot_pgd_classification_hits = 0

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
    
    transformations = [jpeg_compression, image_resizing, gaussian_blur]
    
    for img_class, imgs in source_img_map.items():
        for img in imgs:
            t = img2tensorVGG(img, device)
            
            # compute adversarial examples
            target_class = random.choice([i for i in range(len(classes)) if i != img_class])
            ae = target_pgd_attack(img, target_class, model, device), device
            eot_ae = eot_attack(model, t, torch.tensor([target_class])), device
            
            # stuff
            random_transformation = random.choice(transformations)
            # new_t = random_transformation(t)
            new_ae = random_transformation(ae)
            new_eot_ae = random_transformation(eot_ae)
            # _, output_class = torch.max(model(new_t), 1)
            _, ae_output_class = torch.max(model(new_ae), 1)
            _, eot_ae_output_class = torch.max(model(new_eot_ae), 1)
            # record results
            if ae_output_class == img_class:
                pgd_classification_hits += 1
            if eot_ae_output_class == img_class:
                eot_pgd_classification_hits += 1
            if ae_output_class == target_class:
                pgd_attack_hits += 1
            if eot_ae_output_class == target_class:
                eot_pgd_attack_hits += 1
    
    total = len(classes) * num_imgs_per_class

    print(
        f"\n---------- PART 2 ----------\n"
        f"PGD Classification Success Rate: {pgd_classification_hits / total} \n"
        f"EOT PGD Classification Success Rate: {eot_pgd_classification_hits / total} \n" 
        f"PGD Attack Success Rate: {pgd_attack_hits / total} \n"
        f"EOT PGD Attack Success Rate: {eot_pgd_attack_hits / total} \n"   
    )
    


# --------- Part 3: Defensive Distillation + Evaluation ---------

# from hw1
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

# from hw1
def train_model(train_loader, val_loader, model, parent, temp):
    
    learning_rate = 0.005
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 20
    assert(num_epochs <= 50)
    
    loss_list = []
    accuracy_list = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = []
        for images, _ in train_loader:
                        
            images = images.to(device)
            
            
            # softmax stuff w temp for loss calc
            student_probs = softmax_with_temperature(model(images), temp)
            with torch.no_grad():
                teacher_probs = softmax_with_temperature(parent(images), temp)
            # get loss
            loss = torch.nn.MSELoss() # cmp loss on tensor to tensor
            loss_val = loss(student_probs, teacher_probs)
            
            # back prop and update params
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            epoch_loss.append(loss_val.data.item())
        
        accuracy = evaluate_model(model, val_loader)
        
        loss_list.append(np.mean(epoch_loss))
        accuracy_list.append(accuracy)
        print(f"Epoch: {epoch}, Training Loss: {loss_list[-1]:.2f}, Validation Accuracy: {accuracy*100:.2f}%")


def student_VGG(teacher_path: str = "models/vgg16_cifar10_robust.pth", temperature: float = 20.0) -> None:
    """
    TODO: Trains a student model using knowledge distillation and saves it as 'student_VGG.pth'
    
    Args:
        teacher_path: Path to the pretrained teacher model.
        temperature: Softmax temperature for distillation.
    """
    # load teacher model
    parent = VGG('VGG16').to(device)
    parent.load_state_dict(torch.load("models/vgg16_cifar10_robust.pth", map_location=device))
    
    # student model
    train_loader, val_loader = load_dataset()
    student = hw3_utils.get_vgg_model().to(device)
    train_model(train_loader, val_loader, student, parent, temperature)
    torch.save(student.state_dict(), "models/student_VGG.pth")

def evaluate_distillation() -> None:
    """
    TODO: Evaluates the student model on clean data and under targeted PGD attack
    
    """
    
    # load models
    og_model = VGG('VGG16').to(device)
    og_model.load_state_dict(torch.load("models/vgg16_cifar10_robust.pth", map_location=device))
    model = VGG('VGG16').to(device)
    model.load_state_dict(torch.load("models/student_VGG.pth", map_location=device)) # eventually this will be the actual student model - rn it's a copy to eval
    
    # declare all counters for each method
    pgd_hits = og_hits = pgd_attack_success_og_model = pgd_attack_success_new_model = 0

    _, val_loader = load_dataset()
    evals = []
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        for img, lbl in zip(images, labels):
            evals.append((lbl, tensor2imgVGG(img)))
    print(len(evals))
        
    for img_class, img in evals:
        og_t = img2tensorVGG(img, device)
        
        # compute adversarial examples
        target_class = random.choice([i for i in range(len(classes)) if i != img_class])
        ae_t = img2tensorVGG(target_pgd_attack(img, target_class, model, device), device)

        _, ae_output_class = torch.max(model(ae_t), 1)
        _, og_output_class = torch.max(model(og_t), 1)
        _, ae_output_class_og_model = torch.max(og_model(ae_t), 1)
        # record results
        if ae_output_class == img_class:
            pgd_hits += 1
        if og_output_class == img_class:
            og_hits += 1
        if ae_output_class == target_class:
            pgd_attack_success_new_model += 1
        if ae_output_class_og_model == target_class:
            pgd_attack_success_og_model += 1
                
    
    total = len(classes) * num_imgs_per_class

    print(
        f"\n---------- PART 3 ----------\n"
        f"Original Image Classification Success Rate: {og_hits / total} \n"
        f"PGD Image Classification Success Rate: {pgd_hits / total} \n"
        f"PGD Attack Success Rate: {pgd_attack_success_new_model / total} \n"  
        f"PGD Attack Success Rate (Parent Model): {pgd_attack_success_og_model / total} \n"   
    )
    
    
    
    

# # --------- Bonus Part: Adaptive Attack ---------

# def adaptive_attack(student_model: nn.Module, x: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
#     """
#     Bonus: Implements a stronger adaptive attack on distilled student model from Part 3

#     Args:
#         student_model: The distilled student model to attack
#         x: Clean input images
#         y_target: Target labels

#     Returns:
#         Adversarial examples
#     """
#     pass


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
    
    # evaluate_transformations() # everything is here oops

    # PART 2: EOT Attack
    # TODO: Implement and run your EOT attack

    # TODO: Evaluate model accuracy under EOT attack (with and without defenses)

    # TODO: Save your results and write your short analysis separately
    
    # evaluate_new_attack()


    # PART 3: Distillation Defense
    # TODO: Train a student model via distillation using the teacher model

    # TODO: Save your model as 'student_VGG.pth'
    
    # TODO: Evaluate student model accuracy on clean and PGD-attacked images
    
    # TODO: Save or print results and include your writeup separately
    
    evaluate_distillation()

    
    # BONUS: Adaptive attack (optional)
    # TODO: Implement a new attack to beat the distilled model
        
    # TODO: Evaluate and report attack success rate



if __name__ == "__main__":
    main()
