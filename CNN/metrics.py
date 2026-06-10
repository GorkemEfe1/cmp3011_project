import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import random
import kagglehub
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


dataset_path = kagglehub.dataset_download("jonathanoheix/face-expression-recognition-dataset")
VAL_DIR = os.path.join(dataset_path, "images", "validation")


val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
classes = val_dataset.classes 

model = torch.jit.load('CNN\model.pt')
model.eval()

def display_random_prediction():

    idx = random.randint(0, len(val_dataset) - 1)
    image, true_label = val_dataset[idx]
    image_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_idx = torch.max(output, 1)
        
    predicted_label = predicted_idx.item()
    
    display_img = image.squeeze().cpu().numpy()
    display_img = (display_img * 0.5) + 0.5
    
    plt.figure(figsize=(4, 4))
    plt.imshow(display_img, cmap='gray')
    color = 'green' if predicted_label == true_label else 'red'
    plt.title(f"True: {classes[true_label]} | Pred: {classes[predicted_label]}", color=color)
    plt.axis('off')
    plt.show()


def evaluate_model():
    print("Evaluating model and generating confusion matrix. This might take a few seconds...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # --- Calculate and print accuracy ---
    accuracy = accuracy_score(all_labels, all_preds) * 100
    print(f"\n>>> Overall Validation Accuracy: {accuracy:.2f}% <<<\n")
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix (Overall Accuracy: {accuracy:.2f}%)', fontsize=14)
    
    # --- NEW: Save the confusion matrix plot to a file ---
    output_filename = "confusion_matrix.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix image saved successfully as '{output_filename}'")
    
    plt.show()

if __name__ == "__main__":
    display_random_prediction()
    evaluate_model()