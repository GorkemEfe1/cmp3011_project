import torch
import torchvision.models as models

# Load a pre-trained ResNet model
model = models.resnet50(pretrained=True)

# Example to print the names of each layer block
for name, layer in model.named_children():
    print(name)