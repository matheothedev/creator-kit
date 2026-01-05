import torch
import torchvision.models as models
import torch.nn as nn

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)
torch.save(model.state_dict(), 'resnet18_cifar10.pt')
print(f'ResNet18 saved: {sum(p.numel() for p in model.parameters())} params (~44MB)')