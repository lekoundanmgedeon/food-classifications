import torch.nn as nn
import torchvision.models as models

def simple_cnn(num_classes):

    model = nn.Sequential(
        nn.Conv2d(3,32,3),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32,64,3),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Flatten(),
        nn.Linear(64*54*54,128),
        nn.ReLU(),
        nn.Linear(128,num_classes)
    )

    return model


def resnet18_transfer(num_classes):

    model = models.resnet18(pretrained=True)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model