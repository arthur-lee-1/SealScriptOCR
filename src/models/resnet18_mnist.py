import torch
import torch.nn as nn
from torchvision import models


def resnet18_single_channel(num_classes=10, pretrained=False):
    """Return a ResNet18 adapted for single-channel input."""
    model = models.resnet18(pretrained=pretrained)
    # replace conv1 to accept 1 channel
    in_channels = 1
    old_conv = model.conv1
    new_conv = nn.Conv2d(in_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                         stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias is not None)

    if pretrained:
        # average weights across existing 3 channels
        with torch.no_grad():
            old_weights = old_conv.weight.clone()
            if old_weights.size(1) == 3:
                new_weights = old_weights.mean(dim=1, keepdim=True)
                new_conv.weight.copy_(new_weights)
    else:
        # default initialization
        pass

    model.conv1 = new_conv

    # replace fc
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


if __name__ == '__main__':
    m = resnet18_single_channel(num_classes=10, pretrained=False)
    x = torch.randn(2, 1, 28, 28)
    y = m(x)
    print(y.shape)
