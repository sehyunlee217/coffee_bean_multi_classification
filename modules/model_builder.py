import torch, torchvision
from torch import nn


def build_effnet_v2_s(device: torch.device):
    # weight & model initialization
    effnet_v2_s_weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
    effnet_v2_s = torchvision.models.efficientnet_v2_s(weights=effnet_v2_s_weights)
    effnet_v2_s.name = "effnet_v2_s"

    # Freeze params
    for params in effnet_v2_s.parameters():
        params.requires_grad = False

    # Edit Classifiers
    effnet_v2_s.classifier = torch.nn.Sequential(
        nn.Dropout(p=0, inplace=True),
        nn.Linear(in_features=1280,
                  out_features=4,
                  bias=True).to(device)
    )

    return effnet_v2_s, effnet_v2_s_weights

def build_effnetb1(device: torch.device):
    # weight & model initialization
    effnetb1_weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT
    effnetb1 = torchvision.models.efficientnet_b1(weights=effnetb1_weights)
    effnetb1.name = "effnetb1"

    # Freeze params
    for params in effnetb1.parameters():
        params.requires_grad = False

    # Edit Classifiers
    effnetb1.classifier = torch.nn.Sequential(
        nn.Dropout(p=0, inplace=True),
        nn.Linear(in_features=1280,
                  out_features=4,
                  bias=True).to(device)
    )

    return effnetb1, effnetb1_weights
