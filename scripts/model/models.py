import torch
import torch.nn as nn


class Alexnet(nn.Module):
    def __init__(self, dropout: float = 0.5) -> None:
        super(Alexnet, self).__init__()

        # load pretrained Alexnet from github
        # convolution layers
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "alexnet", pretrained=True
        )
        self.model.classifier[-1] = nn.Linear(in_features=4096, out_features=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


class Mobilenet(nn.Module):
    def __init__(self) -> None:
        super(Mobilenet, self).__init__()

        # Load mobilenet v3
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "mobilenet_v3_large", pretrained=True
        )
        self.model.classifier[-1] = nn.Linear(1280, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


class ResNet18(nn.Module):
    def __init__(self) -> None:
        super(ResNet18, self).__init__()

        # load pretrained architecture from pytorch
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet18", pretrained=True
        )
        self.model.fc = nn.Linear(
            512, 2
        )  # * torchvision.models.resnet.BasicBlock.expansion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


class ResNet34(nn.Module):
    def __init__(self) -> None:
        super(ResNet34, self).__init__()

        # load pretrained architecture from pytorch
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet34", pretrained=True
        )
        self.model.fc = nn.Linear(
            512, 2
        )  # * torchvision.models.resnet.BasicBlock.expansion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


class ResNet50(nn.Module):
    def __init__(self) -> None:
        super(ResNet50, self).__init__()

        # load pretrained architecture from pytorch
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet50", pretrained=True
        )
        self.model.fc = nn.Linear(2048, 2)  # for binary classification

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x
