import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# Brightness statistics that the pretrained encoder was trained with
IMAGENET_MEAN = 0.449
IMAGENET_STD = 0.226


class SolarResNetUNet(nn.Module):
    """
    U-Net with a ResNet-34 encoder (pretrained on ImageNet).

    Input:  1-channel image with values in [0, 1]
    Output: 2 channels (0 = background, 1 = filament)
    """

    def __init__(self, pretrained=True, num_classes=2):
        super().__init__()

        # AMD's MIOpen library cannot compile its batch-norm kernel on this
        # PC, so PyTorch is told to use its own built-in kernels instead.
        torch.backends.cudnn.enabled = False

        self.net = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=1,
            classes=num_classes
        )

    def forward(self, x):
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return self.net(x)


def build_model_from_state(state):
    """
    Create the right kind of model for a set of saved weights,
    load the weights into it, and return it.
    """

    if any(key.startswith("net.encoder") for key in state):
        # Pretrained-encoder model (weights come from the checkpoint)
        model = SolarResNetUNet(pretrained=False)
    else:
        # The original small U-Net
        from src.model import SolarUNet

        num_classes = state["output.weight"].shape[0]
        model = SolarUNet(num_classes=num_classes)

    model.load_state_dict(state)

    return model