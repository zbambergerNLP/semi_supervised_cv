import torch
import torchvision.models as models
import torch.nn as nn

resnet50 = models.resnet50(pretrained=True, progress=True)


class Encoder(nn.Module):
    """
    Resnet +fc
    """
    def __init__(self):
        super(Encoder, self).__init__()
        # self.resnet50 = resnet50
        self.resnet50 = nn.Sequential(*(list(resnet50.children())[:-1]))
        self.fc1 = nn.Linear(1, 128)
        # TODO chekc if 1 is the size needed in fc1 layer

    def forward(self, x):
        x = self.resnet50(x)
        x = self.fc1(x)
        # Normlaize by the L2 norm
        l2_norm = torch.linalg.norm(x, dim=(1, 2, 3))
        x = (x.T / l2_norm).T
        return x


if __name__ == '__main__':
    encoder_model = Encoder()
    x_demo = torch.rand((4, 3, 32, 32))
    y = encoder_model(x_demo)
    print("Done")
