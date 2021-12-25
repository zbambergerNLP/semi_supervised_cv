import torch
import torchvision.models as models
import torch.nn as nn

resnet50 = models.resnet50(pretrained=True, progress=True)


class Encoder(nn.Module):
    """
    Resnet +fc
    """
    def __init__(self, end_num_of_features):
        self.final_num_of_features = end_num_of_features
        super(Encoder, self).__init__()
        self.resnet50 = nn.Sequential(*(list(resnet50.children())[:-1]))
        self.fc1 = nn.Linear(1, self.final_num_of_features)
        self.relu1 = nn.ReLU()
        # TODO check if 1 is the size needed in fc1 layer

    def forward(self, x):
        x = self.resnet50(x)
        x = self.fc1(x)
        x = self.relu1(x)
        # Normalize by the L2 norm
        l2_norm = torch.linalg.norm(x, dim=(1, 2, 3))
        x = (x.T / l2_norm).T
        return x


if __name__ == '__main__':
    encoder_model = Encoder(128)
    print(encoder_model)
    x_demo = torch.rand((4, 3, 224, 224))
    y = encoder_model(x_demo)
    print("Done")
