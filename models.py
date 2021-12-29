import torch
import torchvision.models as models
import torch.nn as nn

import consts

resnet50 = models.resnet50(pretrained=True, progress=True)


class Encoder(nn.Module):
    """
    Resnet +fc
    """
    def __init__(self, end_num_of_features):
        self.final_num_of_features = end_num_of_features
        super(Encoder, self).__init__()
        self.resnet50 = nn.Sequential(*(list(resnet50.children())[:-1]))
        #MOCO v2 changes
        self.fc1 = nn.Sequential(nn.Linear(consts.HIDDEN_REPRESENTATION_DIM, consts.HIDDEN_REPRESENTATION_DIM),
                                 nn.ReLU(),
                                 nn.Linear(consts.HIDDEN_REPRESENTATION_DIM, self.final_num_of_features))
        self.non_linear_func = nn.ReLU()

    def forward(self, x):
        x = self.resnet50(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.non_linear_func(x)
        # Normalize by the L2 norm
        l2_norm = torch.linalg.norm(x, dim=1)
        x = (x.T / l2_norm).T
        return x


if __name__ == '__main__':
    encoder_model = Encoder(128)
    print(encoder_model)
    x_demo = torch.rand((4, 3, 224, 224))
    y = encoder_model(x_demo)
    print("Done")
