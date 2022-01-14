import torch
import torchvision.models as models
import torch.nn as nn

import consts

resnet50 = models.resnet50(pretrained=False, progress=True)


class Encoder(nn.Module):
    """
    Resnet +fc
    """
    def __init__(self, end_num_of_features):
        """Construct a MoCo encoder composed of a resent50 base, and then followed by a fully connected head.
        The fully connected head of the model is composed of one linear transformation to the same (hidden)
        dimensionality, followed by a linear projection to `final_num_of_features` dimensions. Each of these linear
        layers is followed by a ReLU non-linearity.

        :param end_num_of_features: The dimensionality of the final layer of a sequential head at the end of a
            resnet50 base model.
        """
        self.final_num_of_features = end_num_of_features
        super(Encoder, self).__init__()
        self.resnet50 = nn.Sequential(*(list(resnet50.children())[:-1]))

        # MoCo v2 utilizes two fully connected layers as opposed to one.
        self.fc1 = nn.Sequential(nn.Linear(consts.HIDDEN_REPRESENTATION_DIM, consts.HIDDEN_REPRESENTATION_DIM),
                                 nn.ReLU(),
                                 nn.Linear(consts.HIDDEN_REPRESENTATION_DIM, self.final_num_of_features))
        self.non_linear_func = nn.ReLU()

    def forward(self, x, device):
        """Perform a forward pass through the MoCo encoder with some input batch `x`.

        :param x: A batch of image input tensors of shape [batch_size, 224, 224, 3]
        :param device: A torch.device.Device instance representing the device on which we perform training.
        """
        with device as d:
            x = self.resnet50(x.to(d))
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
