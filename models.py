import torchvision.models as models
from torchsummary import summary
import torch.nn as nn



resnet50 = models.resnet50(pretrained=False,progress=True)

class Encoder(nn.Module):
    '''
    Resnet +fc
    '''
    def __init__(self):
        super(Encoder, self).__init__()
        self.resnet50 = resnet50
        self.fc1 = nn.Linear(self.resnet50.fc.in_features, 128)
        #TODO add a L2 normalization like suggested in paper
        #TODO check how to remove the last fc layer in resnet
        #TODO chekc if self.resnet50.fc.in_features is what needed
    def forward(self, x):
        x = self.resnet50(x)
        x = self.fc1(x)
        return x


if __name__ == '__main__':
    encoder_model = Encoder()
    print(encoder_model)
    print("Done")