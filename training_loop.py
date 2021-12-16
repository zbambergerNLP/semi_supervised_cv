import torch
import torch.nn as nn

# train function
def train(encoder,m_encoder, train_loader, epochs=20, lr=0.001, momentum=0.9):

    loss_fn = nn.CrossEntropyLoss

    #the optimization is done only to the encoder weights and not to the momentum encoder
    optimizer = torch.optim.SGD(encoder.parameters(), lr=lr, momentum=momentum)

    for epoch in range(epochs):
        print(f'start epoch {epoch}')

        for x in train_loader:
            #TODO call augmentaion function

            x = encoder.forward(x)

# evalutate function top-1 accuracy with linear evaluation

# data loaders function

# save model

# load model
