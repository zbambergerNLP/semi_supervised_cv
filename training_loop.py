import torch
import torch.nn as nn
import random
import numpy as np

import consts
import models
import matplotlib.pyplot as plt
from augment import load_imagenette
from data_loader import ImagenetteDataset, Rescale, RandomCrop,ToTensor
from torch.utils.data import DataLoader
from torchvision import transforms, utils

# train function
def train(encoder,m_encoder, train_loader, epochs=20, lr=0.001, momentum=0.9, t = 0.07, m = 0.999):

    dict = {} #keys are the ouptut
    queue_dict = [] #will add in FIFO order keys of mini batches
    loss_fn = nn.CrossEntropyLoss

    #the optimization is done only to the encoder weights and not to the momentum encoder
    optimizer = torch.optim.SGD(encoder.parameters(), lr=lr, momentum=momentum)


    for epoch in range(epochs):
        print(f'start epoch {epoch}')

        for x in train_loader:
            x = x.double()
            optimizer.zero_grad()
            #TODO call augmentaion function

            q = encoder.forward(x) #queries NxC
            k = m_encoder.forward(x) #keys NxC
            k = k.detach() # no gradients to keys

            #positive logits Nx1
            q = torch.flatten(q, start_dim=1)
            q = torch.unsqueeze(q, dim=1)
            k = torch.flatten(k, start_dim=1)
            k = torch.unsqueeze(k, dim=2)
            l_pos = (q @ k).squeeze(dim = 2)

            if len(queue_dict) >0 :
                queue_view = torch.concat([queue_dict[x] for x in queue_dict], 1)

                #negative logits NxK
                l_neg = q @ queue_view
                logits = torch.concat((l_pos, l_neg), dim=0)  # Nx(k+1)
            else:
                logits = l_pos

            labels = torch.zeros(l_pos.shape[0])
            labels[0] = 1 #first logit is  from the same image so the label is 1

            loss = loss_fn(logits / t , labels)
            #SGD update query network
            loss.backward()
            optimizer.step() #update only encoder parmas and not m_encoder params #TODO make sure that this is what happens

            #momentum update key network
            m_encoder_state_dict = m_encoder.state_dict()
            encoder_state_dict = encoder.state_dict()

            for m_name, m_param, name, param in zip(m_encoder_state_dict.items(), encoder_state_dict.item()):
                # Transform the parameter as required.
                transformed_param = m * m_param  + (1-m) * param

                # Update the parameter.
                m_encoder_state_dict[m_name].copy_(transformed_param)

            #enqueue queue and queue dict
            queue_dict.append(k)

            #dequeue the oldest minibatch
            queue_dict.pop(0)





# evalutate function top-1 accuracy with linear evaluation

# data loaders function

# save model

# load model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    seed = 1
    epochs = 2
    lr = 0.1
    momentum = 0.9
    bs = 8

    set_seed(seed)

    #train_loader = load_imagenette() #TODO add batch size
    imagenette_dataset = ImagenetteDataset(csv_file=consts.csv_filename,
                                        root_dir=consts.image_dir,
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))
    train_loader = DataLoader(imagenette_dataset, batch_size=bs)

    # fig = plt.figure()
    #
    # for i in range(len(imagenette_dataset)):
    #     sample = imagenette_dataset[i]
    #
    #     print(i, sample.shape)
    #
    #     ax = plt.subplot(1, 4, i + 1)
    #     plt.tight_layout()
    #     ax.set_title('Sample #{}'.format(i))
    #     ax.axis('off')
    #
    #
    #     if i == 3:
    #         plt.show()
    #         break

    encoder = models.Encoder().double()
    m_endcoder  = models.Encoder().double()

    # train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    m_endcoder.to(device)

    train(encoder, m_endcoder, train_loader, epochs=epochs,lr = lr,  momentum= momentum)

